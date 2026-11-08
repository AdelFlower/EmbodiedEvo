"""Simulator worker with scaled gripper drives and contact-force observations."""

import argparse
import contextlib
import json
import re
import socket
import traceback
from pathlib import Path

import numpy as np

from integrations.friction_rl.control import decode_action
from orchestrator.files import write_json
from orchestrator.transport import receive, send
from simulators.dynamics_native import beneath
from simulators.dynamics_rl import DynamicsRLTask, json_value


class RLWorker:
    def __init__(self, env, run_dir):
        self.env = env
        self.run_dir = Path(run_dir).resolve()
        self.episode_dir = None
        self.state = None
        self.transitions = self.commands = None
        self.finished = False
        self.episode_initialized = False
        self.failed = False

    def _record(self, stream, value):
        stream.write(json.dumps(json_value(value), allow_nan=False) + "\n")
        stream.flush()

    def finish_episode(self, reason):
        if self.episode_dir is None or self.finished:
            return
        self.env.close_videos()
        if self.episode_initialized:
            write_json(self.episode_dir / "final-state.json", self.env.host_snapshot())
        write_json(self.episode_dir / "result.json", {
            "success": bool(self.env.success) and self.episode_initialized and not self.failed,
            "complete": self.episode_initialized and not self.failed and bool(self.env.done),
            "terminated": self.episode_initialized and bool(self.env.terminated),
            "truncated": self.episode_initialized and bool(self.env.truncated),
            "steps": self.env.episode_steps if self.episode_initialized else 0,
            "reason": reason,
            "videos": dict(self.env.video_paths),
        })
        self.finished = True

    def _close_records(self):
        for stream in (self.transitions, self.commands):
            if stream is not None:
                stream.close()
        self.transitions = self.commands = None

    def reset(self, command):
        self.finish_episode("host_reset")
        self._close_records()
        episode = Path(command["episode_dir"]).resolve()
        if not episode.is_relative_to(self.run_dir) or episode == self.run_dir:
            raise ValueError("Episode evidence must live beneath the explicit run directory")
        episode.mkdir(parents=True, exist_ok=False)
        self.episode_dir = episode
        self.finished = False
        self.failed = False
        self.episode_initialized = False
        write_json(episode / "scene.json", command["scene"])
        self.commands = (episode / "commands.jsonl").open("w")
        self.transitions = (episode / "transitions.jsonl").open("w")
        self._record(self.commands, command)
        self.env.reset_episode(command["scene"], episode)
        self.episode_initialized = True
        self.state = self.env.rl_state()
        write_json(episode / "initial-state.json", self.env.host_snapshot())
        result = self.reply(native_reward=0.0)
        self._record(self.transitions, {"reset": True, **result})
        return result

    def reply(self, *, native_reward):
        return {
            "ok": True,
            "native_reward": float(native_reward),
            "terminated": bool(self.env.terminated),
            "truncated": bool(self.env.truncated),
            "info": {
                "success": bool(self.env.success),
                "steps": self.env.episode_steps,
                "videos": dict(self.env.video_paths),
            },
            "state": self.state,
        }

    def step(self, command):
        if self.state is None:
            raise RuntimeError("Reset before stepping the RL simulator")
        if self.finished or self.env.done:
            raise RuntimeError("The episode has ended; no additional physics action is allowed")
        self._record(self.commands, command)
        action = np.asarray(command["action"], dtype=np.float32)
        native_action = decode_action(action, self.env.left_pose, self.env.left_grip)
        _, native_reward, terminated, truncated, _ = self.env.control_step(native_action)
        next_state = self.env.rl_state()
        self.state = next_state
        result = self.reply(native_reward=native_reward)
        result["applied_action"] = json_value(native_action)
        self._record(self.transitions, {"action": action.tolist(), **result})
        if terminated or truncated:
            self.finish_episode("native_episode_end")
        return result

    def inspect(self):
        if self.state is None:
            return {"ok": True, "ready": True, "state": None}
        return {"ok": True, "state": self.state, "snapshot": self.env.host_snapshot()}

    def close(self):
        try:
            self.finish_episode("host_close")
        finally:
            self._close_records()


def run_worker(stream, run_dir, *, parameters, mode="state", builder=None, worker_class=RLWorker):
    env = worker = None
    run_dir = Path(run_dir)
    try:
        factory = DynamicsRLTask if builder is None else builder
        env = factory(mode=mode, settings=parameters)
        worker = worker_class(env, run_dir)
        send(stream, {"ok": True, "ready": True, "mode": mode,
                      "parameters": parameters})
        while True:
            command = receive(stream)
            operation = command.get("op")
            if operation == "close":
                worker.close()
                send(stream, {"ok": True, "closed": True})
                return 0
            if operation not in {"reset", "step", "inspect"}:
                raise ValueError(f"Unknown RL worker operation: {operation}")
            result = (
                worker.inspect() if operation == "inspect" else getattr(worker, operation)(command)
            )
            send(stream, json_value(result))
    except BaseException as exc:
        error = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "classification": "unclassified",
        }
        write_json(run_dir / "worker-error.json", error)
        if worker is not None and worker.episode_dir is not None:
            worker.failed = True
            write_json(worker.episode_dir / "worker-error.json", error)
        with contextlib.suppress(OSError, ValueError):
            send(stream, {"ok": False, "fatal": True, "error": error})
        return 1
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                write_json(run_dir / "evidence-close-error.json", {
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
        if env is not None:
            env.close()


class GripperForceTask(DynamicsRLTask):
    """Apply a fixed multiplier to nominal gripper stiffness and force limits."""

    def __init__(self, *, mode, settings, gripper_scale):
        self.gripper_scale, self.grip_force_command = gripper_scale, None
        self.nominal_drives = {}
        super().__init__(mode=mode, settings=settings)

    def reset_episode(self, scene, episode_dir):
        # An early-stopped episode can end in force mode; never carry it into native reset.
        if self.grip_force_command:
            self.set_grip_force(0.0)
        result = super().reset_episode(scene, episode_dir)
        manager = self.native.robot_manager
        for robot, articulation in zip(manager.robot_list, manager.robot_key, strict=True):
            if robot.arm_name == "right_arm":
                self.right_gripper = robot, articulation
            ids = robot.gripper_joint_indices
            view = articulation.root_physx_view
            if robot.arm_name not in self.nominal_drives:
                self.nominal_drives[robot.arm_name] = (
                    view.get_dof_stiffnesses()[:, ids].clone(),
                    view.get_dof_max_forces()[:, ids].clone(),
                )
            stiffness, force = self.nominal_drives[robot.arm_name]
            articulation.write_joint_stiffness_to_sim(
                (stiffness * self.gripper_scale).to(articulation.device), joint_ids=ids)
            articulation.write_joint_effort_limit_to_sim(
                (force * self.gripper_scale).to(articulation.device), joint_ids=ids)
        self.set_grip_force(0.0)
        return result

    def set_grip_force(self, force):
        """Switch right fingers between scaled position drives and closing force."""
        if force == self.grip_force_command:
            return
        robot, articulation = self.right_gripper
        nominal = self.nominal_drives[robot.arm_name][0].to(articulation.device)
        stiffness = nominal * (self.gripper_scale if force == 0 else 0)
        articulation.write_joint_stiffness_to_sim(stiffness, joint_ids=robot.gripper_joint_indices)
        articulation.set_joint_effort_target(
            nominal.new_full(nominal.shape, -force), joint_ids=robot.gripper_joint_indices)
        self.grip_force_command = force

    def rl_state(self):
        state = super().rl_state()
        robot, articulation = self.right_gripper
        view, ids = articulation.root_physx_view, robot.gripper_joint_indices
        state["gripper_drive"] = {
            "stiffness": view.get_dof_stiffnesses()[0, ids].detach().cpu().tolist(),
            "max_force": view.get_dof_max_forces()[0, ids].detach().cpu().tolist(),
        }
        contacts = self.native.contact_monitor.contacts()
        paths = self.native.finger_paths[self.native.robot_manager.process_name(robot.arm_name)]
        state["right_grip_force"] = [
            sum(abs(float(np.dot(point["impulse"], point["normal"])))
                for pair in contacts
                if beneath(pair["other_actor"], path) or beneath(pair["other_collider"], path)
                for point in pair["points"]) / self.native.sim_cfg.sim.dt
            for path in paths]
        state["grip_force_command"] = self.grip_force_command
        return state


class ForceRLWorker(RLWorker):
    """Apply the requested gripper force before each native action."""

    def step(self, command):
        force = float(command.get("grip_force", 0.0))
        if not 0 <= force <= 100:
            raise ValueError("grip_force must lie in [0, 100] N")
        self.env.set_grip_force(force)
        return super().step(command)


def main(argv=None):
    """Parse worker arguments and serve the existing simulator transport."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--mode", choices=("state", "render"), default="state")
    parser.add_argument("--parameters", required=True, type=json.loads)
    args = parser.parse_args(argv)
    keys = {"friction_scale", "mass_scale", "restitution", "gripper_scale"}
    if (not isinstance(args.parameters, dict) or set(args.parameters) != keys
            or any(type(v) not in (int, float) or not np.isfinite(v)
                   for v in args.parameters.values())
            or args.parameters["gripper_scale"] <= 0):
        parser.error("--parameters requires four numeric parameters and gripper_scale > 0")
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name) or args.gpu < 0:
        parser.error("require a YYYYMMDDHHMM run prefix and nonnegative --gpu")
    scale = args.parameters.pop("gripper_scale")

    def builder(**kwargs):
        return GripperForceTask(**kwargs, gripper_scale=scale)

    # SimulatorClient creates this directory before launching the internal worker.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(args.socket)
        with connection.makefile("rwb") as stream:
            return run_worker(stream, args.run_dir, parameters=args.parameters,
                              mode=args.mode, builder=builder, worker_class=ForceRLWorker)


if __name__ == "__main__":
    raise SystemExit(main())
