"""Simulator worker with scaled gripper drives and contact-force observations."""

import argparse
import json
import re
import socket
from pathlib import Path

import numpy as np

from orchestrator.dynamics_rl_worker import RLWorker, run_worker
from simulators.dynamics_native import beneath
from simulators.dynamics_rl import DynamicsRLTask


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
