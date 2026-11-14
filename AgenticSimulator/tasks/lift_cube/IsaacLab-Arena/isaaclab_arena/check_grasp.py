"""Check physical graspability with IK through the task's unchanged joint actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name):
        parser.error("--run-dir requires a YYYYMMDDHHMM prefix")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    with (args.run_dir / "grasp.log").open("x") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "isaaclab_arena.check_grasp", "--worker",
                                 "--run-dir", str(args.run_dir.resolve()), *remaining],
                                stdout=log, stderr=subprocess.STDOUT)
    return result.returncode if result.returncode else int(not (args.run_dir / "grasp_check_complete.json").exists())


def worker():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num_envs must be positive")
    launcher = AppLauncher(args)
    try:
        run(args)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        launcher.app.close()


def run(args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms, quat_apply, matrix_from_quat
    from isaaclab.utils.io import dump_yaml
    from isaaclab_arena.env_cfg import LiftCubeEnvCfg

    cfg = LiftCubeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset()
        dump_yaml(str(args.run_dir / "env.yaml"), cfg)
        robot, cube = env.scene["robot"], env.scene["dex_cube"]
        arm = env.action_manager.get_term("arm_action")
        ids = arm._joint_ids
        hand_id = robot.find_bodies("panda_hand")[0][0]
        ik = DifferentialIKController(DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
        ), num_envs=args.num_envs, device=env.device)
        offset = torch.tensor([0., 0., 0.1034], device=env.device).repeat(args.num_envs, 1)
        actions = torch.zeros(args.num_envs, 8, device=env.device)
        start = None
        trace = []
        lifted = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        success = lifted.clone()
        for step in range(env.max_episode_length - 1):
            hand = robot.data.body_state_w[:, hand_id, :7]
            root = robot.data.root_state_w[:, :7]
            if step == 0:
                start = cube.data.root_pos_w.clone()
                q = cube.data.root_quat_w
                yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
                yaw = (yaw + torch.pi/4) % (torch.pi/2) - torch.pi/4
                orientation = torch.stack((yaw*0, torch.cos(yaw/2), torch.sin(yaw/2), yaw*0), dim=1)
            if step < 50:
                target = cube.data.root_pos_w.clone()
                target[:, 2] += 0.12
                phase = "above"
            elif step < 110:
                target = cube.data.root_pos_w.clone()
                phase = "descend"
            elif step < 155:
                if step == 110:
                    grasp = cube.data.root_pos_w.clone()
                target = grasp.clone()
                phase = "close"
            else:
                command = env.command_manager.get_command("object_pose")
                target = root[:, :3] + quat_apply(root[:, 3:], command[:, :3])
                phase = "lift_to_goal"
            hand_target = target - quat_apply(orientation, offset)
            pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:], hand[:, :3], hand[:, 3:])
            goal_pos, goal_quat = subtract_frame_transforms(root[:, :3], root[:, 3:], hand_target, orientation)
            jac = robot.root_physx_view.get_jacobians()[:, hand_id-1, :, ids].clone()
            com = quat_apply(hand[:, 3:], robot.data.com_pos_b[:, hand_id])
            jac[:, :3] += torch.cross(com.unsqueeze(-1).expand_as(jac[:, 3:]), jac[:, 3:], dim=1)
            rot = matrix_from_quat(root[:, 3:]).transpose(1, 2)
            jac[:, :3] = rot @ jac[:, :3]
            jac[:, 3:] = rot @ jac[:, 3:]
            ik.set_command(torch.cat((goal_pos, goal_quat), dim=1))
            joints = robot.data.joint_pos[:, ids]
            desired = ik.compute(pos, quat, jac, joints)
            desired = joints + (desired-joints).clamp(-0.3, 0.3)
            limits = robot.data.soft_joint_pos_limits[:, ids]
            desired = desired.maximum(limits[..., 0]).minimum(limits[..., 1])
            actions[:, :7] = (desired-arm._offset)/arm._scale
            actions[:, 7] = 1. if step < 110 else -1.
            _, rewards, terminated, truncated, _ = env.step(actions)
            obj = cube.data.root_pos_w
            goal = root[:, :3] + quat_apply(root[:, 3:], env.command_manager.get_command("object_pose")[:, :3])
            error = (obj-goal).norm(dim=1)
            lifted |= obj[:, 2] > 0.79
            success |= error < 0.05
            trace.append({"step": step, "phase": phase, "cube": obj.tolist(),
                          "eef": env.scene["ee_frame"].data.target_pos_w[:, 0].tolist(),
                          "fingers": robot.data.joint_pos[:, -2:].tolist(),
                          "joints": joints.tolist(), "joint_targets": desired.tolist(),
                          "hand_target": hand_target.tolist(),
                          "goal_error": error.tolist(), "reward": rewards.tolist(),
                          "terminated": terminated.tolist(), "truncated": truncated.tolist()})
        (args.run_dir / "trajectory.json").write_text(json.dumps(trace))
        summary = {"num_envs": args.num_envs, "steps": len(trace), "lifted": lifted.tolist(),
                   "goal_success": success.tolist(), "lift_rate": lifted.float().mean().item(),
                   "goal_success_rate": success.float().mean().item(), "controller": "IK diagnostic, not PPO"}
        (args.run_dir / "grasp_check_complete.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(summary, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(worker() if "--worker" in sys.argv else main())
