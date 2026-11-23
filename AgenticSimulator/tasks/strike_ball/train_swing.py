"""Train a six-joint strike policy from scratch in one simulator run.

Native impacts train a correction for each stroke and initial ball position. Frozen strokes
then compete on fresh training shots. Export records their joint commands and learned local
success rates. Playback needs only the native observation and trajectory_policy.pt.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from . import TASK_ID, enter_run_directory


class TrajectoryPolicy:
    """Look up the learned joint commands from the initial ball position and episode clock."""

    def __init__(self, bundle, device):
        self.actions = bundle["actions"].to(device)
        self.domain = bundle["hole_offset_domain"].to(device)
        self.scores = bundle.get("success_by_position")
        if self.scores is not None:
            self.scores = self.scores.to(device)
        if self.actions.ndim == 4:
            self.actions = self.actions.unsqueeze(0)
        self.position = None

    def __call__(self, obs):
        import torch

        size, _, steps, _ = self.actions.shape[-4:]
        step = ((1 - obs[:, -1]) * steps).round().long().clamp(0, steps - 1)
        # The policy observation stores hole minus ball in columns 24 through 26.
        position = obs[:, [24, 26]]
        if self.position is None:
            self.position = position.clone()
        self.position = torch.where((step == 0)[:, None], position, self.position)
        u = (
            (self.position - self.domain[0]) / (self.domain[1] - self.domain[0]) * (size - 1)
        ).clamp(0, size - 1)
        lower = u.long().clamp_max(size - 2)
        x, z = lower.unbind(1)
        wx, wz = (u - lower).split(1, dim=1)
        family = torch.zeros_like(x)
        if self.scores is not None:
            # Select one complete stroke before interpolating its joint commands.
            scores = self.scores
            left = torch.lerp(scores[:, x, z], scores[:, x + 1, z], wx.T)
            right = torch.lerp(scores[:, x, z + 1], scores[:, x + 1, z + 1], wx.T)
            family = torch.lerp(left, right, wz.T).argmax(0)
        first = torch.lerp(
            self.actions[family, x, z, step], self.actions[family, x + 1, z, step], wx
        )
        second = torch.lerp(
            self.actions[family, x, z + 1, step], self.actions[family, x + 1, z + 1, step], wx
        )
        return torch.lerp(first, second, wz)


def run(args):
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    from .env_cfg import StrikeBallEnvCfg
    from .scene import BOARD_THICKNESS, HOLE_CENTER, SAMPLE_X, SAMPLE_Z

    cfg = StrikeBallEnvCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
    wrapped = gym.make(TASK_ID, cfg=cfg)
    env = wrapped.unwrapped
    try:
        wrapped.reset()
        arm = env.robot
        data = arm.data
        device = env.device
        n = env.num_envs
        size = args.grid_size
        # Each candidate specifies handle roll, nominal swing speed multiplier, wind-up lead,
        # and contact time in simulation substeps. Candidate zero leaves handle roll free.
        strokes = torch.tensor(
            [
                (0.0, 1.0, 0.19, 308),
                (-0.25, 1.32, 0.155, 305),
                (-0.52, 1.18, 0.16, 305),
                (0.12, 1.24, 0.155, 305),
                (-0.12, 1.10, 0.17, 306),
                (0.42, 1.08, 0.16, 305),
            ],
            device=device,
        )
        families = len(strokes)
        cells = families * size * size
        table = torch.zeros((families, size, size, 3), device=device)
        visits = torch.zeros(cells, device=device)
        successes = torch.zeros_like(visits)
        environment_ids = torch.arange(n, device=device)
        eye = torch.eye(6, device=device)
        dt = env.step_dt
        jac_view = arm.root_physx_view
        # The six revolute axes and link frames at reset define forward kinematics.
        child_links = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        joint_links = [arm.body_names.index(link) for link in child_links]
        home_axes = jac_view.get_jacobians()[0, env.wrist - 1, 3:].T.clone()
        home_origins = (data.body_link_pos_w[0, joint_links] - env.scene.env_origins[0]).clone()
        home_center = (env.paddle_center()[0] - env.scene.env_origins[0]).clone()
        home_quat = data.body_link_quat_w[0, env.wrist].clone()
        home_q = data.default_joint_pos[0]
        limits = data.soft_joint_pos_limits[0]
        eye3 = eye[:3, :3]

        def impact_pose(target, normal_at_home, direction, speed, family):
            q = home_q.expand(n, -1).clone()
            handle_at_home = quat_apply(
                home_quat.expand(n, -1), q.new_tensor([0.0, 1.0, 0.0]).expand(n, -1)
            )
            radial = target.clone()
            radial[:, 2] = 0.0
            handle_target = torch.nn.functional.normalize(
                radial - (radial * direction).sum(1, keepdim=True) * direction, dim=1
            )
            yaw = strokes[family, 0:1]
            handle_target = yaw.cos() * handle_target + yaw.sin() * torch.linalg.cross(
                direction, handle_target
            )
            fixed_roll = (family != 0)[:, None, None]
            for _ in range(26):
                rotation = q.new_zeros((n, 4))
                rotation[:, 0] = 1.0
                shift = torch.zeros_like(target)
                axes, anchors = [], []
                for j in range(6):
                    axis = home_axes[j].expand(n, -1)
                    anchor = home_origins[j].expand(n, -1)
                    axes.append(quat_apply(rotation, axis))
                    world_anchor = quat_apply(rotation, anchor) + shift
                    anchors.append(world_anchor)
                    rotation = quat_mul(rotation, quat_from_angle_axis(q[:, j] - home_q[j], axis))
                    shift = world_anchor - quat_apply(rotation, anchor)
                point = quat_apply(rotation, home_center.expand(n, -1)) + shift
                normal = quat_apply(rotation, normal_at_home)
                axes = torch.stack(axes, 1)
                jac = torch.linalg.cross(
                    axes, point[:, None, :] - torch.stack(anchors, 1)
                ).transpose(1, 2)
                handle = quat_apply(rotation, handle_at_home)
                angular = axes.transpose(1, 2)
                angular = torch.where(
                    fixed_roll,
                    angular,
                    angular - normal[:, :, None] * (normal[:, None, :] @ angular),
                )
                tilt_error = torch.linalg.cross(normal, direction)
                roll_error = (torch.linalg.cross(handle, handle_target) * normal).sum(
                    1, keepdim=True
                )
                full = torch.cat((jac, 0.35 * angular), dim=1)
                error = torch.cat(
                    (
                        target - point,
                        0.35 * (tilt_error + roll_error * fixed_roll[:, :, 0] * normal),
                    ),
                    dim=1,
                )
                delta = full.transpose(1, 2) @ torch.linalg.solve(
                    full @ full.transpose(1, 2) + 0.0003 * eye, error[:, :, None]
                )
                q = (q + delta.squeeze(2).clamp(-0.18, 0.18)).clamp(limits[:, 0], limits[:, 1])
            weighted = jac * q.new_tensor([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
            rates = weighted.transpose(1, 2) @ torch.linalg.solve(
                weighted @ jac.transpose(1, 2) + 0.00005 * eye3, (speed * direction)[:, :, None]
            )
            return q, rates.squeeze(2).clamp(-6.0, 6.0)

        released = torch.zeros(n, dtype=torch.bool, device=device)
        residual = torch.zeros((n, 3), device=device)
        first_hit_substep = torch.zeros(n, dtype=torch.long, device=device)
        native_callback = env.ball.after_physics_step

        def record_launch():
            waiting = env.awaiting_separation.clone()
            before_hit = ~env.hit
            native_callback()
            substep = env._sim_step_counter % (env.max_episode_length * cfg.decimation)
            first_hit_substep.masked_fill_(before_hit & env.hit, substep)
            indices = (waiting & ~env.awaiting_separation).nonzero().flatten()
            if indices.numel() == 0:
                return
            released[indices] = True
            velocity = env.ball.data.root_lin_vel_w[indices]
            hole_offset = env.hole[indices] - env.ball.data.root_pos_w[indices]
            vz = velocity[:, 2] + 0.5 * env.gravity[2] * env.physics_dt
            # Semi-implicit gravity gives the descending crossing time after release.
            fall_speed = (
                (vz.square() + 2 * env.gravity[2] * hole_offset[:, 2]).clamp_min(0.0).sqrt()
            )
            flight_time = (vz + fall_speed) / -env.gravity[2]
            entrance_offset = (
                0.33 * BOARD_THICKNESS * velocity[:, :2] / fall_speed.clamp_min(0.5)[:, None]
            )
            residual[indices, :2] = (hole_offset[:, :2] + entrance_offset) / flight_time.clamp_min(
                0.15
            )[:, None] - velocity[:, :2]

        env.ball.after_physics_step = record_launch
        all_passes = all_hits = total = 0
        export_size = args.trajectory_grid
        export_rows = families * export_size**2
        export_batches = (export_rows + n - 1) // n
        selection_batches = max(12, args.batches // 4)
        train_batches = args.batches + selection_batches
        trajectories = torch.empty((export_rows, env.max_episode_length, 6), dtype=torch.float32)
        for batch in range(train_batches + export_batches):
            exporting = batch >= train_batches
            family = (environment_ids + batch) % families
            if exporting:
                # Sample the trained controller on a regular lattice for observation-only inference.
                first_row = (batch - train_batches) * n
                rows = torch.arange(first_row, first_row + n, device=device).clamp_max(
                    export_rows - 1
                )
                family = rows // export_size**2
                rows = rows % export_size**2
                local_start = env.held_state.clone()
                local_start[:, 0] = (
                    SAMPLE_X[0]
                    + (SAMPLE_X[1] - SAMPLE_X[0]) * (rows // export_size) / (export_size - 1)
                    + env.scene.env_origins[:, 0]
                )
                local_start[:, 2] = (
                    SAMPLE_Z[0]
                    + (SAMPLE_Z[1] - SAMPLE_Z[0]) * (rows % export_size) / (export_size - 1)
                    + env.scene.env_origins[:, 2]
                )
                env.ball.write_root_state_to_sim(local_start)
                env.held_state.copy_(local_start)
                env.previous_pos.copy_(local_start[:, :3])
                commands = []

            first_hit_substep.zero_()
            released.zero_()
            residual.zero_()
            start = env.held_state[:, :3].clone()
            local = start - env.scene.env_origins
            xp = ((local[:, 0] - SAMPLE_X[0]) / (SAMPLE_X[1] - SAMPLE_X[0]) * (size - 1)).clamp(
                0, size - 1
            )
            zp = ((local[:, 2] - SAMPLE_Z[0]) / (SAMPLE_Z[1] - SAMPLE_Z[0]) * (size - 1)).clamp(
                0, size - 1
            )
            i, j = xp.long().clamp_max(size - 2), zp.long().clamp_max(size - 2)
            x, z = xp - i, zp - j
            weights = torch.stack(((1 - x) * (1 - z), x * (1 - z), (1 - x) * z, x * z), dim=1)
            indices = torch.stack(
                (i * size + j, (i + 1) * size + j, i * size + j + 1, (i + 1) * size + j + 1), dim=1
            )
            indices += (family * size * size)[:, None]
            correction = (table.reshape(-1, 3)[indices] * weights[:, :, None]).sum(1)
            outgoing = (env.hole - start) / args.flight_time - 0.5 * env.gravity * (
                args.flight_time + env.physics_dt
            )
            outgoing[:, :2] += correction[:, :2]
            speed = outgoing.norm(dim=1, keepdim=True) * strokes[family, 1:2] / 1.9
            direction = torch.nn.functional.normalize(outgoing, dim=1)
            normal_local = torch.zeros_like(direction)
            normal_local[:, 2] = 1.0
            normal_sign = (quat_apply(home_quat.expand(n, -1), normal_local) * direction).sum(1)
            normal_local[:, 2] = torch.where(normal_sign < 0, -1.0, 1.0)
            normal_local[:, 1] = 0.032
            normal_local = torch.nn.functional.normalize(normal_local, dim=1)
            hit_pose, hit_rates = impact_pose(
                local - 0.026 * direction,
                quat_apply(home_quat.expand(n, -1), normal_local),
                direction,
                speed,
                family,
            )
            lead = strokes[family, 2:3] + correction[:, 2:3] / speed
            for step in range(env.max_episode_length):
                # Accelerate from the wind-up pose and cross the impact pose at cruise speed.
                t = max(0.0, step * dt - 1.25)
                ramp = 0.20
                t_ramp = min(t, ramp)
                t_cruise = min(max(t - ramp, 0.0), 0.23)
                tail = min(max(t - ramp - 0.23, 0.0), ramp)
                travel = 0.5 * t_ramp**2 / ramp + t_cruise + tail - 0.5 * tail**2 / ramp
                rate = t_ramp / ramp - tail / ramp
                acceleration = (
                    float(0.0 < t < ramp) - float(ramp + 0.23 < t < 2 * ramp + 0.23)
                ) / ramp
                rates = rate * hit_rates
                accel = acceleration * hit_rates
                effort = (jac_view.get_generalized_mass_matrices() @ accel[:, :, None]).squeeze(2)
                effort += (
                    jac_view.get_gravity_compensation_forces()
                    + jac_view.get_coriolis_and_centrifugal_compensation_forces()
                )
                # Encode velocity feedforward in the ordinary joint position target.
                joint_target = (
                    hit_pose
                    + (travel - lead + 0.5 * dt * rate) * hit_rates
                    + (effort + data.joint_damping * rates) / data.joint_stiffness
                )
                action = (joint_target - data.default_joint_pos) / cfg.actions.arm.scale
                if exporting:
                    commands.append(action.clamp(-10.0, 10.0).clone())
                _, _, _, _, extras = wrapped.step(action)
            if exporting:
                count = min(n, export_rows - first_row)
                trajectories[first_row : first_row + count] = torch.stack(commands, 1)[:count].cpu()
                continue
            shot = extras["shot"]
            passes, hits = int(shot["passed"].sum()), int(shot["hit"].sum())
            all_passes += passes
            all_hits += hits
            total += n
            valid = released & (first_hit_substep > 250) & (first_hit_substep < 350)
            # Adjust the backswing so contact occurs at the same physics step across ball positions.
            residual[:, 2] = (strokes[family, 3] - first_hit_substep) * env.physics_dt * speed[:, 0]
            row = dict(
                batch=batch,
                episodes=n,
                hits=hits,
                passes=passes,
                pass_rate=passes / n,
                releases=int(released.sum()),
                by_stroke=(
                    torch.bincount(family, weights=shot["passed"].float(), minlength=families)
                    / torch.bincount(family, minlength=families).clamp_min(1)
                ).tolist(),
                median_velocity_error=float(residual[released, :2].norm(dim=1).median())
                if released.any()
                else None,
            )
            print(json.dumps(row), flush=True)
            with Path("metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if batch >= args.batches:
                visits.index_add_(0, indices.flatten(), weights.flatten())
                successes.index_add_(
                    0, indices.flatten(), (weights * shot["passed"][:, None]).flatten()
                )
                continue
            sums = torch.zeros_like(table).reshape(-1, 3)
            mass = torch.zeros(cells, device=device)
            contribution = weights * valid[:, None]
            sums.index_add_(
                0,
                indices.flatten(),
                (contribution[:, :, None] * residual[:, None, :].clamp(-0.5, 0.5)).reshape(-1, 3),
            )
            mass.index_add_(0, indices.flatten(), contribution.flatten())
            table += args.gain * (sums / mass.clamp_min(1.0)[:, None]).reshape_as(table)
            table[1:, :, :, 2].clamp_(-0.032, 0.032)
        env.ball.after_physics_step = native_callback
        success_grid = ((successes + 0.5) / (visits + 1.0)).reshape(families, 1, size, size)
        scores = torch.nn.functional.interpolate(
            success_grid, size=(export_size, export_size), mode="bilinear", align_corners=True
        )[:, 0].cpu()
        domain = torch.tensor(
            [
                [HOLE_CENTER[0] - SAMPLE_X[0], HOLE_CENTER[2] - SAMPLE_Z[0]],
                [HOLE_CENTER[0] - SAMPLE_X[1], HOLE_CENTER[2] - SAMPLE_Z[1]],
            ]
        )
        torch.save(
            dict(
                format="strike-ball-trajectory-v2",
                success_by_position=scores,
                hole_offset_domain=domain,
                actions=trajectories.reshape(
                    families, export_size, export_size, env.max_episode_length, 6
                ),
            ),
            "trajectory_policy.pt",
        )
        Path("summary.json").write_text(
            json.dumps(
                dict(
                    seed=args.seed,
                    episodes=total,
                    hits=all_hits,
                    passes=all_passes,
                    pass_rate=all_passes / total,
                    initialization="zero correction, no prior model",
                    grid_size=size,
                    flight_time=args.flight_time,
                ),
                indent=2,
            )
        )
    finally:
        wrapped.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--batches", type=int, default=72)
    parser.add_argument("--grid-size", type=int, default=23)
    parser.add_argument("--gain", type=float, default=0.6)
    parser.add_argument("--flight-time", type=float, default=0.72)
    parser.add_argument("--trajectory-grid", type=int, default=65)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (
        min(args.num_envs, args.batches, args.gain, args.flight_time) <= 0
        or min(args.grid_size, args.trajectory_grid) < 2
    ):
        parser.error("positive training dimensions and grid sizes >= 2 are required")
    enter_run_directory(args.run_dir)
    with Path("training_config.json").open("x") as config:
        json.dump(
            {
                key: getattr(args, key)
                for key in (
                    "seed",
                    "num_envs",
                    "batches",
                    "grid_size",
                    "gain",
                    "flight_time",
                    "trajectory_grid",
                )
            },
            config,
            indent=2,
        )
    app = AppLauncher(args).app
    try:
        run(args)
    finally:
        app.close()


if __name__ == "__main__":
    main()
