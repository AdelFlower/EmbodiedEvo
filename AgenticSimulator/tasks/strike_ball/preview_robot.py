"""在 Isaac Sim 中预览方台上的 UR10 和目标矩形中的乒乓球。"""

import argparse
import random
import time

from isaaclab.app import AppLauncher

from . import enter_run_directory


class PaddleStrike:
    """按拍心位置和拍面方向控制六个关节，用球拍碰撞报告释放球。"""

    def __init__(self, sim, robot, ball):
        import torch
        from scipy.interpolate import RegularGridInterpolator
        from .scene import make_contact_view

        self.robot = robot
        self.ball = ball
        self.wrist = robot.body_names.index("wrist_3_link")
        self.physics, self.contacts = make_contact_view(robot, robot.cfg.prim_path.rsplit("/", 1)[0].replace(".*", "*"))
        mass = ball.root_physx_view.get_masses()[0, 0].item()
        self.support = -mass * ball.data.default_root_state.new_tensor(sim.cfg.gravity).reshape(1, 1, 3)
        self.zero_force = torch.zeros_like(self.support)
        self.control_dt = 4 * sim.get_physics_dt()
        self.identity = torch.eye(6, device=robot.device)
        # 在九个击球点用实际碰撞标定的挥拍指令修正量（m/s）。
        # 行对应 z，列对应 x；只修正机械臂指令，球速始终由碰撞产生。
        self.swing_correction = RegularGridInterpolator(
            ((0.75, 0.93, 1.11), (-0.05, 0.15, 0.35)),
            [
                [(0.0257, 0.0168, 0.0465), (0.0439, 0.0272, 0.2147), (-0.0008, 0.0315, 0.1878)],
                [(0.0848, -0.0313, 0.3041), (0.0383, 0.011, 0.2787), (-0.0047, -0.0016, 0.0936)],
                [(0.0295, 0.0294, 0.038), (0.0099, -0.0311, 0.0907), (-0.0212, -0.0271, 0.204)],
            ],
        )

    def begin(self, start, velocity):
        import torch
        from isaaclab.utils.math import quat_apply

        robot = self.robot
        home = robot.data.default_joint_pos
        robot.write_joint_state_to_sim(home, robot.data.default_joint_vel)
        robot.reset()
        robot.set_joint_position_target(home)
        robot.set_joint_velocity_target(torch.zeros_like(home))
        # 球以单位姿态静止且已锁定旋转，用等于球重的反向力原地托住它。
        self.ball.set_external_force_and_torque(self.support, self.zero_force)
        self.hit = False
        self.crossed = False
        self.passed = False
        self.start = home.new_tensor(start)
        self.held_state = self.ball.data.default_root_state.clone()
        self.held_state[0, :3] = self.start
        self.held_state[:, 7:] = 0
        self.previous = self.start
        self.linear_integral = torch.zeros_like(self.start)
        self.angle_integral = torch.zeros_like(self.start)
        correction = self.swing_correction((start[2], start[0]))
        outgoing = home.new_tensor(velocity) + home.new_tensor(correction)
        self.direction = outgoing / outgoing.norm()
        # 对静止球，近似出球法向速度为 (1 + 恢复系数) × 拍速。
        self.speed = outgoing.norm().item() / 1.9
        quat = robot.data.body_link_quat_w[0, self.wrist]
        self.local_normal = home.new_tensor([0., 0., 1.])
        if quat_apply(quat, self.local_normal).dot(self.direction) < 0:
            self.local_normal.neg_()
        # 拍面凸包从较厚的拍柄向拍头略微倾斜。
        self.local_normal[1] = 0.032
        self.local_normal /= self.local_normal.norm()

    def move(self, elapsed):
        import torch
        from isaaclab.utils.math import quat_apply

        from .scene import PADDLE_OFFSET

        robot = self.robot
        quat = robot.data.body_link_quat_w[0, self.wrist]
        normal = quat_apply(quat, self.local_normal)
        # 模型的拍心在腕部局部 +Y 方向 0.265 m，两面沿局部 ±Z。
        offset = quat_apply(quat, self.start.new_tensor(PADDLE_OFFSET))
        center = robot.data.body_link_pos_w[0, self.wrist] + offset
        jac = robot.root_physx_view.get_jacobians()[0, self.wrist - 1].clone()
        lever = center - robot.data.body_com_pos_w[0, self.wrist]
        jac[:3] += torch.linalg.cross(jac[3:].T, lever.expand_as(jac[3:].T)).T

        # 只约束拍面倾角，绕法向的旋转不影响击球。
        jac[3:] -= normal[:, None] * (normal @ jac[3:])

        # 先在球后方就位，再沿法向挥过球心，留出随挥距离。
        swing_time = max(0., elapsed - 1.25)
        ramp = 0.24
        acceleration_time = min(swing_time, ramp)
        travel = self.speed * (0.5 * acceleration_time**2 / ramp + max(0., swing_time - ramp))
        target = self.start + (min(travel, 0.61) - 0.41) * self.direction
        feedforward = self.speed * acceleration_time / ramp if travel < 0.61 else 0.
        linear = 10. * (target - center) + feedforward * self.direction
        tilt = torch.linalg.cross(normal, self.direction)
        if 0. < travel < 0.61:
            velocity_error = feedforward * self.direction - jac[:3] @ robot.data.joint_vel[0]
            self.linear_integral += self.control_dt * velocity_error
            self.angle_integral += self.control_dt * tilt
            linear = feedforward * self.direction + 0.55 * velocity_error + 15. * self.linear_integral
        angular = 28. * tilt + 100. * self.angle_integral
        twist = torch.cat((linear, angular))
        rates = jac.T @ torch.linalg.solve(jac @ jac.T + 0.0004 * self.identity, twist)
        rates = rates.clamp(-6., 6.)
        limits = robot.data.soft_joint_pos_limits[0]
        view = robot.root_physx_view
        accel = ((rates - robot.data.joint_vel[0]) / self.control_dt).clamp(-65., 65.)
        effort = view.get_generalized_mass_matrices()[0] @ accel + view.get_gravity_compensation_forces()[0]
        effort += view.get_coriolis_and_centrifugal_compensation_forces()[0]
        # 把前馈力矩换算成位置目标偏置，所有驱动力矩均经过关节驱动器的限幅。
        joints = robot.data.joint_pos[0] + self.control_dt * rates + effort / robot.data.joint_stiffness[0]
        joints = joints.clamp(limits[:, 0], limits[:, 1])
        robot.set_joint_position_target(joints.unsqueeze(0))
        robot.set_joint_velocity_target(rates.unsqueeze(0))

    def observe(self):
        import torch
        from .scene import HOLE_CENTER, check_passage, paddle_contacts

        if not self.hit:
            touching = paddle_contacts(self.contacts, self.robot, self.wrist)
            if touching.item():
                self.hit = True
                self.ball.set_external_force_and_torque(self.zero_force, self.zero_force)
                print("球拍接触，撤去托力。球速 (m/s):", self.ball.data.root_lin_vel_w[0].tolist(), flush=True)
            if not self.hit and not torch.equal(self.ball.data.root_state_w, self.held_state):
                # 换球时旧接触可能残留一个物理步，等待期间保持采样位置和零速度。
                self.ball.write_root_state_to_sim(self.held_state)
        if self.hit and not self.passed:
            pos = self.ball.data.root_pos_w[0]
            target = self.start.new_tensor(HOLE_CENTER)
            down, error, pending, passed = check_passage(self.previous, pos, target, self.crossed)
            self.crossed = pending.item()
            if down.item():
                print(f"向下经过板中面，球心偏差 {error.item() * 1000:.1f} mm", flush=True)
            if passed.item():
                self.passed = True
                print("整球已穿孔", flush=True)
            self.previous = pos.clone()


def play_ball_flights(app, sim, robot, ball, sample_center, strike=None, *, duration_s=3.0):
    """按指定的仿真时长播放每球，用 PhysX 返回的球心位置绘制轨迹。"""
    from pxr import UsdGeom
    from .scene import CAMERA_EYE, CAMERA_TARGET, HOLE_CENTER, target_velocity

    dt = sim.get_physics_dt()
    steps_per_frame = 4
    steps_per_sample = round(duration_s / dt)
    trail = UsdGeom.BasisCurves.Define(sim.stage, "/World/BallTrajectory")
    trail.CreateTypeAttr(UsdGeom.Tokens.linear)
    trail.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    trail.CreateWidthsAttr([0.003])
    trail.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    trail.CreateDisplayColorAttr([(0.03, 0.72, 0.85)])
    points_attr = trail.CreatePointsAttr()
    counts_attr = trail.CreateCurveVertexCountsAttr()

    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
    while app.is_running():
        start = sample_center()
        state = ball.data.default_root_state.clone()
        state[0, :3] = state.new_tensor(start)
        velocity = target_velocity(state[0, :3], state.new_tensor(HOLE_CENTER), sim.cfg.gravity, dt).tolist()
        if strike is None:
            state[0, 7:10] = state.new_tensor(velocity)
        else:
            strike.begin(start, velocity)
            state[0, 7:10] = 0
        state[0, 10:13] = 0
        ball.write_root_state_to_sim(state)
        print(f"采样位置 (m): {start}，目标出球速度 (m/s): {velocity}", flush=True)

        points = [start]
        sample_start = time.monotonic()
        for step in range(1, steps_per_sample + 1):
            if not app.is_running():
                return
            if strike is not None and (step - 1) % steps_per_frame == 0:
                strike.move((step - 1) * dt)
            if strike is not None:
                ball.write_data_to_sim()
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            ball.update(dt)
            if strike is not None:
                strike.observe()
            if step % steps_per_frame == 0 or step == steps_per_sample:
                points.append(tuple(ball.data.root_pos_w[0].tolist()))
                points_attr.Set(points)
                counts_attr.Set([len(points)])
                sim.render()
                time.sleep(max(0.0, sample_start + step * dt - time.monotonic()))
        if strike is not None:
            hit_result = "已打出" if strike.hit else "未击中"
            hole_result = "已穿孔" if strike.passed else "未穿孔"
            print(f"本球 {duration_s:g} 秒结束，{hit_result}，{hole_result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="以 YYYYMMDDHHMM 开头的运行目录")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--ballistic", action="store_true", help="按正常速度预览随机发球，每球 3 秒仿真时间")
    modes.add_argument("--rule-strike", action="store_true", help="机械臂挥拍击打静止的随机球，每球使用任务配置的回合时长")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    enter_run_directory(args.run_dir)

    launcher = AppLauncher(args)
    app = launcher.app
    sim = None
    try:
        import isaaclab.sim as sim_utils

        from isaaclab.scene import InteractiveScene
        from .scene import SAMPLE_X, SAMPLE_Y, SAMPLE_Z
        from .env_cfg import StrikeBallEnvCfg

        task_cfg = StrikeBallEnvCfg()
        task_cfg.sim.device = args.device
        sim = sim_utils.SimulationContext(task_cfg.sim)
        if sim.has_gui():
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            viewport.resolution_scale = 1.0
            viewport.resolution = (1920, 1080)

        dynamic_ball = args.ballistic or args.rule_strike
        if not dynamic_ball:
            sim.set_camera_view(eye=(2.3, -2.6, 1.9), target=(0.0, 0.0, 0.85))

        def sample_center():
            return (random.uniform(*SAMPLE_X), SAMPLE_Y, random.uniform(*SAMPLE_Z))

        task_cfg.scene.num_envs = 1
        task_cfg.scene.ball.spawn.activate_contact_sensors = args.rule_strike
        task_cfg.scene.ball.spawn.rigid_props.kinematic_enabled = not dynamic_ball
        scene = InteractiveScene(task_cfg.scene)
        robot, ball = scene["robot"], scene["ball"]
        sim.reset()
        pose = robot.data.default_joint_pos
        robot.write_joint_state_to_sim(pose, robot.data.default_joint_vel)
        robot.reset()
        robot.set_joint_position_target(pose)

        if not dynamic_ball:
            state = ball.data.default_root_state.clone()
            state[0, :3] = state.new_tensor(sample_center())
            ball.write_root_state_to_sim(state)
            print("预览球心 (m):", state[0, :3].tolist(), flush=True)

        if args.rule_strike:
            strike = PaddleStrike(sim, robot, ball)
            play_ball_flights(app, sim, robot, ball, sample_center, strike, duration_s=task_cfg.episode_length_s)
            return
        if args.ballistic:
            play_ball_flights(app, sim, robot, ball, sample_center)
            return

        dt = sim.get_physics_dt()
        while app.is_running():
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
    finally:
        if sim is not None:
            # Release Isaac Lab's timeline stop listener before Kit closes the stage.
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close()


if __name__ == "__main__":
    main()
