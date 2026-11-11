"""M3a GPU physics/control runtime. No rendering, auto reset, PPO or Agent programs.

The caller owns AppLauncher/SimulationApp. Importing this module is CPU-safe.
"""

import json
from pathlib import Path

import numpy as np
import torch

from integrations.cube_to_plate.batch_inputs import policy_inputs
from integrations.cube_to_plate.batch_rows import ErrorRows, ScoringRows, OutcomeRows
from integrations.cube_to_plate.tactile import pad_normal_forces
from integrations.deployable.observation import Observer
from integrations.friction_rl.control import SCENE_PATH
from orchestrator.harness.contracts import StepBatch
from simulators.cube_to_plate_batch_ik import ArmKinematics
from simulators.cube_to_plate_batch_motion import BatchPoseMotion, BatchStability
from simulators.cube_to_plate_batch_math import BatchDecoder, arm_control_profile, gripper_profile, gripper_drive_target, pad_normals
from simulators.cube_to_plate_batch_scoring import BatchEpisodeState


class BatchRuntime:
    def __init__(self, environment_ref, num_envs, render=False):
        if render is not False or type(num_envs) is not int or num_envs < 1:
            raise ValueError('M3a requires render=False and a positive num_envs')
        if set(environment_ref) != {'target', 'calibration'}:
            raise ValueError('batch environment requires opaque target (or None) and frozen calibration')
        from integrations.cube_to_plate.target import TargetRef
        if environment_ref['target'] is not None and not isinstance(environment_ref['target'], TargetRef):
            raise ValueError('target must be an opaque load_target_spec result')
        from simulators.cube_to_plate_batch_scene import build_scene
        self.target = environment_ref['target']
        self.calibration = environment_ref['calibration']
        self.sim, self.scene, self.manager, self.ids, self.paths = build_scene(num_envs, self.target)
        self.n, self.device = num_envs, self.sim.device
        self.arms = [self.scene[f'robot{i}'] for i in range(2)]
        self.right = self.arms[1]
        self.arm_ids, self.grip_ids, self.body_ids = self.ids[1]
        self.kinematics = ArmKinematics(device=self.device)
        self.arm_target = self.right.data.default_joint_pos[:, self.arm_ids].clone()
        self.nominal_stiffness = [a.data.joint_stiffness[:, ids[1]].clone() for a, ids in zip(self.arms, self.ids)]
        self.nominal_effort = [a.data.joint_effort_limits[:, ids[1]].clone() for a, ids in zip(self.arms, self.ids)]
        self.force_mode = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.detection = torch.zeros((num_envs, 2, 2), device=self.device)
        self.detection_host = np.zeros((num_envs, 2, 2))
        self.episode_ids = [None]*num_envs
        self.history = []
        self.resets = []
        self.generation = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.initial = None
        self.episodes = None
        self.cross_replica_contacts = 0
        self._audit_subscription = None
        import carb
        from simulators.cube_to_plate_batch_errors import PhysicsErrors
        self.physics_errors = PhysicsErrors(Path(carb.settings.get_settings().get('/log/file')))

    def enable_contact_audit(self):
        """Host gate only: audit all reported robot/object contacts, including reset."""
        import re
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        def report(headers, data):
            for h in headers:
                paths = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.actor0, h.actor1)]
                envs = [re.search(r'/envs/env_(\d+)(?:/|$)', p) for p in paths]
                if all(envs) and envs[0][1] != envs[1][1]:
                    self.cross_replica_contacts += 1
        self._audit_subscription = get_physx_simulation_interface().subscribe_contact_report_events(report)

    def _frozen(self, ids):
        return (ids, [a.data.joint_pos[ids].clone() for a in self.arms],
                [a.data.joint_vel[ids].clone() for a in self.arms],
                self.scene['cube'].data.root_state_w[ids].clone())

    def _advance(self, frozen):
        ids, positions, velocities, cubes = frozen
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(.004)
        if len(ids):
            for art, q, v in zip(self.arms, positions, velocities):
                art.write_joint_state_to_sim(q, v, env_ids=ids)
            self.scene['cube'].write_root_state_to_sim(cubes, env_ids=ids)

    def _layout_pose(self, layouts, name):
        nominal = json.loads(SCENE_PATH.read_text())['Rigid'][name][0]
        rows = []
        for layout in layouts:
            pos = np.array(nominal['default_pos'])
            pos[:2] += layout[f'{name}_offset']
            rows.append([*pos, *nominal['default_ori']])
        return torch.tensor(rows, device=self.device, dtype=torch.float32)

    def reset(self, layouts, env_ids=None):
        """M2 layouts; optional subset reset preserves every other replica's state."""
        ids = torch.arange(self.n, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if (len(layouts) != len(ids) or len(set(ids.cpu().tolist())) != len(ids)
                or bool(((ids < 0) | (ids >= self.n)).any())):
            raise ValueError('reset requires one layout per unique in-range env id')
        if self.episodes is None and len(ids) != self.n:
            raise ValueError('the first reset must initialize all replicas')
        for layout in layouts:
            if not {'stream', 'index', 'scene_seed', 'cube_offset', 'plate_offset'} <= set(layout):
                raise ValueError('reset requires M2 layout entries')
            if not np.isfinite([*layout['cube_offset'], *layout['plate_offset']]).all():
                raise ValueError('layout offsets must be finite')
        mask = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        mask[ids] = True
        frozen = self._frozen(torch.where(~mask)[0])
        for i, art in enumerate(self.arms):
            _, grips, _ = self.ids[i]
            art.write_joint_stiffness_to_sim(self.nominal_stiffness[i][ids], joint_ids=grips, env_ids=ids)
            art.write_joint_effort_limit_to_sim(self.nominal_effort[i][ids], joint_ids=grips, env_ids=ids)
            art.set_joint_effort_target(torch.zeros((len(ids), 2), device=self.device), joint_ids=grips, env_ids=ids)
            q, v = art.data.default_joint_pos[ids].clone(), art.data.default_joint_vel[ids].clone()
            art.write_joint_state_to_sim(q, v, env_ids=ids)
            art.set_joint_position_target(q, env_ids=ids)
            art.set_joint_velocity_target(v, env_ids=ids)
            art.reset(ids)
        self.force_mode[ids] = False
        # Native: default robot targets settle 300 steps, table settles 20,
        # restored objects 20, stability observation 300, capture warm-up 200.
        for _ in range(320):
            self._advance(frozen)
        for name in ('cube', 'plate'):
            obj = self.scene[name]
            pose = self._layout_pose(layouts, name)
            pose[:, :3] += self.scene.env_origins[ids]
            state = torch.cat((pose, torch.zeros((len(ids), 6), device=self.device)), -1)
            obj.write_root_state_to_sim(state, env_ids=ids)
            obj.reset(ids)
        for _ in range(20):
            self._advance(frozen)
        stability = BatchStability(self.scene['cube'].data.root_link_pose_w[ids], self.scene.env_origins[ids, 2]+.765)
        for k in range(300):
            self._advance(frozen)
            stability.update(k, self.scene['cube'].data.root_link_pose_w[ids])
        for _ in range(200):
            self._advance(frozen)
        self.scene.reset(ids)
        self.generation[ids] += 1
        sample, pose, plate = self._sample()
        if self.episodes is None:
            self.decoder = BatchDecoder(pose)
            self.episodes = BatchEpisodeState(plate[:, :3], pose, sample['cube_position'], sample['fingers'][:, 1])
        else:
            self.decoder.reset(ids, pose[ids])
            self.episodes.reset(ids, plate[ids, :3], pose[ids], sample['cube_position'][ids], sample['fingers'][ids, 1])
        broken, fatal = self.physics_errors.poll()
        reset_error = stability.unstable | fatal
        reset_error |= torch.tensor([int(i) in broken for i in ids.cpu().tolist()], device=self.device)
        self.episodes.error[ids] = reset_error
        self.arm_target[ids] = self.right.data.default_joint_pos[ids][:, self.arm_ids]
        self.linear = torch.zeros((self.n, 3), device=self.device) if self.initial is None else self.linear
        self.angular = torch.zeros_like(self.linear) if self.initial is None else self.angular
        self.linear[ids], self.angular[ids] = 0, 0
        for env, layout in zip(ids.cpu().tolist(), layouts):
            self.episode_ids[env] = (layout['stream'], layout['index'], int(layout['scene_seed']), int(self.generation[env]))
        # The host controls detector RNGs. Default episode streams are keyed by
        # scene seed; gates explicitly replace these with the frozen DP streams.
        self.set_detection(ids.cpu().tolist(), [[int(l['scene_seed']), 1] for l in layouts])
        batch = self._batch(sample, pose, plate, torch.zeros((self.n, 5), device=self.device),
                            torch.zeros(self.n, device=self.device))
        row = self._row(sample, pose, plate, batch, valid=mask)
        if self.initial is None:
            self.initial = row
        else:
            for key, value in row.items():
                self.initial[key][ids] = value[ids]
        self.resets.append({k: v.clone() for k, v in row.items()})
        return batch

    def set_detection(self, env_ids, seeds):
        """One reset-only Observer draw per replica, exactly the DP implementation."""
        cube = (self.scene['cube'].data.root_link_pos_w-self.scene.env_origins).cpu().numpy()
        plate = (self.scene['plate'].data.root_link_pos_w-self.scene.env_origins).cpu().numpy()
        for i, seed in zip(env_ids, seeds, strict=True):
            state = {'sample': {'cube_position': cube[i]}, 'plate_position': plate[i]}
            observer = Observer('simulated', self.calibration, np.random.default_rng(seed))
            observer.reset(state, None)
            self.detection_host[i] = observer.detection
            self.detection[i] = torch.as_tensor(observer.detection, device=self.device)

    def reset_observation(self):
        sample, pose, plate = self._sample()
        return self._batch(sample, pose, plate, torch.zeros((self.n, 5), device=self.device),
                           torch.zeros(self.n, device=self.device))

    def _sample(self):
        cube = self.scene['cube'].data.root_link_pose_w.clone()
        plate = self.scene['plate'].data.root_link_pose_w.clone()
        cube[:, :3] -= self.scene.env_origins
        plate[:, :3] -= self.scene.env_origins
        sensor = self.scene.sensors['cube_contacts']
        _, _, _, _, counts, _ = sensor.contact_physx_view.get_contact_data(dt=.004)
        flags = counts.reshape(self.n, -1) > 0
        opening, finger_positions = [], []
        for art, (_, grips, bodies) in zip(self.arms, self.ids):
            opening.append((art.data.joint_pos[:, grips].mean(-1)+.01)/.054)
            finger_positions.append(art.data.body_link_pos_w[:, bodies[1:]].mean(1)-self.scene.env_origins)
        pose = self.right.data.body_link_pose_w[:, self.body_ids[0]].clone()
        pose[:, :3] -= self.scene.env_origins
        sample = {'cube_position': cube[:, :3], 'cube_quaternion': cube[:, 3:],
            'table': flags[:, 0], 'plate': flags[:, 1], 'robot': flags[:, 2:].any(-1),
            'fingers': flags[:, -4:].reshape(self.n, 2, 2), 'opening': torch.stack(opening, 1),
            'finger_position': torch.stack(finger_positions, 1)}
        return sample, pose, plate

    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.n, 5) or not bool(torch.isfinite(actions).all()):
            raise ValueError('actions must be finite [N,5]')
        active = self.episodes.active.clone()
        applied, force, closed, native = self.decoder.step(self.episodes.steps, actions, active)
        current = torch.where(active[:, None], self.right.data.joint_pos[:, self.arm_ids],
                              self.right.data.default_joint_pos[:, self.arm_ids])
        root = self.right.data.root_link_pose_w.clone()
        root[:, :3] -= self.scene.env_origins
        inactive_pose, _ = self.kinematics.forward(current, root)
        ik_command = torch.where(active[:, None], applied, inactive_pose)
        solution, success, p_error, r_error = self.kinematics.solve(current, ik_command, root)
        self.last_control = {'current': current.clone(), 'command': applied.clone(), 'root': root.clone(),
                             'solution': solution, 'success': success, 'position_error': p_error,
                             'orientation_error': r_error, 'valid': active.clone()}
        self.arm_target, profile = arm_control_profile(current, solution, self.arm_target, success, active)
        grips = gripper_profile(self.right.data.joint_pos[:, self.grip_ids], closed)
        changed = active & (closed != self.force_mode)
        changed_ids = torch.where(changed)[0]
        if len(changed_ids):
            self.right.write_joint_stiffness_to_sim(
                torch.where(closed[changed_ids, None], 0, self.nominal_stiffness[1][changed_ids]),
                joint_ids=self.grip_ids, env_ids=changed_ids)
        self.force_mode = torch.where(active, closed, self.force_mode)
        self.right.set_joint_effort_target(torch.where(active, -force, 0)[:, None].expand(-1, 2), joint_ids=self.grip_ids)
        frozen = self._frozen(torch.where(~active)[0])
        motion = BatchPoseMotion(self.scene['cube'].data.root_link_pose_w)
        nonfinite = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        for k in range(10):
            self.right.set_joint_position_target(torch.where(active[:, None], profile[k], self.arm_target), joint_ids=self.arm_ids)
            self.right.set_joint_position_target(gripper_drive_target(
                self.right.data.joint_pos[:, self.grip_ids], grips[k]), joint_ids=self.grip_ids)
            self._advance(frozen)
            motion.record(self.scene['cube'].data.root_link_pose_w)
            for obj in (self.scene['cube'], self.scene['plate']):
                nonfinite |= ~torch.isfinite(obj.data.root_state_w).all(-1)
            for art in self.arms:
                nonfinite |= (~torch.isfinite(art.data.joint_pos).all(-1)
                              | ~torch.isfinite(art.data.joint_vel).all(-1)
                              | ~torch.isfinite(art.data.body_link_state_w).all(-1).all(-1))
        self.linear, self.angular = motion.linear, motion.angular
        bad, fatal = self.physics_errors.poll()
        nonfinite[list(bad)] = True
        nonfinite |= fatal
        sample, pose, plate = self._sample()
        sample['step'] = self.episodes.steps+1
        false = torch.zeros_like(active)
        valid = self.episodes.update(sample, pose, closed, native_terminated=false,
            native_truncated=self.episodes.steps+1 >= 500, native_success=false,
            execution_error=nonfinite)
        batch = self._batch(sample, pose, plate, actions.clamp(-1, 1), force)
        batch.valid = valid.clone()
        self.history.append(self._row(sample, pose, plate, batch, valid))
        return batch

    def _batch(self, sample, pose, plate, actions, force):
        sample['cube_linear_velocity'] = self.linear
        actor, critic = policy_inputs(self.episodes.steps, self.decoder.pad_cmd, sample['opening'][:, 1],
            self.detection, pose, sample, plate[:, :3])
        actor = torch.where(self.episodes.error[:, None], 0, actor)
        critic = torch.where(self.episodes.error[:, None], 0, critic)
        net = torch.stack([self.scene.sensors[f'pad{i}'].data.net_forces_w[:, 0] for i in (2, 3)], 1)
        finger_q = self.right.data.body_link_quat_w[:, self.body_ids[1:]]
        tactile = pad_normal_forces(net, pad_normals(finger_q))
        public = {'time_s': self.episodes.steps*.04, 'joint_position': self.right.data.joint_pos[:, self.arm_ids].clone(),
            'joint_velocity': self.right.data.joint_vel[:, self.arm_ids].clone(), 'gripper_opening': sample['opening'][:, 1],
            'executed_action': actions, 'commanded_ee_pose': self.decoder.commanded_pose.clone(),
            'grip_force_command': force, 'ee_pose': pose, 'tactile_normal_force': tactile,
            'video_frame_index': torch.full_like(self.episodes.steps, -1)}
        truth = {'cube_position': sample['cube_position'], 'cube_quaternion': sample['cube_quaternion'],
            'cube_linear_velocity': self.linear, 'cube_angular_velocity': self.angular,
            'plate_position': plate[:, :3], 'plate_quaternion': plate[:, 3:], 'table_contact': sample['table'],
            'plate_contact': sample['plate'], 'robot_contact': sample['robot'], 'right_finger_contacts': sample['fingers'][:, 1],
            'stop_reason': self.episodes.reason.clone(), 'slip': self.episodes.slip_step.clone(),
            'lift_height': self.episodes.lift_height.clone(),
            'release_distance': (sample['cube_position'][:, :2]-plate[:, :2]).norm(dim=-1)}
        # Runtime keeps masks/outcomes on GPU; adapters serialize at episode chunk boundaries.
        host = {'strict': (self.episodes.reason == 1) & self.episodes.held_to_release,
                'lenient': self.episodes.reason == 1, 'reason': self.episodes.reason.clone(),
                'max_clear': self.episodes.max_clear.clone(), 'native_success': self.episodes.placement.success.clone()}
        batch = StepBatch(actor, critic, public, {} if self.target is not None else truth,
            self.episodes.done.clone(), ErrorRows(self.episodes.error),
            OutcomeRows(host, self.episodes.done, self.episodes.error), ScoringRows(host))
        batch.valid = torch.zeros_like(self.episodes.done)
        batch.episode_ids = list(self.episode_ids)
        return batch

    def _row(self, sample, pose, plate, batch, valid):
        # Host-private traces include target truth for frozen backend gates only.
        return {**{k: v.clone() for k, v in batch.public.items()},
                'cube_position': sample['cube_position'].clone(), 'cube_quaternion': sample['cube_quaternion'].clone(),
                'cube_linear_velocity': self.linear.clone(), 'cube_angular_velocity': self.angular.clone(),
                'plate_position': plate[:, :3].clone(), 'right_finger_contacts': sample['fingers'][:, 1].clone(),
                'contacts': torch.stack((sample['table'], sample['plate'], sample['robot']), -1),
                'episode_generation': self.generation.clone(), 'closed': self.force_mode.clone(), 'valid': valid.clone(), 'done': batch.done.clone(),
                'execution_error': self.episodes.error.clone(), 'reason': self.episodes.reason.clone(),
                'max_clear': self.episodes.max_clear.clone(), 'native_success': self.episodes.placement.success.clone()}

    def episode_chunk(self):
        """Single chunk transfer per field; no per-step JSON or per-env scoring."""
        return {key: torch.stack([row[key] for row in self.history]).cpu().numpy() for key in self.history[0]}

    def close(self):
        self._audit_subscription = None
        # Isaac Lab's STOP callback otherwise renders forever in standalone scripts (as build_simulation_context does).
        self.sim._disable_app_control_on_stop_handle = True
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.scene = None
        self.sim.clear_instance()
        from isaacsim.core.utils.stage import create_new_stage
        create_new_stage()
