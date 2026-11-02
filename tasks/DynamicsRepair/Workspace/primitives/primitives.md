# Robot Primitives: workspace-session-v1

This is the handwritten control-interface manual. Submitted Python runs in a
persistent host-owned session. Exactly the following four functions are
environment primitives. `scripts/env.py` and `scripts/dynamics.py` are workspace
CLI clients, not additional robot primitives.

Control code receives restricted Python builtins and `np` operations.
Restricted `numpy` and `math` imports are supported, including public numerical
members imported with `from ... import ...`. Private attributes and other modules
are unavailable. File access, networking, simulator internals, and GPU access are
unavailable. Use the workspace shell to inspect exported raw observations.
Submitted control code, including comments, must use ASCII.
Use f-strings or `%` formatting for text. The `format` and `format_map` methods
are unavailable in submitted code.
Class patterns support public keyword attributes; positional class patterns are
unavailable. Mapping and sequence patterns remain supported.

## get_instruction()

Returns the current task instruction as a string without advancing physics.
Follow it to grasp, carry, and place the red cube in the plate. Pushing the cube
into the plate, holding it in the air, or continuing to support it with the
gripper does not complete the task.

## get_observation()

Returns a detached copy of the latest observation without advancing physics.
Editing returned arrays does not change simulator state.

- `instruction`: the current task instruction.
- `vision`: camera dictionaries, normally `cam_head`, `cam_left_wrist`, and
  `cam_right_wrist`. Each `color` is a native RGB image, normally a 480 by 640 by 3
  uint8 array; `shape` records the actual dimensions.
- `state`: measured arm joints, absolute end-effector poses, and the controller's
  normalized gripper command state.
- `action`: the native joint/gripper command dictionary.
- `additional_info.frequency`: the native policy control frequency, 25 Hz.

The default dual-arm ARX X5 has six joints per arm. State keys are
`left_arm_joint_state`, `right_arm_joint_state`, `left_ee_joint_state`,
`right_ee_joint_state`, `left_ee_pose`, and `right_ee_pose`. Read actual array
shapes instead of assuming dimensions from memory. End-effector poses are
`[x, y, z, qw, qx, qy, qz]` in the native world frame, with positions in metres,
scalar-first quaternions, and +z pointing up. This is the same end-effector frame
used by EE action targets.

Image pixels are not world coordinates. This interface does not expose depth,
camera calibration, object ground truth, contact forces, or tactile feedback.
Gripper state is not a grip-force reading and cannot be used as a contact sensor.

## step(action)

Accepts a complete native action dictionary. Provide finite one-dimensional
arrays or lists for every controlled arm and gripper. Select either joint targets
or end-effector pose targets for an action; do not mix modes.

Example holding the current joints:

```python
obs = get_observation()
action = {k: v for k, v in obs['state'].items() if k.endswith('joint_state')}
obs, reward, terminated, truncated, info = step(action)
```

Example holding the current end-effector poses:

```python
obs = get_observation()
s = obs['state']
action = {
    'left_ee_pose': s['left_ee_pose'],
    'right_ee_pose': s['right_ee_pose'],
    'left_ee_joint_state': s['left_ee_joint_state'],
    'right_ee_joint_state': s['right_ee_joint_state'],
}
obs, reward, terminated, truncated, info = step(action)
```

Arm joint targets are absolute radians. EE targets are absolute world poses,
not displacement increments. Gripper values are one-element normalized arrays:
0 closed and 1 open; the native controller clamps them to that range.
There is no direct grip-force, stiffness, or actuator-torque control. Native IK
handles EE targets and may leave the arm unchanged if it cannot solve a target.
Each call executes one 25 Hz control interval, or 10 physics ticks at 250 Hz.
No automatic planner or grasp primitive is exposed.

Returns `(observation, reward, terminated, truncated, info)`.
`reward` is the binary task-completion signal, and `info['success']` reports the
same result. Private intermediate rewards and object state used for scoring are
not exposed. Failure can also terminate an episode.

This custom task permits at most 500 native actions per episode. Calls after
termination or truncation do not advance physics; check `terminated` and
`truncated` and stop taking actions. The session also has outer limits on execution
requests, episode counts, and cumulative actions. Query them with
`python scripts/dynamics.py status`; do not plan solely from the current episode's
remaining actions. Raw images, robot state, and per-action RGB videos are exported
automatically after each code execution.

## reset()

Allowed only during this research workflow's Playground learning/repair stages.
Forbidden during final frozen evaluation and the original protocol's Test.
Restores the current trial's same initial task state and returns
`(observation, info)`. Accepts no seed or scene arguments. Preserves Python
variables, functions, and the current trial's parameter overrides.

Reset consumes another global episode; in a target trial it also consumes a
target episode. It replenishes the episode's 500-action allowance but not
cumulative execution requests, cumulative actions, or elapsed time.
A standalone `python scripts/env.py reset` consumes one execution request. A
reset inside code belongs to that execution request but still consumes an episode.

For an empty Python namespace, create a new trial with the workspace command
`scripts/dynamics.py start`. It ends the previous trial and creates a new
simulator, without replenishing global budgets.
