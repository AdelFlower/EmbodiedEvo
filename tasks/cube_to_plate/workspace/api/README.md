# Host API

Every request is an object with `op` and exactly the corresponding fields below.
Side-effecting requests also require a nonempty `request_id`. Reusing an ID with
identical content returns the original response, including rejection. Changed
content with the same ID is rejected. One experiment job runs at a time.

| op | Additional fields |
|---|---|
| status | none |
| build | request_id, candidate, hypothesis |
| probe | request_id, policy_id, environment_id, prediction, layouts |
| deploy | request_id, policy_id, purpose |
| reproduce | request_id, policy_id, environment_id, source_jobs, target_jobs, comparison, uncertainty |
| train | request_id, environment_id, claim_id, reward, rationale, environment_round_id |
| wait | job_id, seconds |
| submit | request_id, diagnosis |

Strings are required except: layouts is a nonempty unique list of indices 0–3;
source_jobs and target_jobs are nonempty lists of job IDs; environment_round_id is
null for a new round or a round ID for continuation; seconds is an integer 1–900.
Paths are workspace-relative: candidate under environments/, reward under rewards/,
diagnosis under submission/. Candidates contain regular files, no symlinks, at most
64 MiB; reward files are at most 64 KiB.

Build returns an environment ID and ready/capability_unavailable status. Probe runs
the registered policy with mean actions on the chosen source layouts. Deploy runs
four mean-action target episodes. Reproduce requires completed source probe and
target evidence of the same policy, with the probe in the claimed environment;
it returns a claim ID. A new round requires a claim for the latest policy. A
continuation requires the round's environment, claim and uninterrupted lineage.
Training freezes the environment and reward and continues the latest complete PPO
state, both optimizers and counters. Each task executes 100 iterations of 512
episodes; the first task and each reward change begin with two critic-only iterations.
Reward precheck rejection registers no task. A training job returns a job ID and
round ID. Wait returns public job status/artifacts. Status includes versions,
claims, rounds, jobs, progress and usage. No usage counter imposes a research budget.

A complete 4/4 strict target batch ends adaptation. Only status, wait and submit
remain available during recovery wrap-up. Submit is rejected while a job runs.
Submit accepts a JSON diagnosis with exact keys: conclusion (string),
mechanism_hypotheses (nonempty string list), environment_ids, claim_ids, evidence_jobs
(existing ID lists), competing_explanations (list of objects with explanation,
test_jobs, status = supported/rejected/untested, reason), uncertainty (string),
reward_rationale (string), preferred_policy (registered ID or null), and
parameter_estimates (object mapping names to objects with numeric value and string unit).

# Evidence: cube_to_plate_evidence_v1

Per-episode traces contain parallel `public` and (source only) `source_truth` row
lists. Row 0 is reset, then one row follows each executed action. A T-action trace
has T+1 rows. Vectors have no replica axis in an individual episode; runtimes batch
these fields on axis 0. Time and video_frame_index align with control-step sampling.
Quaternions are scalar-first; poses are xyz followed by quaternion in the world
frame. Distances, time, force and velocities use SI units.

| Public step field | Meaning |
|---|---|
| time_s | seconds since reset, step / 25 |
| joint_position, joint_velocity | right arm, joint order joint1–joint6, rad and rad/s |
| gripper_opening | right gripper encoder, 0–1 |
| executed_action | five actions after clip to [-1,1]; reset row is zeros |
| commanded_ee_pose | world pose (7), initial pose at reset |
| grip_force_command | commanded force (N), zero at reset |
| ee_pose | world end-effector pose (7), from proprioception |
| tactile_normal_force | two finger pads' compression forces (N), all partners, no identities |
| video_frame_index | corresponding frame index, or null for runs without video |

Outcome fields: strict, lenient, execution_status (complete or execution_error).
An execution error has false strict/lenient. Cancellation is a batch interruption,
not a completed outcome. Exported episode envelopes also carry layout_id, steps
and host execution status.

Source truth fields: cube_position (3), cube_quaternion (4), cube_linear_velocity
(3), cube_angular_velocity (3); plate_position (3), plate_quaternion (4);
table_contact, plate_contact, robot_contact (cube contacts, booleans),
right_finger_contacts (two cube–finger booleans); stop_reason (null until terminal),
slip (m, hand-frame cube-offset change since previous row, zero unless both pads
touch the cube in both rows), lift_height (m, instantaneous lowest cube corner
above the table), release_distance (m, horizontal cube-to-plate centre distance).
Reset slip is zero. Default reward returns T rewards from this T+1-row schema.

Target exports contain public fields and outcomes only. Policy inputs, detection
truth/error, object truth, scoring internals and physical target specifications are
host-private. Raw episode length may convey information through behavior; the host
preserves termination and raw evidence. Source model candidates may describe added
structure, state, geometry, resources and dynamics programs; support is reported by
build, which never silently substitutes another environment.

# Fixed task contract

Policy actions are pad_dx, pad_dy, pad_dz, close_if_positive, squeeze_force.
They clip to [-1,1] and use the decoder described in source_reference/robot.md.
Control is 25 Hz, horizon 300 steps and gripper multiplier 1.0.

Lenient: DP success: after an open command following a close, 13 consecutive steps with cube centre within 0.045 m horizontally of plate centre, lowest corner between plate bottom - 0.002 m and rim + 0.003 m, plate contact, no table/robot/finger contact and opening >= 0.8; or terminated native success.

Strict: DP success and uninterrupted bilateral finger contact from the first two-finger lift with neither table nor plate contact and clearance > 0.002 m through the first subsequent close-to-open command transition.

Termination checks run in this order:

1. success: DP rest or terminated native success
2. native_end: native termination/truncation
3. dropped: table contact after maximum held clearance >= 0.005 m
4. no_lift: step >= 160 and maximum held clearance < 0.005 m
5. timeout: step == 300

Native scoring:

Success is judged at every control step from the cube's pose and the simulator's contact reports. The table top is at z = 0.765 m and the plate's rim at z = 0.8006 m.

1. **Grasp and carry.** Both fingers of one gripper touch the cube while its lowest corner is at least 3 cm above the table top and it touches neither the table nor the plate. This airborne two-finger grasp must last at least 5 consecutive control steps and move the cube at least 5 cm horizontally from where it began. If any of these conditions lapses, the count and the starting point begin again.
2. **Release over the plate.** At the first step after the carry when neither finger of the carrying gripper touches the cube, every corner of the cube must lie horizontally within 4 cm of the plate's centre, the cube must not touch the table, and its lowest corner must be no more than 5 cm above the rim. A cube that is tilted or off-centre when the fingers lose contact can fail this check even if it then settles in the middle of the plate. If the check fails, the carry no longer counts, and only a new grasp and carry can count again. Losing finger contact anywhere else, for example by slipping during the carry, is treated the same way.
3. **Rest.** The cube must then rest on the plate for 13 consecutive control steps: touching the plate but not the table, every corner still within 4 cm of the centre, its lowest corner no higher than 3 mm above the rim, no robot link touching it, the carrying gripper at least 80% open, and that gripper's fingers at least 5 cm from the cube's centre.

Touching the table after a carry cancels that carry. The cube's speed is not checked.
