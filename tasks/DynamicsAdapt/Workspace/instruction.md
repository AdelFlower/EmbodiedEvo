Pick up the red cube, lift it until its bottom is about 5 cm above the table, carry
it level over the plate centre, lower it about 4 cm, and open the gripper. Strict
success requires both fingers to hold the cube continuously from lift-off until
the gripper opens, followed by rest in the plate: centre within 4.5 cm of the plate
centre, inside the plate's height band, for 13 steps with no robot contact. Ends
in plate (lenient) is reported alongside strict success.

N000 succeeds in the nominal source but may fail in a target whose dynamics differ
in unknown ways. Diagnose the discrepancy, reconstruct it in the source, and
continue training there. You never send robot actions; the host runs registered
networks only. nominal.json contains nominal physics and the permitted ranges.
Omitted parameters take nominal values. Gripper scale remains 1.0.

The policy observes 33 numbers: a normalized time clock and sine/cosine clocks at
frequencies 1, 2, 4, 8 and 16; commanded pad position; cube and plate positions
relative to the observed pad; cube clearance; cube quaternion and linear velocity;
gripper opening; two finger contacts; and table and plate contact. The five
actions lie in [-1, 1]: the first three displace the pad by up to 1 cm per axis per
step, the fourth closes the gripper when positive, and the fifth sets squeeze
force to 25 + 20 times that value N while closed (5–45 N). Orientation follows a
fixed top-down schedule. Control runs at 25 Hz for at most 300 steps.

Training is PPO continuing the latest network and both optimizers. Exploration
and meaningful change happen only in gripper and force outputs; an imitation term
keeps motion near the original. Each iteration runs {{EPISODES_PER_ITER}} stochastic
episodes. The first {{WARMUP_ITERATIONS}} iterations train only the value function
when starting from N000 or changing the reward text. You choose physics parameters,
reward program and iteration count, and nothing else. Target data is for diagnosis
and selection only. The simulator is not bit-repeatable across successive episodes
on a worker, including deterministic episodes.

The default reward gives four times normalized progress in maximum held clearance
(capped at 5 cm), plus four times progress toward the plate after reaching 2 cm
clearance while off the table, minus a quarter point per millimetre of stepwise
slip when both fingers touched in the previous and current states. Success adds
10; dropping onto the table or failing to lift subtracts 3. See
scripts/default_reward.py for the exact supplied formula. You may write a Python
file defining reward(trace), returning one finite reward per step. NumPy and
frictionlib are available. Termination stays fixed. Each value must have magnitude
at most 10000; program size is at most 64 KiB. Programs run with no network, a
read-only filesystem except output, {{SANDBOX_MEMORY_MB}} MiB memory, one CPU and
{{SANDBOX_TIMEOUT}} seconds per batch. Test locally on a directory of exported
trajectory files, for example:

    python3 scripts/reward_runner.py rewards/my_reward.py observations/baseline/source/D0 notes/reward_check

Use scripts/adapt.py for all experiments. Create notes/, rewards/ and submission/
for your own files. Examples (notes/p.json is a JSON object of chosen physics):

    python3 scripts/adapt.py status
    python3 scripts/adapt.py probe --network N000 --parameters notes/p.json --episodes 0,2 --hypothesis "Lower friction" --prediction "Contact loss during carry"
    python3 scripts/adapt.py train --parameters notes/p.json --iterations 4 --reward default --rationale "Reproduced the target failure"
    python3 scripts/adapt.py train --parameters notes/p.json --iterations 4 --reward rewards/my_reward.py --rationale "Reward continued holding"
    python3 scripts/adapt.py wait J0001 --seconds 600
    python3 scripts/adapt.py submit submission/diagnosis.json

Only one job may run at a time. wait accepts 1–900 seconds. D0 and D1 use the
policy mean; D2 and D3 are stochastic with fixed episode seeds. Every deployment
runs D0–D3; probes use any nonempty subset. Limits: {{WALL_HOURS}} hours,
{{ROUNDS}} rounds, 4–{{ITERATIONS_PER_ROUND}} iterations per round,
{{ITERATIONS_TOTAL}} total iterations, {{TARGET_EPISODES}} target episodes and
{{DIAGNOSTIC_EPISODES}} diagnostic episodes. Refusals cost nothing. Accepted jobs
are charged at start even if they fail; target episodes are charged at deployment.
A reward dry run uses all four baseline source traces and reports their reward
sums. No requests are accepted after submission.

Reproduce before training (enforced): the parent is the latest network and must
have at least one strict target failure. A completed probe of that parent at
exactly the requested, nominal-filled parameters must match the parent's strict
target outcome on every probed episode and include at least one target failure.
Record a hypothesis and prediction for every probe. Test at least one competing
hypothesis. Compare failure modes as well as outcomes. Keep failed attempts.

Read observations/baseline/{source,target}/D0 through D3. Job exports are under
observations/jobs/J0001/: request.json, episodes/Dk/ for probes, and training/ plus
deploy/Dk/ for training. trajectory.npz rows are post-step states, with exactly:
commanded_ee_pose, observed_ee_pose, action (executed clipped five outputs), closed,
grip_force_command, right_grip_force, gripper_opening, right_finger_contacts,
contacts (table, plate, robot), cube_position, cube_quaternion, plate_position,
initial_observed_ee_pose, initial_cube_position, initial_cube_quaternion,
initial_right_finger_contacts, and scalar stop_reason. Use frictionlib.load_trace
to decode it. Initial arrays and plate_position have no time dimension.
result.json reports strict_success, success_lenient, held_to_release, stop_reason,
steps, max_clear, slip, force_mean_held, force_min_held, release_dist,
cmd_peak_speed and cmd_peak_accel. Raw cam_head.mp4, cam_left_wrist.mp4 and
cam_right_wrist.mp4 are present when available. Training exports iterations.csv
with iteration, warmup, strict_successes, lenient_successes, mean_return,
mean_max_clear, mean_slip, mean_force_mean_held, approx_kl, policy_loss and value_loss;
last_iteration/Ek/ contains the final episodes and rewards.npy. Failed episodes
may have only partial evidence. Instruction, nominal.json, scripts and observations
are read-only.

The host selects among N000 and deployed networks by most strict development
successes, then most lenient successes, then earliest network. Your preference is
recorded but does not select. Finish by submitting this diagnosis schema:

    {
      "estimated_target_parameters": {"friction_scale": 1.0, "mass_scale": 1.0, "restitution": 0.0},
      "conclusion": "Your conclusion and uncertainty",
      "evidence_jobs": ["J0001"],
      "competing_hypotheses": [
        {"hypothesis": "Alternative", "test_jobs": [], "outcome": "untested", "reason": "Why unresolved"}
      ],
      "limitations": "What the observations cannot establish",
      "preferred_network": "N000",
      "reward_rationale": "Why the chosen reward should help"
    }

Competing outcomes are rejected, not_rejected or untested.
