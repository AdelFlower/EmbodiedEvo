# Cube to plate

Pick up the red cube, lift it clear of the table, carry it to the plate, release
it inside the plate, and withdraw so it rests stably. The host supplies policy P0.
P0 behaves differently in the source and the target.

All robot actions come from the neural policy. Request experiments through the
host API; do not send actions or change the host's task contract. The contract is
five clipped actions, 25 Hz control, a 300-step horizon, native scoring and fixed
strict/lenient outcomes, termination order and gripper scaling. See api/README.md
for the interface and evidence schema and source_reference/ for the nominal model.

Observe raw video, proprioception, executed commands, finger tactile responses,
and strict/lenient outcomes and execution status. Source experiments also expose
source ground truth. Use local scripts and plots to inspect this evidence.

Build a versioned source candidate, probe it with the current policy, compare it
with target evidence, and record a reproduction claim when justified. Then request
training in that frozen source with a reward program. Training continues the latest
complete policy for 100 iterations; a completed task receives a target judgment.
You decide when to revise, continue training or submit. A complete 4/4 strict target
judgment ends adaptation; thereafter read existing evidence and submit your diagnosis.
Only source data enter training. Keep target observations out of gradients,
demonstrations and normalization statistics. Do not use an observation-summary model.

Maintain notes/PROGRESS.md with evidence references, current hypotheses, decisions,
uncertainties and pending work. Author candidates under environments/, rewards under
rewards/, analysis under tools/, and the final diagnosis under submission/. Preserve
failed and partial attempts in your analysis. See the API for the diagnosis fields.
