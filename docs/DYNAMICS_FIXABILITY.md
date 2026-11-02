# Dynamics fixability study

FX1 implements the [study specification](DYNAMICS_FIXABILITY_SPEC.md). The overseer
runs all GPU stages. This is a separate privileged-state study, outside RoboDojo Test.
Use `.venv-rl`; no agent or model calls are involved.

Every command requires a fresh, timestamp-prefixed `--run-dir`. Failed and incomplete
attempts remain there. Commands neither resume nor retry. Freeze the configuration
before S; use it unchanged for A. The actor uses columns 0–48, the critic all 57.
The Gaussian variance is fixed; PPO clips sampled controls, warms only the critic for
its first two rollouts after loading, and backtracks actor updates by exact KL.

The commands below use example timestamped run directories. Replace them with new
run names. `S` and `A` point to the selected network directories from each run's
`selection.json`; `TARGET` is chosen by the transfer rule below.

```bash
fx() { .venv-rl/bin/python -m orchestrator.dynamics_fixability "$@"; }
C=tasks/DynamicsRL/fixability.example.json
GPU=0
```

1. Capacity: prepare three copies of C, changing only `num_envs` to 1, 4 and 8.
   Run sequentially; divide total actions in `feasibility.json` by its
   `elapsed_seconds` for end-to-end throughput, including startup and resets.
   Set `num_envs` in C from the measurement, then freeze C.

```bash
for N in 1 4 8; do
  fx feasibility --config /path/to/fx-capacity-$N.json --friction-scale 1.0 \
    --carry-time-scale 1 --episodes 20 --gpu "$GPU" \
    --run-dir runs/202611020000_fx_capacity_$N
done
```

2. S1–S3: noisy source demonstrations, clean-label BC, report-only BC evaluation,
   then source PPO. BC reads only complete successful demonstration attempts.

```bash
fx demos --config "$C" --friction-scale 1.0 --gpu "$GPU" --run-dir runs/202611020000_fx_demos
fx bc --config "$C" --demonstrations runs/202611020000_fx_demos --run-dir runs/202611020000_fx_bc
fx evaluate --config "$C" --network runs/202611020000_fx_bc --friction-scale 1.0 \
  --episodes 20 --gpu "$GPU" --run-dir runs/202611020000_fx_bc_eval
fx train --config "$C" --init runs/202611020000_fx_bc --friction-scale 1.0 \
  --gpu "$GPU" --run-dir runs/202611020000_fx_source
S=runs/202611020000_fx_source/networks/Rrr
```

3. Source gate: require deterministic ≥48/50 and stochastic ≥45/50. Otherwise stop.

```bash
fx evaluate --config "$C" --network "$S" --friction-scale 1.0 --episodes 50 \
  --render 5 --gpu "$GPU" --run-dir runs/202611020000_fx_source_gate
fx evaluate --config "$C" --network "$S" --friction-scale 1.0 --episodes 50 \
  --stochastic --gpu "$GPU" --run-dir runs/202611020000_fx_source_noise
```

4. T: target = highest friction in {0.3, 0.25} with source success ≤25/50.
   If neither qualifies, stop. All evaluation uses the same `fix_eval` layouts.
   Compare P0 deterministically on those layouts at each friction too.

```bash
for F in 0.3 0.25; do
  fx evaluate --config "$C" --network "$S" --friction-scale "$F" --episodes 50 \
    --render 5 --gpu "$GPU" --run-dir runs/202611020000_fx_transfer_$F
done
P0=/path/to/best_rl.pt
for F in 1.0 0.3 0.25; do
  fx evaluate --config "$C" --network "$P0" --friction-scale "$F" --episodes 50 \
    --gpu "$GPU" --run-dir runs/202611020000_fx_p0_$F
done
TARGET=0.3  # Replace with the qualifying target selected above.
```

5. F and A: feasibility is reference only; it never supplies BC data. Scales 1 and 3
   use 345 and 475 scripted actions respectively, within the native 500-action limit.

```bash
for SCALE in 1 3; do
  fx feasibility --config "$C" --friction-scale "$TARGET" --carry-time-scale "$SCALE" \
    --episodes 20 --gpu "$GPU" --run-dir runs/202611020000_fx_feasibility_$SCALE
done
fx train --config "$C" --init "$S/training.zip" --friction-scale "$TARGET" \
  --gpu "$GPU" --run-dir runs/202611020000_fx_adapt
A=runs/202611020000_fx_adapt/networks/Rrr
```

6. Verdict: deterministic target ≥45/50 means fixable by RL. Otherwise report the
   improvement paired by layout. Also report stochastic target performance and
   deterministic source retention. Gates and target choice are overseer decisions.

```bash
fx evaluate --config "$C" --network "$A" --friction-scale "$TARGET" --episodes 50 \
  --render 5 --gpu "$GPU" --run-dir runs/202611020000_fx_verdict
fx evaluate --config "$C" --network "$A" --friction-scale "$TARGET" --episodes 50 \
  --stochastic --gpu "$GPU" --run-dir runs/202611020000_fx_adapt_noise
fx evaluate --config "$C" --network "$A" --friction-scale 1.0 --episodes 50 \
  --gpu "$GPU" --run-dir runs/202611020000_fx_retention
```

Artifacts: every run has `config.json`, `run.json` and its layout plans. Demonstrations
have `demonstrations.json` plus per-attempt `transitions.npz` (executed `actions`, clean
`labels`), summaries and noise levels. BC writes `bc.jsonl`. Each network has
`training.zip`, `policy.pt`, `network.json`. Training writes `online-episodes.jsonl`,
`rl-metrics.jsonl`, `rounds.jsonl` and `selection.json`. Selection orders by dev successes,
then online success rate (R00 = −1), then earliest round. Online rates exclude incomplete
and unused auto-reset episodes, which remain in the evidence. Wall-time checks happen
between rounds and rollouts; completed updates in a shortened final round are saved and
evaluated. No checkpoint is selected using held-out results.

The simulator pool remains alive through training and dev evaluation. Partial online
episodes close before evaluation, and the next round resets the vector environment.
`evaluation.json` separates state-mode scores from rendered replay records. Stochastic
inference uses a seeded NumPy stream; thread scheduling can change which episode receives
each draw. P0 supports deterministic evaluation only. Feasibility records live separately
in `feasibility.json`. Exceptions retain episode evidence and propagate to `error.json`.

CPU validation:

```bash
.venv-rl/bin/python -m unittest tests.test_dynamics_fixability -v
.venv-rl/bin/python -m unittest tests.test_adaptation_training tests.test_dynamics_rl_learning tests.test_dynamics_rl_features -v
.venv-rl/bin/python -m unittest discover -s tests -v
```
