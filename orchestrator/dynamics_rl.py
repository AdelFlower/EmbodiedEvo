"""Train and evaluate a privileged-state cube-to-plate neural policy."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from integrations.dynamics_rl.client import RemoteEnv, SimulatorClient
from integrations.dynamics_rl.data import (
    EpisodeData,
    layout_seeds,
    load_demonstrations,
    scene_plan,
)
from integrations.dynamics_rl.features import contract, sample_scene
from integrations.dynamics_rl.teacher import TEACHER_VERSION, R1Teacher
from orchestrator.files import append_jsonl, run_lock, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tasks/DynamicsRL/train.example.json"


class BudgetExpired(RuntimeError):
    pass


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def load_config(path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    if path:
        overrides = json.loads(Path(path).read_text())
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown RL settings: {sorted(unknown)}")
        config.update(overrides)
    for key in ("demonstration_attempts", "validation_episodes", "evaluation_episodes",
                "required_successes", "reset_checks", "bc_updates", "online_steps",
                "validation_interval", "checkpoint_interval", "learner_threads", "batch_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_train_hours", "startup_timeout", "step_timeout", "learning_rate",
                "entropy_coefficient", "actor_learning_rate", "bc_weight", "bc_final_weight",
                "bc_online_rotation_weight"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["gamma"] != 0.99:
        raise ValueError("Worker shaping and SAC both use gamma=0.99")
    if not 0 <= config["position_radius"] <= 0.03:
        raise ValueError("position_radius must be in [0, 0.03]")
    if config["required_successes"] > config["evaluation_episodes"]:
        raise ValueError("Required successes cannot exceed evaluation episodes")
    if config["bc_mode"] not in {"classic", "robust"}:
        raise ValueError("bc_mode must be classic or robust")
    if not math.isfinite(config["initial_log_std"]) or not -20 <= config["initial_log_std"] <= -2:
        raise ValueError("initial_log_std must be finite and in [-20, -2]")
    if config["bc_final_weight"] > config["bc_weight"]:
        raise ValueError("BC auxiliary weight must decay")
    limit = config["actor_log_std_max"]
    if limit is not None and (not math.isfinite(limit) or not -20 <= limit <= 2):
        raise ValueError("actor_log_std_max must be null or finite and in [-20, 2]")
    return config


def new_run(path, command, config):
    path = Path(path).resolve()
    if not re.match(r"^\d{12}(?:_|$)", path.name):
        raise ValueError("--run-dir must start with YYYYMMDDHHMM")
    if path.exists() and any(item.name != ".protocol.lock" for item in path.iterdir()):
        raise FileExistsError("A new run cannot overwrite existing evidence")
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "config.json", config)
    write_json(path / "contract.json", contract())
    write_json(path / "run.json", {
        "command": command, "started_at": time.time(), "formal_test": False,
        "experiment": "dynamics-rl-v1", "model_calls": 0,
    })
    sources = [
        *list((ROOT / "integrations/dynamics_rl").glob("*.py")),
        ROOT / "orchestrator/dynamics_rl.py", ROOT / "orchestrator/dynamics_rl_worker.py",
        ROOT / "simulators/dynamics_rl.py", ROOT / "simulators/dynamics_task.py",
        ROOT / "simulators/dynamics_native.py", ROOT / "simulators/dynamics_scoring.py",
        ROOT / "requirements-rl.txt",
    ]
    for source in sources:
        destination = path / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return path


def simulator(run, name, config, gpu, mode="state", *, parameters):
    return SimulatorClient(
        run / f"{run.name[:12]}_{name}", gpu=gpu, mode=mode,
        simulator_python=config["simulator_python"],
        startup_timeout=config["startup_timeout"], step_timeout=config["step_timeout"],
        parameters=parameters,
    )


def check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExpired("The recorded training wall-time budget has expired")


def validate_preflight(path, config):
    path = Path(path).resolve()
    evidence = json.loads((path / "preflight.json").read_text())
    if not evidence["passed"]:
        raise ValueError("Supplied preflight did not pass")
    if json.loads((path / "contract.json").read_text()) != contract():
        raise ValueError("Supplied preflight uses a different observation/action contract")
    original = json.loads((path / "config.json").read_text())
    keys = ("simulator_python", "teacher_policy", "reset_checks", "position_radius", "gamma")
    if any(original[key] != config[key] for key in keys):
        raise ValueError("Supplied preflight configuration differs from this experiment")
    return evidence


def run_episode(client, scene, name, *, policy=None, deadline=None):
    """One host-controlled episode, retaining successful, failed and partial data."""
    folder = client.run_dir / name
    data, trace = EpisodeData(), []
    result = {"complete": False, "success": False, "actions": 0,
              "episode_dir": str(folder)}
    started = time.monotonic()
    try:
        check_time(deadline)
        response = client.reset(scene, folder)
        teacher = R1Teacher(response["state"]) if policy is None else None
        if teacher:
            write_json(folder / "teacher.json", teacher.metadata())
        trace.append(response["state"]["sample"]["cube_position"])
        while not response["terminated"] and not response["truncated"]:
            check_time(deadline)
            action = (teacher.act(response["state"]) if teacher else
                      policy.predict(np.asarray(response["obs"], np.float32),
                                     deterministic=True)[0])
            following = client.step(action)
            data.append(response, action, following)
            response = following
            trace.append(response["state"]["sample"]["cube_position"])
        result.update(complete=True, success=bool(response["info"]["success"]),
                      terminated=response["terminated"], truncated=response["truncated"],
                      actions=len(data.rows), reward=float(data.arrays()["rewards"].sum()),
                      final_state=response["state"], videos=response["info"].get("videos", {}))
    except BaseException as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                      classification="incomplete", actions=len(data.rows))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        folder.mkdir(parents=True, exist_ok=True)
        data.save(folder / "transitions.npz")
        np.save(folder / "cube-trajectory.npy", np.asarray(trace))
        write_json(folder / "episode-summary.json", result)
    return result


def preflight(run, config, gpu, *, deadline=None):
    """Rendered/state parity, then twenty retained-body randomized reset checks."""
    result = {"passed": False, "checks": [], "render_success": False, "state_success": False}
    try:
        nominal = sample_scene(0, radius=0)
        parameters = {"friction_scale": 1.0, "mass_scale": 1.0, "restitution": 0.0}
        with simulator(run, "preflight_render", config, gpu, "render",
                       parameters=parameters) as rendered:
            record = run_episode(rendered, nominal, "nominal", deadline=deadline)
            result["render_success"] = record["success"]
            rendered_trace = np.load(Path(record["episode_dir"]) / "cube-trajectory.npy")
        with simulator(run, "preflight_state", config, gpu, parameters=parameters) as state_client:
            record = run_episode(state_client, nominal, "nominal", deadline=deadline)
            result["state_success"] = record["success"]
            state_trace = np.load(Path(record["episode_dir"]) / "cube-trajectory.npy")
            result["trajectory_shape_equal"] = rendered_trace.shape == state_trace.shape
            result["maximum_position_difference"] = (
                float(np.max(np.abs(rendered_trace - state_trace)))
                if rendered_trace.shape == state_trace.shape else None
            )
            if (not result["render_success"] or not result["state_success"]
                    or result["maximum_position_difference"] is None
                    or result["maximum_position_difference"] > 1e-4):
                raise RuntimeError("Rendered/state teacher parity failed; inspect evidence")
            for index, item in enumerate(scene_plan("demonstration", config["reset_checks"],
                                                    config["seed"] + 1000)):
                check_time(deadline)
                response = state_client.reset(
                    item["scene"], state_client.run_dir / f"reset_{index:03d}",
                )
                state = response["state"]
                cube_expected = item["scene"]["Rigid"]["cube"][0]["default_pos"]
                plate_expected = item["scene"]["Rigid"]["plate"][0]["default_pos"]
                cube_error = float(np.linalg.norm(np.asarray(state["sample"]["cube_position"])
                                                  [:2] - np.asarray(cube_expected)[:2]))
                plate_error = float(np.linalg.norm(np.asarray(state["plate_position"])
                                                   [:2] - np.asarray(plate_expected)[:2]))
                scoring = state["scoring"]
                passed = (cube_error < 0.002 and plate_error < 0.002
                          and response["info"]["steps"] == 0
                          and not scoring["transport_completed"]
                          and not scoring["released_over_plate"] and scoring["stable_steps"] == 0
                          and np.linalg.norm(state["sample"]["cube_solver_linear_velocity"]) < 0.02
                          and np.linalg.norm(state["right_arm_joint_velocity"]) < 0.1)
                result["checks"].append({
                    "index": index, "passed": bool(passed),
                    "cube_xy_error": cube_error, "plate_xy_error": plate_error,
                })
                write_json(run / "preflight.json", result)
                if not passed:
                    raise RuntimeError(f"Random reset {index} did not restore the requested scene")
                # Move away from reset before the next check, with native dynamics intact.
                teacher = R1Teacher(state)
                for _ in range(20):
                    response = state_client.step(teacher.act(response["state"]))
                emit("reset_check", index=index + 1, total=config["reset_checks"])
        result["passed"] = True
    except BaseException as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        write_json(run / "preflight.json", result)
    return result


def save_checkpoint(model, directory, label, metadata):
    from integrations.dynamics_rl.learning import export_policy

    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / f"{label}.zip")
    export_policy(model, directory / f"{label}.pt", metadata)
    write_json(directory / f"{label}.json", {
        "label": label, "online_steps": int(model.num_timesteps),
        "rl_updates": int(model._n_updates),
        "kind": "rl" if model._n_updates > 0 else "behavior_cloning_only",
        "metadata": metadata,
    })


def validate(client, policy, plan, label, run, deadline=None):
    records = []
    for item in plan:
        result = run_episode(client, item["scene"], f"validation/{label}/{item['index']:03d}",
                             policy=policy, deadline=deadline)
        records.append({key: result[key] for key in ("success", "complete", "actions", "reward")})
        emit("validation_episode", checkpoint=label, index=item["index"], **records[-1])
    result = {"label": label, "successes": sum(row["success"] for row in records),
              "episodes": len(records), "records": records}
    append_jsonl(run / "validation.jsonl", result)
    return result


def train(run, config, gpu, *, skip_preflight=False):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from integrations.dynamics_rl.data import import_demonstration_prefix
    from integrations.dynamics_rl.learning import bc_pretrain, create_model

    started = time.monotonic()
    deadline = started + config["max_train_hours"] * 3600
    torch.set_num_threads(config["learner_threads"])
    torch.manual_seed(config["seed"])
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    metadata = {"contract": contract(), "teacher_version": TEACHER_VERSION,
                "training_config": config, "training_run": str(run)}
    result = {"status": "started", "accepted": False, "online_steps": 0,
              "best_validation_successes": None}
    model = env = None
    original_error = None
    cleanup_errors = []
    try:
        source = Path(config["teacher_policy"]).resolve()
        archive = json.loads(source.read_text())
        write_json(run / "supplied-teacher.json", archive)
        write_json(run / "teacher-provenance.json", {
            "path": str(source), "source_kind": "supplied_development_policy",
            "adaptation": TEACHER_VERSION,
        })
        demo_plan = scene_plan("demonstration", config["demonstration_attempts"],
                               config["seed"], config["position_radius"])
        validation_plan = scene_plan("validation", config["validation_episodes"],
                                     config["seed"], config["position_radius"])
        write_json(run / "demonstration-scenes.json", demo_plan)
        write_json(run / "validation-scenes.json", validation_plan)
        demonstrations = []
        if config.get("demonstration_source") is not None:
            demonstrations = import_demonstration_prefix(
                config["demonstration_source"], run, demo_plan, config,
                teacher_version=TEACHER_VERSION, observation_contract=contract(),
                teacher_archive=archive,
            )
            result["inherited_demonstration_attempts"] = len(demonstrations)
            emit("demonstration_inheritance", attempts=len(demonstrations),
                 incomplete=sum(not row["complete"] for row in demonstrations),
                 remaining=len(demo_plan) - len(demonstrations))
        if not skip_preflight:
            preflight(run, config, gpu, deadline=deadline)
        with simulator(run, "training", config, gpu, parameters={
            "friction_scale": 1.0, "mass_scale": 1.0, "restitution": 0.0,
        }) as client:
            for item in demo_plan[len(demonstrations):]:
                name = f"demonstrations/{item['index']:03d}"
                folder = client.run_dir / name
                record = None
                try:
                    record = run_episode(client, item["scene"], name, deadline=deadline)
                except BaseException as exc:
                    summary = folder / "episode-summary.json"
                    record = json.loads(summary.read_text()) if summary.exists() else {
                        "success": False, "complete": False, "actions": 0,
                        "episode_dir": str(folder), "elapsed_seconds": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    raise
                finally:
                    # The failing attempt belongs in the manifest as well as its raw folder.
                    if record is not None:
                        record = {key: record[key] for key in (
                            "success", "complete", "actions", "episode_dir", "elapsed_seconds",
                            "error", "classification",
                        ) if key in record}
                        record["data_path"] = str(
                            (folder / "transitions.npz").relative_to(run)
                        )
                        record["layout_seed"] = item["layout_seed"]
                        demonstrations.append(record)
                        write_json(run / "demonstrations.json", demonstrations)
                        emit("demonstration", attempt=item["index"] + 1,
                             total=len(demo_plan), success=record["success"],
                             successes=sum(r["success"] for r in demonstrations))
            all_data, successful = load_demonstrations(run)
            if not len(successful["obs"]):
                raise RuntimeError("No successful demonstrations; BC cannot start")
            # Sampling 100,000 full scene dictionaries wastes time and memory. Preserve
            # the same independent stream as scene_plan("online") and build on reset.
            online_seeds = layout_seeds("online", config["online_steps"] + 1, config["seed"])
            write_json(run / "online-layout-seeds.json", online_seeds)

            def episode_finished(record):
                append_jsonl(run / "online-episodes.jsonl", record)
                emit("online_episode", episode=record["episode"], success=record["success"],
                     actions=record["actions"], reward=record["reward"])

            env = RemoteEnv(client, scene_seeds=online_seeds,
                            radius=config["position_radius"],
                            episode_root=client.run_dir / "online", on_episode=episode_finished)
            model = create_model(env, all_data, successful, seed=config["seed"],
                                 bc_mode=config["bc_mode"],
                                 initial_log_std=config["initial_log_std"],
                                 actor_log_std_max=config["actor_log_std_max"],
                                 bc_online_rotation_weight=config["bc_online_rotation_weight"],
                                 actor_learning_rate=config["actor_learning_rate"],
                                 bc_weight=config["bc_weight"],
                                 bc_final_weight=config["bc_final_weight"],
                                 batch_size=config["batch_size"],
                                 learning_rate=config["learning_rate"], gamma=config["gamma"],
                                 ent_coef=config["entropy_coefficient"])
            model.set_logger(configure(str(run / "learner"), ["csv"]))

            def bc_progress(metrics):
                append_jsonl(run / "bc.jsonl", metrics)
                if metrics["updates"] % 1000 == 0:
                    emit("behavior_cloning", **metrics)
                return time.monotonic() < deadline

            bc_metrics = bc_pretrain(model, successful["obs"], successful["actions"],
                                     updates=config["bc_updates"], batch_size=config["batch_size"],
                                     callback=bc_progress, seed=config["seed"])
            write_json(run / "bc-result.json", bc_metrics)
            save_checkpoint(model, checkpoints, "bc", metadata)
            result["bc_checkpoint"] = str(checkpoints / "bc.pt")
            check_time(deadline)

            def measure(label):
                append_jsonl(run / "validation-attempts.jsonl", {
                    "label": label, "status": "started", "episodes": len(validation_plan),
                    "online_steps": int(model.num_timesteps),
                })
                try:
                    measured = validate(client, model, validation_plan, label, run, deadline)
                except BaseException as exc:
                    append_jsonl(run / "validation-attempts.jsonl", {
                        "label": label, "status": "incomplete",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise
                append_jsonl(run / "validation-attempts.jsonl", {
                    "label": label, "status": "complete", "successes": measured["successes"],
                })
                return measured

            baseline = measure("bc")
            result["bc_validation_successes"] = baseline["successes"]
            best = -1
            last_validation_steps = None

            def finish_training_episode(reason):
                if env.active:
                    # DummyVecEnv auto-resets on done. Preserve that unused reset separately
                    # instead of pretending it was an executed successful/failed trajectory.
                    kind = "Unused vector auto-reset" if env.episode_actions == 0 else reason
                    env._finish_episode(complete=False, error=kind)
                model._last_obs = None

            def validate_rl(*, resume):
                nonlocal best, last_validation_steps
                steps = int(model.num_timesteps)
                result["online_steps"] = steps
                if model._n_updates < 1:
                    return False
                label = f"rl_{steps:06d}"
                save_checkpoint(model, checkpoints, label, metadata)
                check_time(deadline)
                finish_training_episode("Online action budget reached before episode termination")
                measured = measure(label)
                last_validation_steps = steps
                if measured["successes"] > best:
                    best = measured["successes"]
                    save_checkpoint(model, checkpoints, "best_rl", metadata)
                    result["best_validation_successes"] = best
                    result["selected_checkpoint"] = str(checkpoints / "best_rl.pt")
                    result["selection_reason"] = "best_complete_development_validation"
                write_json(run / "training-result.json", result)
                reached = best >= math.ceil(0.95 * len(validation_plan))
                if resume and not reached:
                    # The simulator has just run validation. Neither its terminal state nor
                    # SB3's pre-validation observation may become the next online transition.
                    model._last_obs = model.get_env().reset()
                    model._last_episode_starts = np.ones(model.n_envs, dtype=bool)
                return reached

            class ValidationTargetReached(Exception):
                pass

            class Recorder(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.next_validation = config["validation_interval"]
                    self.episode_ended = False

                def _on_step(self):
                    self.episode_ended = bool(np.all(self.locals["dones"]))
                    # Returning False here drops the just-executed transition in SB3.
                    # Stop only in _on_rollout_end, after SB3 has added it to replay.
                    return True

                def _on_rollout_end(self):
                    if self.num_timesteps % 100 == 0:
                        append_jsonl(run / "rl-metrics.jsonl", {
                            "online_steps": self.num_timesteps,
                            "rl_updates": int(model._n_updates),
                            "elapsed_seconds": time.monotonic() - started,
                            **model.last_train_metrics,
                        })
                    if self.num_timesteps % config["checkpoint_interval"] == 0:
                        save_checkpoint(
                            model, checkpoints, f"step_{self.num_timesteps:06d}", metadata,
                        )
                        emit("checkpoint", online_steps=self.num_timesteps,
                             **model.last_train_metrics)
                    check_time(deadline)
                    if self.episode_ended and self.num_timesteps >= self.next_validation:
                        if validate_rl(resume=True):
                            raise ValidationTargetReached()
                        interval = config["validation_interval"]
                        self.next_validation = (self.num_timesteps // interval + 1) * interval

            try:
                # One learn call retains each native episode until its real end. Serial
                # development validation is scheduled only at those episode boundaries.
                model.learn(total_timesteps=config["online_steps"], callback=Recorder(),
                            log_interval=10, progress_bar=False)
            except ValidationTargetReached:
                result["status"] = "validation_target_reached"
            else:
                result["status"] = "online_step_budget_exhausted"
                if last_validation_steps != model.num_timesteps and validate_rl(resume=False):
                    result["status"] = "validation_target_reached"
    except BudgetExpired as exc:
        result.update(status="time_budget_exhausted", error=str(exc))
    except BaseException as exc:
        original_error = exc
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        if model is not None:
            result["online_steps"] = model.num_timesteps
            result["rl_updates"] = model._n_updates
            try:
                save_checkpoint(model, checkpoints, "last", metadata)
                model.save_replay_buffer(checkpoints / "last-replay.pkl")
                result["last_checkpoint"] = str(checkpoints / "last.pt")
                result["last_checkpoint_kind"] = (
                    "rl" if model._n_updates > 0 else "behavior_cloning_only"
                )
                if model._n_updates > 0 and "selected_checkpoint" not in result:
                    result["selected_checkpoint"] = str(checkpoints / "last.pt")
                    result["selection_reason"] = "last_rl_without_complete_validation"
            except Exception as exc:
                cleanup_errors.append({"operation": "save_checkpoint",
                                       "error": f"{type(exc).__name__}: {exc}"})
            if hasattr(model, "_logger"):
                try:
                    model.logger.close()
                except Exception as exc:
                    cleanup_errors.append({"operation": "close_learning_log",
                                           "error": f"{type(exc).__name__}: {exc}"})
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                cleanup_errors.append({"operation": "close_environment",
                                       "error": f"{type(exc).__name__}: {exc}"})
        result["elapsed_seconds"] = time.monotonic() - started
        try:
            plot_training(run)
        except Exception as exc:
            cleanup_errors.append({"operation": "plot_training",
                                   "error": f"{type(exc).__name__}: {exc}"})
        if cleanup_errors:
            result["artifact_errors"] = cleanup_errors
        write_json(run / "training-result.json", result)
    if cleanup_errors and original_error is None:
        raise RuntimeError(f"Training artifacts or cleanup failed: {cleanup_errors}")
    return result


def evaluate(run, config, gpu, checkpoint, *, episodes=None, scene=None, split="evaluation",
             friction_scale=1.0):
    import torch

    from integrations.dynamics_rl.learning import load_policy

    torch.set_num_threads(config["learner_threads"])
    # Freeze a standalone artifact before choosing any evaluation layouts.
    frozen = run / "frozen-policy.pt"
    shutil.copyfile(checkpoint, frozen)
    policy = load_policy(frozen)
    if policy.metadata.get("contract") != contract():
        raise ValueError("Checkpoint observation/action contract differs from this environment")
    count = config["evaluation_episodes"] if episodes is None else episodes
    if type(count) is not int or count <= 0:
        raise ValueError("Inference episodes must be a positive integer")
    if not math.isfinite(friction_scale) or not 0.05 <= friction_scale <= 2.0:
        raise ValueError("Inference friction scale must be finite and in [0.05, 2.0]")
    if split not in {"evaluation", "rollout"}:
        raise ValueError("Unknown inference scene split")
    write_json(run / "inference-config.json", {
        "source_checkpoint": str(Path(checkpoint).resolve()),
        "friction_scale": friction_scale, "episodes": count, "scene_split": split,
        "online_learning": False, "teacher_takeover": False,
    })
    plan = (scene_plan(split, count, config["seed"], config["position_radius"])
            if scene is None else [{"index": 0, "layout_seed": None, "scene": scene}])
    write_json(run / "evaluation-scenes.json", plan)
    records = []
    for item in plan:
        client = simulator(run, f"evaluation_{item['index']:03d}", config, gpu,
                           "render", parameters={"friction_scale": friction_scale,
                                                 "mass_scale": 1.0, "restitution": 0.0})
        record = {"index": item["index"], "success": False, "complete": False}
        interruption = None
        try:
            client.start()
            details = run_episode(client, item["scene"], "episode", policy=policy)
            record.update({key: details[key] for key in
                           ("success", "complete", "actions", "elapsed_seconds", "videos")})
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", classification="unclassified")
        except (KeyboardInterrupt, SystemExit) as exc:
            interruption = exc
            record.update(error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                code = client.close()
            except Exception as exc:
                record.update(success=False, complete=False,
                              close_error=f"{type(exc).__name__}: {exc}",
                              classification="unclassified")
            except (KeyboardInterrupt, SystemExit) as exc:
                if interruption is None:
                    interruption = exc
                record.update(close_error=f"{type(exc).__name__}: {exc}")
            else:
                if client.failed or code != 0:
                    record.update(success=False, complete=False, close_returncode=code,
                                  classification="unclassified")
        if interruption is not None:
            record.update(success=False, complete=False, classification="interrupted")
        records.append(record)
        result = {"checkpoint": str(frozen), "attempted": len(records), "planned": len(plan),
                  "scene_split": split,
                  "friction_scale": friction_scale,
                  "policy_kind": "rl" if policy.rl_updates > 0 else "behavior_cloning_only",
                  "rl_updates": policy.rl_updates, "online_steps": policy.online_steps,
                  "successes": sum(row["success"] for row in records),
                  "incomplete": sum(not row["complete"] for row in records),
                  "records": records, "accepted": False}
        result["accepted"] = (interruption is None and split == "evaluation"
                              and scene is None and policy.rl_updates > 0
                              and len(records) == config["evaluation_episodes"]
                              and result["successes"] >= config["required_successes"])
        write_json(run / "evaluation-result.json", result)
        emit("evaluation", index=item["index"] + 1, total=len(plan),
             success=record["success"], successes=result["successes"])
        if interruption is not None:
            raise interruption
    return result


def plot_training(run):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, axis, xkey, ykeys in (
        ("bc.jsonl", axes[0], "updates", ["bc_loss"]),
        ("rl-metrics.jsonl", axes[1], "online_steps", ["actor_loss", "critic_loss", "bc_loss"]),
    ):
        file = run / name
        if file.exists():
            rows = [json.loads(line) for line in file.read_text().splitlines()]
            for key in ykeys:
                valid = [row for row in rows if row.get(key) is not None]
                if valid:
                    axis.plot([row[xkey] for row in valid], [row[key] for row in valid], label=key)
            if axis.lines:
                axis.legend()
        axis.set_xlabel(xkey)
    validation = run / "validation.jsonl"
    if validation.exists():
        rows = [json.loads(line) for line in validation.read_text().splitlines()]
        axes[2].plot(range(len(rows)), [r["successes"] / r["episodes"] for r in rows], "o-")
        axes[2].set_xticks(range(len(rows)), [r["label"] for r in rows], rotation=45)
    axes[2].set_ylabel("Validation success fraction")
    axes[2].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(run / "training-curves.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "evaluate", "rollout"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--friction-scale", type=float, default=1.0,
                        help="Fixed cube friction for evaluate/rollout only")
    parser.add_argument("--episodes", type=int,
                        help="Number of frozen-policy evaluate/rollout attempts")
    parser.add_argument("--preflight-run", type=Path,
                        help="Reuse recorded passing preflight evidence; no unrecorded skip")
    args = parser.parse_args(argv)
    if not math.isfinite(args.friction_scale) or not 0.05 <= args.friction_scale <= 2.0:
        parser.error("--friction-scale must be finite and in [0.05, 2.0]")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.command not in {"evaluate", "rollout"} and (
        args.friction_scale != 1.0 or args.episodes is not None
    ):
        parser.error("Friction overrides and episode counts apply only to evaluate/rollout")
    config = load_config(args.config)
    if args.command in {"evaluate", "rollout"} and args.checkpoint is None:
        parser.error("--checkpoint is required for neural evaluation")
    with run_lock(args.run_dir):
        run = new_run(args.run_dir, args.command, config)
        try:
            if args.command == "preflight":
                result = preflight(run, config, args.gpu)
            elif args.command == "train":
                if args.preflight_run:
                    evidence = validate_preflight(args.preflight_run, config)
                    write_json(run / "preflight-reference.json", {
                        "path": str(args.preflight_run.resolve()), "result": evidence,
                    })
                result = train(run, config, args.gpu, skip_preflight=args.preflight_run is not None)
            else:
                result = evaluate(run, config, args.gpu, args.checkpoint,
                                  episodes=(args.episodes if args.episodes is not None else
                                            1 if args.command == "rollout" else None),
                                  friction_scale=args.friction_scale,
                                  split="rollout" if args.command == "rollout" else "evaluation")
            emit("finished", result=result)
            return 0
        except BaseException as exc:
            write_json(run / "error.json", {"type": type(exc).__name__, "error": str(exc),
                                            "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
