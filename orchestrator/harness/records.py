"""Frozen versions, linear lineage, usage and host/public record destinations."""

from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
import shutil
from threading import RLock

import numpy as np

from orchestrator.files import append_jsonl, write_json
from .contracts import CapabilityUnavailable, PolicyBundle


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def workspace_path(workspace: Path, name, area: str, directory=False):
    """Request boundary: no traversal, symlink or special-file candidates."""
    if not isinstance(name, str) or not name:
        return None, "workspace path required"
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != area:
        return None, f"path must be under {area}/"
    path = workspace / relative
    if any((workspace / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        return None, "symlinks are not accepted"
    if directory:
        if not path.is_dir():
            return None, "candidate directory required"
        entries = list(path.rglob("*"))
        if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
            return None, "candidate must contain regular files and directories only"
        if sum(p.stat().st_size for p in entries if p.is_file()) > 64 * 1024 * 1024:
            return None, "candidate exceeds 64 MiB"
    elif not path.is_file():
        return None, "regular file required"
    elif area == "rewards" and path.stat().st_size > 64 * 1024:
        return None, "reward exceeds 64 KiB"
    return path, None


def reward_snapshot(host: Path, path: Path):
    content = path.read_bytes()
    version = sha256(content).hexdigest()
    snapshot = host / "rewards" / f"{version}.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(content)
    return version, snapshot


def usage() -> dict:
    return dict.fromkeys((
        "tasks_registered", "tasks_complete", "tasks_failed", "precheck_rejections",
        "planned_iterations", "actual_iterations", "training_rollouts_planned",
        "training_rollouts_completed", "training_rollouts_partial", "training_rollouts_unstarted",
        "transitions", "source_evaluation_episodes", "probe_episodes", "probe_transitions",
        "source_baseline_episodes", "source_baseline_transitions", "target_baseline_episodes",
        "target_judgment_episodes", "target_deploy_episodes", "target_episodes_planned",
        "target_episodes_cancelled", "builds", "claims", "elapsed_s", "gpu_s"), 0)


class Records:
    def __init__(self, run: Path, p0: PolicyBundle, source_ref):
        self.host = run / "host"
        self.workspace = run / "workspace"
        self.host.mkdir(parents=True)
        (self.workspace / "observations").mkdir(parents=True)
        self.lock = RLock()
        self.policies = {"P0000": {"bundle": p0, "round_id": None, "reward_version": None}}
        self.environments = {"E0000": {"id": "E0000", "ref": source_ref, "status": "ready",
                                        "candidate": None, "hypothesis": "initial source", "manifest": {}}}
        self.claims, self.rounds, self.jobs = {}, {}, {}
        self.head = "P0000"
        self.end_state = None
        self.recovered_policy = None
        self.submitted = False
        self.diagnosis = None
        self.batches = []
        self.counters = usage()
        self.save()

    def save(self):
        write_json(self.host / "records.json", json_value({
            "policies": self.policies, "environments": self.environments, "claims": self.claims,
            "rounds": self.rounds, "jobs": self.jobs, "head": self.head,
            "end_state": self.end_state, "recovered_policy": self.recovered_policy,
            "submitted": self.submitted, "diagnosis": self.diagnosis,
            "batches": self.batches, "usage": self.counters}))

    def archive(self, request, response):
        append_jsonl(self.host / "requests.jsonl", json_value({"request": request, "response": response}))

    def build(self, path, hypothesis, adapter):
        eid = f"E{len(self.environments):04d}"
        snapshot = self.host / "environments" / eid / "candidate"
        shutil.copytree(path, snapshot)
        result = adapter.prepare_source(snapshot)
        record = {"id": eid, "candidate": str(snapshot), "hypothesis": hypothesis}
        if isinstance(result, CapabilityUnavailable):
            record.update(status="capability_unavailable", gap=result.gap)
        else:
            record.update(status="ready", ref=result.ref, manifest=result.manifest)
        self.environments[eid] = record
        self.counters["builds"] += 1
        self.save()
        return {"environment_id": eid, **{k: v for k, v in record.items() if k in ("status", "manifest", "gap")}}

    def claim_error(self, request):
        pid, eid = request["policy_id"], request["environment_id"]
        if pid not in self.policies or eid not in self.environments:
            return "unknown policy or environment"
        sources, targets = request["source_jobs"], request["target_jobs"]
        if any(j not in self.jobs or self.jobs[j]["status"] != "complete" for j in sources + targets):
            return "claims require existing complete evidence jobs"
        if not any(self.jobs[j]["kind"] == "probe" and self.jobs[j]["policy_id"] == pid
                   and self.jobs[j]["environment_id"] == eid for j in sources):
            return "claim needs a complete probe of this policy and environment"
        if not any(self.jobs[j]["kind"] in ("baseline_target", "judgment", "deploy")
                   and self.jobs[j]["policy_id"] == pid for j in targets):
            return "claim needs a complete target batch of this policy"
        return None

    def round_error(self, eid, cid, rid):
        if eid not in self.environments or self.environments[eid]["status"] != "ready" or cid not in self.claims:
            return "ready environment and existing claim required"
        claim = self.claims[cid]
        if rid is None:
            if claim["environment_id"] != eid or claim["policy_id"] != self.head:
                return "new round claim must match the environment and current head"
        elif rid not in self.rounds:
            return "unknown round"
        else:
            round_record = self.rounds[rid]
            if round_record["environment_id"] != eid or round_record["claim_id"] != cid:
                return "continuation requires the round's environment and claim"
            successors = list(self.policies)[list(self.policies).index(round_record["entry_policy"])+1:]
            if any(self.policies[p]["round_id"] != rid for p in successors):
                return "round cannot continue after another round registered a policy"
        return None

    def register_job(self, kind, **fields):
        jid = f"J{len(self.jobs)+1:04d}"
        job = {"id": jid, "kind": kind, "status": "running", **fields}
        self.jobs[jid] = job
        self.save()  # launch intent is durable before the worker/backend starts
        return job

    def register_policy(self, job, result):
        pid = f"P{len(self.policies):04d}"
        self.policies[pid] = {"bundle": PolicyBundle(pid, result.checkpoint_ref, True),
                              "round_id": job["round_id"], "reward_version": job["reward_version"],
                              "job_id": job["id"]}
        self.head = pid
        job["registered_policy"] = pid
        return pid

    def account_train(self, job, result, complete):
        counters = job["usage"]
        counters.update(tasks_complete=int(complete), tasks_failed=int(not complete),
                        actual_iterations=result.iterations_completed,
                        training_rollouts_completed=result.rollouts_completed,
                        training_rollouts_partial=result.rollouts_partial,
                        training_rollouts_unstarted=counters["training_rollouts_planned"] -
                        result.rollouts_completed - result.rollouts_partial,
                        transitions=result.transitions, source_evaluation_episodes=result.evaluation_episodes,
                        elapsed_s=result.elapsed_s, gpu_s=result.gpu_s)
        for k, value in counters.items():
            if k not in ("tasks_registered", "planned_iterations", "training_rollouts_planned"):
                self.counters[k] += value
        self.counters["training_rollouts_unstarted"] -= counters["training_rollouts_planned"]

    def recover(self, pid):
        if self.end_state is None:
            self.end_state = "recovered"
            self.recovered_policy = pid
