"""Serialize agent requests, reproduction gates and budgeted friction jobs."""

import json
import math
import select
from pathlib import Path
import threading
import time

import numpy as np

from integrations.friction_rl.reward_sandbox import export_rollout, export_training, run_sandbox
from orchestrator.files import append_jsonl, write_json
from orchestrator.runtime import stop_process
from orchestrator.experiment_jobs import development_episodes, needs_warmup, rollout_spec, start_job


class ExperimentService:
    """Serve the isolated agent and retain every experiment attempt."""

    def __init__(self, branch_dir, config, deadline):
        self.branch, self.config, self.deadline = Path(branch_dir), config, deadline
        self.lock, self.closed = threading.RLock(), threading.Event()
        self.submitted = False
        self.jobs, self.running, self.process = [], None, None
        self.used = dict.fromkeys(("rounds", "iterations", "target_episodes", "diagnostic_episodes"), 0)
        self.dry_runs = 0
        baseline = json.loads((self.branch.parent / "baseline/target/result.json").read_text())
        self.networks = [{"id": "N000", "round": 0, "parent": None, "parameters": config["source_parameters"],
                          "reward": "default", "iterations": 0, **self.outcomes(baseline)}]

    @staticmethod
    def outcomes(result):
        """Collect strict and lenient development outcomes in episode order."""
        rows = sorted(result["episodes"], key=lambda row: int(row["name"][1:]))
        return {f"{kind}_outcomes": [row["complete"] and row[key] for row in rows]
                for kind, key in (("strict", "strict_success"), ("lenient", "success_lenient"))}

    def status(self):
        """Expose budgets, public network metadata and job states."""
        return {"ok": True, "wall_seconds_remaining": max(0, self.deadline - time.time()),
                "budgets": {key: [used, self.config["budgets"]["iterations_total" if key == "iterations" else key]]
                            for key, used in self.used.items()},
                "running": self.running["id"] if self.running else None,
                "networks": self.networks, "jobs": [self.public_job(job) for job in self.jobs],
                "submitted": self.submitted}

    def reproduced(self, parameters):
        """Require a matching parent probe containing at least one strict target failure."""
        parent = self.networks[-1]
        outcomes = parent["strict_outcomes"]
        if outcomes is None or all(outcomes):
            return False
        parameters = {**self.config["source_parameters"], **parameters}
        for job in self.jobs:
            if (job["kind"] == "probe" and job["status"] == "complete"
                    and job["request"]["network"] == parent["id"] and job["spec"]["parameters"] == parameters):
                rows = job["result"]["episodes"]
                if (all(row["complete"] and row["strict_success"] == outcomes[int(row["name"][1:])] for row in rows)
                        and any(not outcomes[int(row["name"][1:])] for row in rows)):
                    return True
        return False

    def diagnosis_valid(self, value):
        """Validate the adaptation diagnosis schema with a reward rationale."""
        keys = {"estimated_target_parameters", "conclusion", "evidence_jobs", "competing_hypotheses",
                "limitations", "preferred_network", "reward_rationale"}
        return (type(value) is dict and value.keys() == keys and self._diagnosis_fields_valid(value)
                and all(value[key].strip() for key in ("conclusion", "limitations", "preferred_network"))
                and type(value["reward_rationale"]) is str and bool(value["reward_rationale"].strip())
                and all(job in [row["id"] for row in self.jobs] for job in value["evidence_jobs"])
                and all(row.keys() == {"hypothesis", "test_jobs", "outcome", "reason"}
                        and row["hypothesis"].strip() and row["reason"].strip()
                        and all(job in [item["id"] for item in self.jobs] for job in row["test_jobs"])
                        for row in value["competing_hypotheses"]))

    def reward_path(self, requested):
        """Resolve an agent-supplied program only within its workspace."""
        if type(requested) is not str or not requested:
            return None
        path = Path("scripts/default_reward.py" if requested == "default" else requested)
        workspace = (self.branch / "workspace").resolve()
        if path.is_absolute() or ".." in path.parts:
            return None
        path = (workspace / path).resolve()
        if not path.is_relative_to(workspace) or not path.is_file() or path.stat().st_size > 65536:
            return None
        return path

    def validate(self, request):
        """Refuse invalid requests before reserving any experiment budget."""
        fields = {"status": (), "probe": ("network", "parameters", "episodes", "hypothesis", "prediction"),
                  "train": ("parameters", "iterations", "reward", "rationale"),
                  "wait": ("job", "seconds"), "submit": ("diagnosis",)}
        if type(request) is not dict or type(request.get("op")) is not str or request["op"] not in fields:
            return "Unknown operation"
        op = request["op"]
        if request.keys() != {"op", *fields[op]}:
            return "Request fields differ from this operation"
        if self.submitted or self.closed.is_set() or time.time() >= self.deadline:
            return "Branch has ended"
        if op == "status":
            return None
        if op == "wait":
            if (type(request["seconds"]) is not int or not 1 <= request["seconds"] <= 900
                    or request["job"] not in [job["id"] for job in self.jobs]):
                return "Invalid job or wait interval"
            return None
        if self.running is not None:
            return "A job is already running"
        if op == "submit":
            return None if self.diagnosis_valid(request["diagnosis"]) else "Invalid diagnosis"
        if not self.parameters_valid(request["parameters"]):
            return "Invalid parameters"
        if any(type(request[key]) is not str or not request[key].strip()
               for key in (("hypothesis", "prediction") if op == "probe" else ("rationale",))):
            return "An explanation is required"
        limits = self.config["budgets"]
        if op == "probe":
            episodes = request["episodes"]
            if (request["network"] not in [row["id"] for row in self.networks]
                    or type(episodes) is not list or not episodes
                    or any(type(k) is not int or not 0 <= k < 4 for k in episodes)
                    or len(set(episodes)) != len(episodes)):
                return "Invalid network or episodes"
            if self.used["diagnostic_episodes"] + len(episodes) > limits["diagnostic_episodes"]:
                return "Diagnostic episode budget exhausted"
        else:
            if (type(request["iterations"]) is not int or not 4 <= request["iterations"] <= min(
                    limits["iterations_per_round"], limits["iterations_total"] - self.used["iterations"])):
                return "Invalid iterations budget"
            if (self.used["rounds"] >= limits["rounds"]
                    or self.used["target_episodes"] + 4 > limits["target_episodes"]):
                return "Round or target episode budget exhausted"
            if not self.reproduced(request["parameters"]):
                return "Reproduce a target failure of the parent network at these parameters before training"
            if self.reward_path(request["reward"]) is None:
                return "Reward must be a workspace-relative regular file of at most 64 KiB"
        return None

    def accept(self, request):
        """Snapshot validated rewards and charge each job when it starts."""
        parameters = {**self.config["source_parameters"], **request["parameters"]}
        sums = None
        if request["op"] == "train":
            reward_text = self.reward_path(request["reward"]).read_text()
            traces = {}
            for k in range(4):
                with np.load(self.branch / f"workspace/observations/baseline/source/D{k}/trajectory.npz") as trace:
                    traces[f"D{k}"] = dict(trace)
            self.dry_runs += 1
            dry = run_sandbox(reward_text, traces, self.branch / f"reward_checks/R{self.dry_runs:04d}",
                              self.config["sandbox"])
            if not dry["ok"]:
                return {"ok": False, "error": dry["error"]}
            if time.time() >= self.deadline:
                return {"ok": False, "error": "Branch has ended"}
            sums = {name: float(value.sum()) for name, value in dry["rewards"].items()}
        job = {"id": f"J{len(self.jobs) + 1:04d}", "kind": request["op"], "status": "running",
               "request": request, "submitted_at": time.time()}
        directory = self.branch / "jobs" / job["id"]
        if request["op"] == "probe":
            job["phase"] = "episodes"
            spec = rollout_spec(self.config, self.network_dir(request["network"]) / "policy.pt", parameters,
                                development_episodes(self.config["seed"], request["episodes"]))
            self.used["diagnostic_episodes"] += len(request["episodes"])
        else:
            self.used["rounds"] += 1
            self.used["iterations"] += request["iterations"]
            parent = self.networks[-1]["id"]
            parent_dir = self.network_dir(parent)
            job.update(phase="train", network=f"N{self.used['rounds']:03d}")
            output = self.network_dir(job["network"])
            output.mkdir(parents=True)
            (output / "reward.py").write_text(reward_text)
            previous = None if parent == "N000" else (parent_dir / "reward.py").read_text()
            spec = {"kind": "train", "config": self.config, "parameters": parameters,
                    "iterations": request["iterations"], "round": self.used["rounds"], "parent_id": parent,
                    "parent": str(parent_dir / ("policy.pt" if parent == "N000" else "training.pt")),
                    "reward": str(output / "reward.py"), "warmup": needs_warmup(parent, reward_text, previous),
                    "demos": str(self.branch.parent / "prior/demos.npz"), "output": str(output)}
        job["spec"] = spec
        write_json(directory / "job.json", job)
        write_json(self.branch / "workspace/observations/jobs" / job["id"] / "request.json", request)
        self.jobs.append(job)
        self.running = job
        self.launch(directory / job["phase"], spec)
        return {"ok": True, "job": job["id"], **({"reward_sums": sums} if sums is not None else {})}

    def launch(self, directory, spec):
        """Launch one retained phase without retrying it."""
        self.process = start_job(directory, spec)

    def finish(self, status, detail):
        """Retain the host result and publish the reward runner's error when present."""
        if status != "complete":
            write_json(self.branch / "workspace/observations/jobs" / self.running["id"] / "error.json",
                       {"status": status, "error": detail.get("error", status)})
        job = self.running
        job.update(status=status, finished_at=time.time(), result=detail)
        write_json(self.branch / "jobs" / job["id"] / "result.json", job)
        self.running, self.process = None, None

    def poll(self):
        """Advance training to deployment and register the resulting development outcomes."""
        with self.lock:
            if time.time() >= self.deadline:
                self.stop("stopped_by_deadline")
                return
            if self.running is None or self.process.poll() is None:
                return
            job = self.running
            directory = self.branch / "jobs" / job["id"]
            public = self.branch / "workspace/observations/jobs" / job["id"]
            path = directory / job["phase"] / "result.json"
            if not path.is_file():
                self.finish("failed", {"error": "Job exited without a result", "classification": "unclassified"})
                return
            result = json.loads(path.read_text())
            if job["phase"] == "train" and result["status"] == "complete":
                spec = job["spec"]
                export_training(spec["output"], public / "training", result)
                if time.time() >= self.deadline:
                    self.stop("stopped_by_deadline")
                    return
                self.networks.append({"id": job["network"], "round": spec["round"], "parent": spec["parent_id"],
                                      "parameters": spec["parameters"], "reward": job["request"]["reward"],
                                      "iterations": spec["iterations"], "strict_outcomes": None,
                                      "lenient_outcomes": None})
                job.update(phase="deploy", training=result)
                self.used["target_episodes"] += 4
                self.launch(directory / "deploy", rollout_spec(
                    self.config, self.network_dir(job["network"]) / "policy.pt", self.config["target_parameters"],
                    development_episodes(self.config["seed"])))
            else:
                if job["phase"] != "train" and "episodes" in result:
                    export_rollout(result, public / job["phase"])
                    if job["phase"] == "deploy":
                        self.networks[-1].update(self.outcomes(result))
                self.finish(result["status"], result)

    def network_dir(self, network):
        root = self.branch.parent if network == "N000" else self.branch
        return root / "networks" / network

    def public_job(self, job):
        return {key: job[key] for key in ("id", "kind", "status")}

    def parameters_valid(self, values):
        ranges = self.config["parameter_ranges"]
        return (type(values) is dict and not values.keys() - ranges.keys() and all(
            type(value) in (int, float) and math.isfinite(value)
            and ranges[key][0] <= value <= ranges[key][1] for key, value in values.items()))

    def request(self, request):
        with self.lock:
            error = self.validate(request)
            if error:
                response = {"ok": False, "error": error}
            elif request["op"] == "status":
                response = self.status()
            elif request["op"] == "submit":
                write_json(self.branch / "diagnosis.json", request["diagnosis"])
                self.submitted = True
                response = {"ok": True, "submitted": True}
            elif request["op"] == "wait":
                response = None
            else:
                response = self.accept(request)
        if response is None:
            until = min(time.time() + request["seconds"], self.deadline)
            job = next(job for job in self.jobs if job["id"] == request["job"])
            while job["status"] == "running" and time.time() < until and not self.closed.is_set():
                self.closed.wait(min(0.1, max(0, until - time.time())))
            with self.lock:
                response = {"ok": True, "job": self.public_job(job)}
        append_jsonl(self.branch / "requests.jsonl", {"time": time.time(), "request": request, "response": response})
        return response

    def stop(self, reason):
        with self.lock:
            self.closed.set()
            if self.running is not None:
                stop_process(self.process)
                self.networks[:] = [row for row in self.networks if row["id"] != self.running.get("network")]
                self.finish(reason, {"classification": "unclassified"})

    def serve(self, listener):
        """Serve one newline-framed connection at a time until the runner closes."""
        while not self.closed.is_set():
            if not select.select([listener], [], [], 0.1)[0]:
                continue
            connection, _ = listener.accept()
            with connection, connection.makefile("rwb") as stream:
                data = b""
                while b"\n" not in data and len(data) <= 16 * 1024 * 1024:
                    if self.closed.is_set():
                        return
                    if not select.select([connection], [], [], 0.1)[0]:
                        continue
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                try:
                    request = json.loads(data) if data.endswith(b"\n") else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request = {"invalid_json": repr(data)}
                response = self.request(request)
                stream.write(json.dumps(response).encode() + b"\n")
                stream.flush()

    def _diagnosis_fields_valid(self, value):
        strings = lambda items: type(items) is list and all(type(item) is str for item in items)
        return (type(value) is dict and
                all(type(value.get(key)) is str for key in ("conclusion", "limitations", "preferred_network"))
                and value["preferred_network"] in [row["id"] for row in self.networks]
                and self.parameters_valid(value.get("estimated_target_parameters"))
                and value["estimated_target_parameters"].keys() == self.config["parameter_ranges"].keys()
                and strings(value.get("evidence_jobs"))
                and type(value.get("competing_hypotheses")) is list and all(
                    type(row) is dict and all(type(row.get(key)) is str for key in
                                             ("hypothesis", "outcome", "reason"))
                    and row["outcome"] in ("rejected", "not_rejected", "untested")
                    and strings(row.get("test_jobs")) for row in value["competing_hypotheses"]))
