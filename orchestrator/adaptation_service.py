"""Serial agent requests and asynchronous adaptation jobs with retained accounting."""

import json
import math
import select
import threading
import time
import traceback
from pathlib import Path

from orchestrator.adaptation_export import export_rollout, export_training
from orchestrator.adaptation_jobs import start_job
from orchestrator.dynamics_repair import nominal_parameters
from orchestrator.files import append_jsonl, write_json
from orchestrator.session_runner import stop_process


class AdaptationService:
    def __init__(self, branch_dir, config, layouts, deadline):
        self.branch = Path(branch_dir)
        self.config, self.layouts, self.deadline = config, layouts, deadline
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.submitted = False
        self.jobs, self.running, self.process = [], None, None
        self.used = dict.fromkeys(("rounds", "actions", "target_episodes", "diagnostic_episodes"), 0)
        baseline = json.loads((self.branch.parent / "baseline/target/result.json").read_text())
        self.networks = [{"id": "N000", "round": 0, "round_actions": 0,
                          "parameters": config["source_parameters"],
                          "target_dev_successes": baseline["successes"],
                          "target_dev_outcomes": self.outcomes(baseline)}]

    @staticmethod
    def outcomes(result):
        return [row["complete"] and row["success"]
                for row in sorted(result["episodes"], key=lambda row: row["layout"])]

    def reproduced(self, parameters):
        parent = self.networks[-1]
        outcomes = parent["target_dev_outcomes"]
        if outcomes is None or all(outcomes):
            return False
        parameters = {**nominal_parameters()["parameters"], **parameters}
        for job in self.jobs:
            if (job["kind"] != "probe" or job["status"] != "complete"
                    or job["request"]["network"] != parent["id"]
                    or job["spec"]["parameters"] != parameters):
                continue
            episodes = job["result"]["episodes"]
            if (all(row["complete"] and row["success"] == outcomes[row["layout"]] for row in episodes)
                    and any(not outcomes[row["layout"]] for row in episodes)):
                return True
        return False

    def network_dir(self, network):
        root = self.branch.parent if network == "N000" else self.branch
        return root / "networks" / network

    def status(self):
        limits = self.config["budgets"]
        return {"ok": True, "wall_seconds_remaining": max(0, self.deadline - time.time()),
                "budgets": {key: [value, limits["actions_total" if key == "actions" else key]]
                            for key, value in self.used.items()},
                "running": None if self.running is None else self.running["id"],
                "networks": self.networks, "jobs": [self.public_job(job) for job in self.jobs],
                "submitted": self.submitted}

    def public_job(self, job):
        return {key: job[key] for key in ("id", "kind", "status")}

    def parameters_valid(self, values):
        ranges = self.config["parameter_ranges"]
        return (type(values) is dict and not values.keys() - ranges.keys() and all(
            type(value) in (int, float) and math.isfinite(value)
            and ranges[key][0] <= value <= ranges[key][1] for key, value in values.items()))

    def diagnosis_valid(self, value):
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

    def validate(self, request):
        fields = {"status": (), "probe": ("network", "parameters", "layouts", "hypothesis", "prediction"),
                  "train": ("parameters", "actions", "rationale"), "wait": ("job", "seconds"),
                  "submit": ("diagnosis",)}
        if type(request) is not dict or type(request.get("op")) is not str or request["op"] not in fields:
            return "Unknown operation"
        op = request["op"]
        if request.keys() != {"op", *fields[op]}:
            return "Request fields differ from this operation"
        if op == "status":
            return None
        if op == "wait":
            if (type(request["seconds"]) is not int or not 1 <= request["seconds"] <= 900
                    or request["job"] not in [job["id"] for job in self.jobs]):
                return "Invalid job or wait interval"
            return None
        if self.submitted or time.time() >= self.deadline or self.closed.is_set():
            return "Branch has ended"
        if self.running is not None:
            return "A job is already running"
        if op == "submit":
            return None if self.diagnosis_valid(request["diagnosis"]) else "Invalid diagnosis"
        if not self.parameters_valid(request["parameters"]):
            return "Invalid parameters"
        for key in (("hypothesis", "prediction") if op == "probe" else ("rationale",)):
            if type(request[key]) is not str or not request[key].strip():
                return "An explanation is required"
        limits = self.config["budgets"]
        if op == "probe":
            layouts = request["layouts"]
            if (request["network"] not in [row["id"] for row in self.networks]
                    or type(layouts) is not list or not layouts
                    or any(type(i) is not int or not 0 <= i < len(self.layouts) for i in layouts)
                    or len(set(layouts)) != len(layouts)):
                return "Invalid network or layouts"
            if self.used["diagnostic_episodes"] + len(layouts) > limits["diagnostic_episodes"]:
                return "Diagnostic episode budget exhausted"
        else:
            actions = request["actions"]
            if (type(actions) is not int or not 2000 <= actions <= min(25000, limits["actions_per_round"],
                                                                    limits["actions_total"] - self.used["actions"])
                    or actions % self.config["training"]["n_steps"]):
                return "Invalid action budget"
            if (self.used["rounds"] >= limits["rounds"] or
                    self.used["target_episodes"] + len(self.layouts) > limits["target_episodes"]):
                return "Round or target episode budget exhausted"
            if not self.reproduced(request["parameters"]):
                return "Reproduce a target failure of the parent network at these parameters before training"
        return None

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

    def rollout_spec(self, network, parameters, layouts):
        return {**{key: self.config[key] for key in ("gpu", "simulator_python", "startup_timeout", "step_timeout")},
                "kind": "rollout", "network": str(self.network_dir(network) / "policy.pt"),
                "parameters": parameters, "layouts": layouts, "fresh_simulator_per_episode": False}

    def accept(self, request):
        parameters = {**nominal_parameters()["parameters"], **request["parameters"]}
        job = {"id": f"J{len(self.jobs) + 1:04d}", "kind": request["op"], "status": "running",
               "request": request, "submitted_at": time.time()}
        if job["kind"] == "probe":
            spec = self.rollout_spec(request["network"], parameters, [self.layouts[i] for i in request["layouts"]])
            job["phase"] = "episodes"
            self.used["diagnostic_episodes"] += len(request["layouts"])
        else:
            self.used["rounds"] += 1
            self.used["actions"] += request["actions"]
            job.update(phase="train", network=f"N{self.used['rounds']:03d}")
            # Never reuse a failed attempt's output directory.
            spec = {**{key: self.config[key] for key in ("gpu", "simulator_python", "startup_timeout", "step_timeout")},
                    "kind": "train", "parameters": parameters, "actions": request["actions"],
                    "round": self.used["rounds"], "seed": self.config["seed"], "training": self.config["training"],
                    "parent": str(self.network_dir(self.networks[-1]["id"]) / "training.zip"),
                    "output": str(self.network_dir(job["network"]))}
        job["spec"] = spec
        directory = self.branch / "jobs" / job["id"]
        write_json(directory / "job.json", job)
        write_json(self.branch / "workspace/observations/jobs" / job["id"] / "request.json", request)
        self.jobs.append(job)
        self.running = job
        self.launch(directory / job["phase"], spec)
        return {"ok": True, "job": job["id"]}

    def launch(self, directory, spec):
        try:
            self.process = start_job(directory, spec)
        except Exception as exc:
            self.finish("failed", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            raise

    def finish(self, status, detail):
        job = self.running
        job.update(status=status, finished_at=time.time(), result=detail)
        write_json(self.branch / "jobs" / job["id"] / "result.json", job)
        self.running, self.process = None, None

    def poll(self):
        """Advance completed subprocesses; the runner calls this independently of status."""
        with self.lock:
            if self.running is None or time.time() >= self.deadline or self.process.poll() is None:
                return
            job = self.running
            directory = self.branch / "jobs" / job["id"]
            public = self.branch / "workspace/observations/jobs" / job["id"]
            path = directory / job["phase"] / "result.json"
            if not path.exists():
                self.finish("failed", {"error": "Job exited without a result", "classification": "unclassified"})
                return
            result = json.loads(path.read_text())
            if job["phase"] != "train":
                if "episodes" in result:
                    export_rollout(result, public / job["phase"])
                if job["phase"] == "deploy" and "episodes" in result:
                    self.networks[-1]["target_dev_successes"] = result["successes"]
                    self.networks[-1]["target_dev_outcomes"] = self.outcomes(result)
                self.finish(result["status"], result)
            elif result["status"] == "complete":
                spec = job["spec"]
                export_training(directory / "train", public / "training")
                if time.time() >= self.deadline:
                    return
                self.networks.append({"id": job["network"], "round": spec["round"],
                                      "parameters": spec["parameters"], "round_actions": spec["actions"],
                                      "target_dev_successes": None, "target_dev_outcomes": None})
                job.update(phase="deploy", training=result)
                self.used["target_episodes"] += len(self.layouts)
                self.launch(directory / "deploy", self.rollout_spec(
                    job["network"], self.config["target_parameters"], self.layouts))
            else:
                self.finish("failed", result)

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
