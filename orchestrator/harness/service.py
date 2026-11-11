"""Request intake, archived deduplication, round authorization and stopping."""

from copy import deepcopy
import json

from orchestrator.files import append_jsonl
from .config import exact, text
from .contracts import EndState, TrainSpec
from .diagnosis import diagnosis_error, string_list
from .jobs import public_job
from .records import reward_snapshot, usage, workspace_path


FIELDS = {
    "status": "", "build": "request_id candidate hypothesis",
    "probe": "request_id policy_id environment_id prediction layouts",
    "deploy": "request_id policy_id purpose",
    "reproduce": "request_id policy_id environment_id source_jobs target_jobs comparison uncertainty",
    "train": "request_id environment_id claim_id reward rationale environment_round_id",
    "wait": "job_id seconds", "submit": "request_id diagnosis"}


def request_error(request):
    if not isinstance(request, dict) or not isinstance(request.get("op"), str) or request["op"] not in FIELDS:
        return "unknown operation"
    op = request["op"]
    if not exact(request, "op " + FIELDS[op]):
        return "request requires the exact operation fields"
    for key, value in request.items():
        if key in ("layouts", "source_jobs", "target_jobs", "seconds", "environment_round_id"):
            continue
        if not isinstance(value, str) or (key == "request_id" and not text(value)):
            return f"{key} must be a string (request_id must be nonempty)"
    if op == "probe":
        layouts = request["layouts"]
        if (not isinstance(layouts, list) or not layouts or any(type(v) is not int or v not in range(4) for v in layouts)
                or len(set(layouts)) != len(layouts)):
            return "layouts must be a nonempty unique subset of 0–3"
    if op == "reproduce" and not all(string_list(request[k], nonempty=True) for k in ("source_jobs", "target_jobs")):
        return "source_jobs and target_jobs must be nonempty string lists"
    if op == "train" and request["environment_round_id"] is not None and not text(request["environment_round_id"]):
        return "environment_round_id must be a string or null"
    if op == "wait" and (type(request["seconds"]) is not int or request["seconds"] not in range(1, 901)):
        return "seconds must be an integer from 1 to 900"
    return None


class HarnessService:
    def __init__(self, experiment):
        self.experiment = experiment
        self.records = experiment.records
        self.jobs = experiment.jobs
        self.responses = {}

    def handle(self, request: dict) -> dict:
        with self.records.lock:
            rid = request.get("request_id") if isinstance(request, dict) else None
            encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
            if isinstance(rid, str) and rid in self.responses:
                original, response = self.responses[rid]
                result = deepcopy(response) if encoded == original else {"ok": False, "error": "request_id content conflict"}
                self.records.archive(request, result)
                return result
            error = request_error(request)
            if error is None:
                error = self.state_error(request["op"])
            if error is not None:
                response = {"ok": False, "error": error}
            elif request["op"] == "wait":
                if request["job_id"] not in self.records.jobs:
                    response = {"ok": False, "error": "unknown job"}
                else:
                    response = None
            else:
                response = self.dispatch(request)
            if response is not None:
                return self.archive(request, response, rid, encoded)
        # Joining outside the records lock lets the worker finish and persist its result.
        if self.records.jobs[request["job_id"]]["status"] == "running":
            self.jobs.wait(request["seconds"])
        with self.records.lock:
            response = {"ok": True, "job": public_job(self.records.jobs[request["job_id"]])}
            return self.archive(request, response, rid, encoded)

    def archive(self, request, response, rid, encoded):
        self.records.archive(request, response)
        if isinstance(rid, str) and rid:
            self.responses[rid] = (encoded, deepcopy(response))
        return response

    def state_error(self, op):
        state = self.records.end_state
        if state == "recovered" and op not in ("status", "wait", "submit"):
            return "adaptation ended: recovered"
        if state is not None and state != "recovered" and op != "status":
            return f"experiment ended: {state}"
        if self.jobs.busy() and op in ("probe", "deploy", "train", "submit"):
            return "an experiment job is running"
        return None

    def dispatch(self, request):
        op = request["op"]
        if op == "status":
            handle = self.jobs.handle
            return {"ok": True, "phase": self.records.end_state or "adapting", "head": self.records.head,
                    "policies": list(self.records.policies),
                    "environments": [{"id": eid, "status": e["status"]} for eid, e in self.records.environments.items()],
                    "claims": {cid: {k: c[k] for k in ("policy_id", "environment_id")} for cid, c in self.records.claims.items()},
                    "rounds": self.records.rounds,
                    "jobs": [public_job(j) for j in self.records.jobs.values()],
                    "progress": handle.progress() if handle is not None else None,
                    "usage": dict(self.records.counters)}
        if op == "build":
            path, error = workspace_path(self.records.workspace, request["candidate"], "environments", directory=True)
            if error:
                return {"ok": False, "error": error}
            return {"ok": True, **self.records.build(path, request["hypothesis"], self.experiment.task_adapter)}
        if op == "reproduce":
            error = self.records.claim_error(request)
            if error:
                return {"ok": False, "error": error}
            cid = f"C{len(self.records.claims)+1:04d}"
            self.records.claims[cid] = {k: deepcopy(v) for k, v in request.items() if k not in ("op", "request_id")}
            self.records.counters["claims"] += 1
            self.records.save()
            return {"ok": True, "claim_id": cid}
        if op in ("probe", "deploy"):
            if request["policy_id"] not in self.records.policies:
                return {"ok": False, "error": "unknown policy"}
            if op == "probe" and (request["environment_id"] not in self.records.environments or
                                  self.records.environments[request["environment_id"]]["status"] != "ready"):
                return {"ok": False, "error": "probe requires a ready environment"}
            fields = {k: v for k, v in request.items() if k not in ("op", "request_id")}
            if op == "deploy":
                fields["layouts"] = list(range(4))
            job = self.records.register_job(op, **fields)
            self.jobs.launch(job)
            return {"ok": True, "job_id": job["id"]}
        if op == "train":
            return self.train(request)
        path, error = workspace_path(self.records.workspace, request["diagnosis"], "submission")
        if error:
            return {"ok": False, "error": error}
        diagnosis = json.loads(path.read_text())
        error = diagnosis_error(diagnosis, self.records)
        if error:
            return {"ok": False, "error": error}
        self.records.diagnosis = diagnosis
        self.records.submitted = True
        if self.records.end_state is None:
            self.records.end_state = "submitted_not_recovered"
        self.records.save()
        return {"ok": True, "end_state": self.records.end_state}

    def train(self, request):
        eid, cid, rid = request["environment_id"], request["claim_id"], request["environment_round_id"]
        error = self.records.round_error(eid, cid, rid)
        if error:
            return {"ok": False, "error": error}
        path, error = workspace_path(self.records.workspace, request["reward"], "rewards")
        if error:
            return {"ok": False, "error": error}
        version, snapshot = reward_snapshot(self.records.host, path)
        claim = self.records.claims[cid]
        completed = [j for j in self.records.jobs.values() if j["kind"] == "train" and j["status"] == "complete"
                     and j["round_id"] == rid] if rid is not None else []
        traces = ([completed[-1]["result"]["trace_ref"]] if completed else
                  [str(self.records.host / "jobs" / j) for j in claim["source_jobs"]
                   if self.records.jobs[j]["kind"] == "probe"])
        precheck = self.experiment.reward_runner.precheck(snapshot, traces)
        append_jsonl(self.records.host / "prechecks.jsonl", {"request_id": request["request_id"],
                     "reward_version": version, "result": precheck})
        if not precheck["ok"]:
            self.records.counters["precheck_rejections"] += 1
            self.records.save()
            return {"ok": False, "error": "reward precheck rejected"}
        if rid is None:
            rid = f"R{len(self.records.rounds)+1:04d}"
            self.records.rounds[rid] = {"environment_id": eid, "entry_policy": self.records.head, "claim_id": cid}
        head = self.records.policies[self.records.head]
        training = self.experiment.config.training
        warmup = training["warmup_iterations"] if self.records.head == "P0000" or head["reward_version"] != version else 0
        task_number = self.records.counters["tasks_registered"] + 1
        spec = TrainSpec(self.experiment.seed, task_number, rid, self.records.environments[eid]["ref"],
                         head["bundle"], snapshot, training["iterations_per_job"], warmup,
                         training["num_envs"], training["minibatch"],
                         {**training["source_evaluation"], "layouts": self.experiment.layouts["source_evaluation"]},
                         self.experiment.layouts["training"], training["profile"], training["device"],
                         training["episodes_per_env"])
        counters = usage()
        counters.update(tasks_registered=1, planned_iterations=100,
                        training_rollouts_planned=100 * spec.num_envs, training_rollouts_unstarted=100 * spec.num_envs)
        for k, value in counters.items():
            self.records.counters[k] += value
        job = self.records.register_job("train", environment_id=eid, policy_id=self.records.head, round_id=rid,
                                        claim_id=cid, reward_version=version, spec=spec, usage=counters,
                                        task_number=task_number, rationale=request["rationale"])
        self.jobs.launch(job)
        return {"ok": True, "job_id": job["id"], "environment_round_id": rid}

    def end(self, state):
        state = EndState(state)
        if state not in (EndState.AGENT_EXITED, EndState.HOST_CANCELLED):
            raise ValueError("end accepts agent_exited or host_cancelled")
        with self.records.lock:
            if self.records.end_state is None:
                self.records.end_state = state.value
                self.records.save()
        self.jobs.cancel()
        with self.records.lock:
            self.records.save()
