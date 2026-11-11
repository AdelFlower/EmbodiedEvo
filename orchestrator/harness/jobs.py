"""Single-worker jobs and deterministic episodes through injected interfaces."""

from dataclasses import asdict
from threading import Event, Thread
import time

from orchestrator.files import write_json
from .contracts import EpisodeRecord
from .records import json_value


def public_job(job):
    fields = ("id", "kind", "status", "policy_id", "environment_id", "round_id", "task_number",
              "registered_policy", "judgment_id", "artifacts", "curve_ref", "evaluation_status", "usage")
    return json_value({k: job[k] for k in fields if k in job})


class Jobs:
    def __init__(self, experiment):
        self.experiment = experiment
        self.records = experiment.records
        self.stop_event = Event()
        self.thread = None
        self.handle = None

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def launch(self, job):
        self.stop_event.clear()
        self.thread = Thread(target=self.run, args=(job,), name="harness-job")
        self.thread.start()

    def wait(self, seconds):
        if self.thread is not None:
            self.thread.join(seconds)

    def cancel(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def episode(self, policy, ref, layout_id, layout):
        runtime = self.experiment.task_adapter.create_runtime(ref, num_envs=1, render=True)
        batch = runtime.reset([layout])
        public, truth, scoring = [], [], []
        steps = 0
        while True:
            public.append(json_value(batch.public))
            truth.append(json_value(batch.source_truth))
            scoring.append(json_value(batch.host_scoring[0]))
            if batch.execution_error[0] is not None:
                status = "execution_error"
                break
            if bool(batch.done[0]):
                status = "complete"
                break
            if self.stop_event.is_set():
                status = "cancelled"
                break
            batch = runtime.step(policy.act(batch.actor, deterministic=True))
            steps += 1
        outcome = batch.outcome[0] if status == "complete" else {"strict": False, "lenient": False}
        runtime.close()
        return EpisodeRecord(layout_id, steps, public, truth, outcome, scoring, status)

    def batch(self, job, ref, layouts, source):
        """Every episode gets a new runtime; only explicit public fields leave host/."""
        records = self.records
        policy = self.experiment.policy_adapter.load(records.policies[job["policy_id"]]["bundle"])
        episodes = []
        started = time.monotonic()
        with records.lock:
            if not source:
                records.counters["target_episodes_planned"] += len(layouts)
            records.save()
        for layout_id, layout in layouts:
            if self.stop_event.is_set():
                break
            episode = self.episode(policy, ref, layout_id, layout)
            episodes.append(episode)
            host_path = records.host / "jobs" / job["id"] / f"episode_{layout_id}.json"
            write_json(host_path, json_value(episode))
            public = {"layout_id": layout_id, "steps": episode.steps, "public": episode.public,
                      "outcome": {k: bool(episode.outcome[k]) for k in ("strict", "lenient")},
                      "status": episode.status}
            if source:
                public["source_truth"] = episode.source_truth
            destination = records.workspace / "observations" / job["id"] / f"episode_{layout_id}.json"
            write_json(destination, json_value(public))
        complete = len(episodes) == len(layouts) and all(e.status != "cancelled" for e in episodes)
        with records.lock:
            job.update(status="complete" if complete else "interrupted", episodes=json_value(episodes))
            job["artifacts"] = f"observations/{job['id']}"
            count = sum(e.status != "cancelled" for e in episodes)
            if source:
                prefix = "probe" if job["kind"] == "probe" else "source_baseline"
                records.counters[f"{prefix}_episodes"] += count
                records.counters[f"{prefix}_transitions"] += sum(e.steps for e in episodes)
            else:
                kind = "baseline" if job["kind"] == "baseline_target" else job["kind"]
                key = f"target_{kind}_episodes"
                records.counters[key] += count
                records.counters["target_episodes_cancelled"] += len(layouts) - count
                job["usage"] = {key: count, "target_episodes_planned": len(layouts),
                                "target_episodes_cancelled": len(layouts) - count}
                batch = {"job_id": job["id"], "policy_id": job["policy_id"], "complete": complete,
                         "strict": sum(e.outcome["strict"] for e in episodes),
                         "lenient": sum(e.outcome["lenient"] for e in episodes)}
                records.batches.append(batch)
                if complete and batch["strict"] == 4:
                    records.recover(job["policy_id"])
            job["elapsed_s"] = time.monotonic() - started
            records.counters["elapsed_s"] += job["elapsed_s"]
            records.save()
        return complete

    def run(self, job):
        if job["kind"] == "train":
            self.train(job)
        else:
            source = job["kind"] == "probe"
            ref = (self.records.environments[job["environment_id"]]["ref"] if source
                   else self.experiment.target_ref)
            layouts = [(i, self.experiment.layouts["judgment"][i]) for i in job["layouts"]]
            self.batch(job, ref, layouts, source)

    def train(self, job):
        records = self.records
        directory = records.host / "jobs" / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "launch.json", json_value(job))
        handle = self.experiment.backend.start(job["spec"], directory)
        self.handle = handle
        stopped = False
        if self.stop_event.is_set():
            handle.stop()
            stopped = True
        result = handle.poll()
        while result is None:
            if self.stop_event.is_set() and not stopped:
                handle.stop()
                stopped = True
            time.sleep(0.01)
            result = handle.poll()
        self.handle = None
        complete = (result.iterations_completed == 100 and result.checkpoint_ref is not None
                    and result.evaluation_status != "failed_mid_task"
                    and (result.status == "complete" or result.evaluation_status == "final_incomplete"))
        with records.lock:
            job.update(result=asdict(result), curve_ref=result.curve_ref,
                       evaluation_status=result.evaluation_status, status="running" if complete else "failed")
            records.account_train(job, result, complete)
            if complete:
                pid = records.register_policy(job, result)
                judgment = records.register_job("judgment", policy_id=pid, layouts=list(range(4)), parent=job["id"])
                job["judgment_id"] = judgment["id"]
            records.save()
        if complete:
            self.batch(judgment, self.experiment.target_ref,
                       list(enumerate(self.experiment.layouts["judgment"])), source=False)
            with records.lock:
                job["usage"].update(judgment["usage"])
                job["status"] = "complete"
                records.save()
        with records.lock:
            # Curves and traces are source-only backend references, never interpreted here.
            destination = records.workspace / "observations" / job["id"] / "training.json"
            write_json(destination, {**public_job(job), "trace_ref": result.trace_ref})
