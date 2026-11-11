"""Host lifecycle: frozen inputs, baselines, selection and private final evaluation."""

from pathlib import Path
import re

from orchestrator.files import write_json
from .contracts import PolicyBundle
from .jobs import Jobs
from .records import Records, json_value


class Experiment:
    def __init__(self, run_dir: Path, config, task_adapter, policy_adapter, backend,
                 reward_runner, initial_policy: PolicyBundle, source_ref, target_ref, seed: int):
        run_dir = Path(run_dir)
        if re.match(r"^\d{12}", run_dir.name) is None:
            raise ValueError("run_dir requires a YYYYMMDDHHMM prefix")
        if type(seed) is not int or initial_policy.id != "P0000":
            raise ValueError("integer experiment seed and P0000 initial bundle required")
        self.config = config
        self.task_adapter = task_adapter
        self.policy_adapter = policy_adapter
        self.backend = backend
        self.reward_runner = reward_runner
        self.target_ref = target_ref
        self.seed = seed
        self.records = Records(run_dir, initial_policy, source_ref)
        self.layouts = {}
        self.jobs = Jobs(self)

    def freeze(self):
        streams = self.config.values["layouts"]
        self.layouts = {
            "judgment": self.task_adapter.make_layouts(streams["judgment_stream"], self.seed, 4),
            "final": self.task_adapter.make_layouts(streams["final_stream"], self.seed, 30),
            "source_evaluation": self.task_adapter.make_layouts(streams["source_eval_stream"], self.seed, 512),
            "training": {"stream": streams["training_stream"], "seed": self.seed}}
        write_json(self.records.host / "frozen.json", json_value({
            "config": self.config.values, "adapter": self.task_adapter.describe(), "layouts": self.layouts,
            "experiment_seed": self.seed, "initial_policy": self.records.policies["P0000"]["bundle"],
            "source_ref": self.records.environments["E0000"]["ref"], "target_ref": self.target_ref}))

    def baselines(self):
        for kind, ref, source in (("baseline_source", self.records.environments["E0000"]["ref"], True),
                                  ("baseline_target", self.target_ref, False)):
            job = self.records.register_job(kind, policy_id="P0000", environment_id="E0000" if source else None,
                                            layouts=list(range(4)))
            self.jobs.batch(job, ref, list(enumerate(self.layouts["judgment"])), source)

    def select_policy(self):
        if self.records.end_state is None or self.jobs.busy():
            raise ValueError("selection requires an ended experiment and no running job")
        if self.records.recovered_policy is not None:
            selected = self.records.recovered_policy
        else:
            first = {}
            for batch in self.records.batches:
                if batch["complete"] and batch["policy_id"] not in first:
                    first[batch["policy_id"]] = batch
            selected = max(first, key=lambda p: (first[p]["strict"], first[p]["lenient"], -int(p[1:])))
        write_json(self.records.host / "selection.json", {"selected_policy": selected})
        return selected

    def final_evaluation(self):
        selected = self.select_policy()
        results = {}
        for pid in dict.fromkeys(("P0000", selected)):
            policy = self.policy_adapter.load(self.records.policies[pid]["bundle"])
            episodes = []
            # Cancellation of adaptation must not cancel the separate host final evaluation.
            self.jobs.stop_event.clear()
            for index, layout in enumerate(self.layouts["final"]):
                episode = self.jobs.episode(policy, self.target_ref, index, layout)
                episodes.append(episode)
                write_json(self.records.host / "final" / pid / f"episode_{index}.json", json_value(episode))
            strict = sum(e.outcome["strict"] for e in episodes)
            results[pid] = {"episodes": 30, "strict": strict,
                            "lenient": sum(e.outcome["lenient"] for e in episodes), "passed": strict >= 27}
        write_json(self.records.host / "final" / "results.json", results)
        return results  # host API only; never exposed by HarnessService
