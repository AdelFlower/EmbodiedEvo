"""宿主实验执行器：候选、GPU、子进程和源/目标反馈发布。

由 session.py 调用 create_executor(cfg=..., run=...)。本模块只依赖标准库
和 orchestrator 工具，不导入 Isaac Sim 或任务 adapter。

cfg 来自 task.toml 经配置加载器解析后生成的运行快照。本模块直接使用
cfg["experiment"] 中的 adapter、GPU、policy、回合、源/目标物理和反馈设置，
不再读取第二份实验配置文件。
cfg["config_dir"] 是原 task.toml 所在目录，由加载器自动记录。policy 文件、
target.physics 和可选 gpu_lock_dir 等宿主路径以此为基准，不以运行快照目录
或当前工作目录为基准。source.physics 则相对于本次源候选目录。
省略 target.physics 时使用 adapter 恢复的固定环境配置。
默认用当前 session 进程的 sys.executable 启动仿真，因此应在具备 Isaac Sim
依赖的 Python 环境中启动 session。仅需独立仿真解释器时才配置可选的 python
路径；不依赖固定的机器目录、Conda 环境名称或 PATH 中的 python 命令。

子进程接口（source.py / target.py）：
    python -m AgenticSimulator.simulators.source|target --run-dir RUN --job JOB
子进程先等待 stdin 的一行 start，再初始化模拟器及执行实验；EOF 表示取消。
宿主创建进程成功后登记 on_start，再发送 start，创建进程失败不扣实验预算。
JOB 只含本侧配置、adapter、回合描述、设备、output_dir 和 report_path。
source 另外收到独立的 source_dir，source_probe 还收到已保存的 script_path。
子进程将 adapter 输出写入 output_dir，向 report_path 写入：
    {"ok": true, "result": <adapter 返回字典>}
预期实验失败使用 ok=false，可在 error 中保存诊断，已有 result 仍保留。
result.artifacts 是嵌套的路径字典，所有文件路径相对于 output_dir。
退出码、日志和 report 同时保存；缺失报告、超时或崩溃不自动归因为基础设施故障。

第一版顺序执行两侧。候选捕获依赖客户端串行提交：请求期间不得在后台改写
source 或诊断脚本，尚未接入容器冻结。子进程配置分侧传递，但普通宿主进程
本身不是文件访问隔离；源入口执行 Agent 代码前仍需落实目标私有数据访问边界。
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from AgenticSimulator.orchestrator.candidates import CandidateStore
from AgenticSimulator.orchestrator.files import write_json
from AgenticSimulator.orchestrator.resources import GpuSlots
from AgenticSimulator.orchestrator.session_runner import stop_process


IMPORT_ROOT = Path(__file__).resolve().parents[2]


def create_executor(*, cfg: dict, run: Path):
    """session.py 的工厂入口，不启动模拟器。"""
    return ExperimentExecutor(cfg, run)


class ExperimentExecutor:
    def __init__(self, cfg: dict, run: Path):
        self.run = Path(run).resolve()
        self.workspace = self.run / "workspace"
        self.settings = cfg["experiment"]
        self.adapter = self.settings["adapter"]
        self.base = Path(cfg["config_dir"]).resolve()
        python_path = self.settings.get("python")
        self.python = self._path(python_path) if python_path is not None else Path(sys.executable)
        self.candidates = CandidateStore(self.run / "candidates")
        lock_dir = self.settings.get("gpu_lock_dir")
        self.slots = GpuSlots(
            self.settings["gpus"], self.settings.get("per_gpu", 1),
            directory=self._path(lock_dir) if lock_dir is not None else None,
        )
        self.feedback = self.settings.get("target_feedback", {
            "fields": ["steps", "step_dt"],
            "artifacts": ["videos.overview", "videos.wrist"],
        })
        # 字段与文件分别声明，避免把 artifacts 原路径当普通字段直接公开。
        if "artifacts" in self.feedback.get("fields", []):
            raise ValueError("Declare target artifacts in target_feedback.artifacts")
        self.process = None
        self.lease = None

    def _path(self, value) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.base / path).resolve()

    def execute(
        self, request: dict, *, request_id: str, private_dir: Path,
        public_dir: Path, timeout_sec: float, on_start,
    ) -> dict:
        """一次请求共用截止时间；一侧实验失败不丢弃另一侧结果。"""
        deadline = time.monotonic() + timeout_sec
        private_dir = Path(private_dir)
        private_dir.mkdir(parents=True, exist_ok=False)
        jobs = self._prepare_request(request, request_id, private_dir)
        results = {}
        for side, job in jobs.items():
            side_private = private_dir / side
            if "preparation_error" in job:
                result = {"ok": False, "error": "preparation_failed",
                          "diagnostic": job["preparation_error"], "result": {}}
            else:
                result = self._run_one(side, job, side_private, deadline, on_start)
            if "candidate_id" in job:
                result["candidate_id"] = job["candidate_id"]
            write_json(side_private / "execution.json", result)
            try:
                results[side] = self._publish_result(
                    side, result, side_private, Path(public_dir) / side, deadline,
                )
            except (OSError, ValueError, TimeoutError) as exc:
                # 发布失败也保留实验原始结果，不重跑模拟器。
                detail = traceback.format_exc()
                (side_private / "publication-error.txt").write_text(detail, encoding="utf-8")
                published = {"ok": False, "error": "publication_failed"}
                if side == "source":
                    published["diagnostic"] = str(exc)
                    if "candidate_id" in result:
                        published["candidate_id"] = result["candidate_id"]
                results[side] = published
            write_json(Path(public_dir) / side / "result.json", results[side])
        return {"ok": all(item["ok"] for item in results.values()), "results": results}

    def _prepare_request(self, request, request_id, private_dir):
        """解析宿主配置，固定源代码及 probe；每侧只收到自己需要的配置。"""
        op = request["op"]
        sides = ["source"] if op == "source_probe" else request["environments"]
        if op not in ("rollout", "source_probe"):
            raise ValueError(f"Unsupported experiment operation: {op}")
        policy_id = request.get("policy_id", "default")
        episode_id = request["episode_id"]
        jobs = {}
        for side in sides:
            directory = private_dir / side
            directory.mkdir()
            job = {"op": op, "environment": side, "request_id": request_id}
            jobs[side] = job
            try:
                policy = copy.deepcopy(self.settings["policies"][policy_id])
                for name in ("checkpoint", "params_dir"):
                    if name in policy:
                        policy[name] = str(self._path(policy[name]))
                episode = copy.deepcopy(self.settings["episodes"][episode_id])
                episode["episode_id"] = episode_id
                settings = self.settings[side]
                adapter_config = {**settings.get("adapter_config", {}), **policy}
                job.update(
                    adapter=self.adapter, adapter_config=adapter_config,
                    policy_id=policy_id, episode=episode,
                    output_dir=str(directory / "output"),
                    report_path=str(directory / "report.json"),
                )
                if side == "source":
                    candidate = self.candidates.capture(
                        self.workspace / "source", request_id=request_id,
                    )
                    job["candidate_id"] = candidate["candidate_id"]
                    source = directory / "source"
                    shutil.copytree(self.candidates.source_path(candidate["candidate_id"]), source)
                    job["source_dir"] = str(source)
                    physics = (source / settings["physics"]).resolve()
                    if not physics.is_relative_to(source.resolve()):
                        raise ValueError("Source physics must stay inside the candidate")
                    job["physics"] = str(physics)
                    if op == "source_probe":
                        script = (self.workspace / request["script_path"]).resolve()
                        if not script.is_relative_to(self.workspace) or not script.is_file():
                            raise ValueError("Probe must be a file inside the workspace")
                        saved_script = directory / "probe.py"
                        saved_script.write_bytes(script.read_bytes())
                        job["script_path"] = str(saved_script)
                else:
                    physics = settings.get("physics")
                    job["physics"] = str(self._path(physics)) if physics is not None else None
            except (KeyError, IndexError, OSError, ValueError):
                job["preparation_error"] = traceback.format_exc()
                (directory / "preparation-error.txt").write_text(
                    job["preparation_error"], encoding="utf-8",
                )
        return jobs

    def _run_one(self, environment, job, private_dir, deadline, on_start):
        """等待资源并运行一侧；进程停止后才能释放 GPU 槽位。"""
        result = {"ok": False, "result": {}}
        try:
            while self.lease is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {**result, "error": "request_timeout"}
                self.lease = self.slots.acquire()
                if self.lease is None:
                    time.sleep(min(0.2, remaining))

            job["adapter_config"]["device"] = "cuda:0"
            job["gpu"] = self.lease.gpu
            job_path = private_dir / "job.json"
            write_json(job_path, job)
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = self.lease.gpu
            command = [
                str(self.python), "-u", "-m",
                f"AgenticSimulator.simulators.{environment}",
                "--run-dir", str(self.run), "--job", str(job_path),
            ]
            if time.monotonic() >= deadline:
                return {**result, "error": "request_timeout"}
            with (
                (private_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
                (private_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
            ):
                try:
                    self.process = subprocess.Popen(
                        command, cwd=IMPORT_ROOT, env=env,
                        stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                        start_new_session=True,
                    )
                except OSError as exc:
                    return {**result, "error": "launch_failed", "diagnostic": str(exc)}
                # 子进程等待 start，先登记实验开始，再允许它初始化和执行。
                # 解释器已启动后的初始化失败同样保留预算；不自动重试。
                on_start(environment)
                try:
                    self.process.stdin.write(b"start\n")
                    self.process.stdin.flush()
                except BrokenPipeError:
                    # 子进程可能在接收 start 前已报错退出，下面仍收集退出码与日志。
                    pass
                finally:
                    try:
                        self.process.stdin.close()
                    except BrokenPipeError:
                        pass
                try:
                    returncode = self.process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    return {**result, "error": "request_timeout"}

            result["returncode"] = returncode
            report_path = Path(job["report_path"])
            if not report_path.is_file():
                return {**result, "error": "missing_report"}
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (not isinstance(report, dict) or type(report.get("ok")) is not bool
                        or not isinstance(report.get("result", {}), dict)):
                    raise ValueError("Expected {ok: bool, result: object}")
            except (ValueError, UnicodeError) as exc:
                return {**result, "error": "invalid_report", "diagnostic": str(exc)}
            result["result"] = report.get("result", {})
            result["ok"] = returncode == 0 and report["ok"]
            if not result["ok"]:
                result["error"] = "process_failed" if returncode else "experiment_failed"
                result["diagnostic"] = report.get("error")
            return result
        finally:
            self._stop_active()

    def _publish_result(self, environment, result, private_dir, public_dir, deadline):
        """先选公开内容再复制文件；目标侧不复制完整报告、轨迹或日志。"""
        public_dir.mkdir(parents=True, exist_ok=False)
        raw = result.get("result", {})
        if environment == "source":
            published = {key: value for key, value in raw.items() if key != "artifacts"}
            for key in ("candidate_id", "returncode", "diagnostic"):
                if key in result:
                    published[key] = result[key]
        else:
            published = {
                key: raw[key] for key in self.feedback.get("fields", []) if key in raw
            }
        published["ok"] = result["ok"]
        if not result["ok"]:
            # 这里只公开执行器自己的错误码，不透传目标进程 error/traceback。
            published["error"] = result["error"]

        artifacts = {}
        output = (private_dir / "output").resolve()

        def publish_files(tree, destination, prefix=()):
            if not isinstance(tree, dict):
                raise ValueError("Artifacts must be a dictionary of relative file paths")
            for key, value in tree.items():
                parts = (*prefix, key)
                if isinstance(value, dict):
                    child = {}
                    publish_files(value, child, parts)
                    if child:
                        destination[key] = child
                    continue
                if environment == "target" and ".".join(parts) not in self.feedback.get("artifacts", []):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("Request deadline reached while publishing artifacts")
                if not isinstance(value, str):
                    raise ValueError("Artifact path must be a string")
                path = (output / value).resolve()
                if not path.is_relative_to(output) or not path.is_file():
                    raise ValueError("Artifact must be a file inside the experiment output directory")
                # 使用输出目录内的相对路径，不用 manifest 的字段名拼文件系统路径。
                destination_path = public_dir / "artifacts" / path.relative_to(output)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination_path)
                destination[key] = str(destination_path.relative_to(self.workspace))

        publish_files(raw.get("artifacts", {}), artifacts)
        if environment == "source":
            logs = {}
            for name in ("stdout.log", "stderr.log", "preparation-error.txt"):
                path = private_dir / name
                if path.is_file():
                    shutil.copyfile(path, public_dir / name)
                    logs[name] = str((public_dir / name).relative_to(self.workspace))
            published["logs"] = logs
        published["artifacts"] = artifacts
        return published

    def select_candidate(self, candidate_id: str) -> bool:
        return self.candidates.select(candidate_id)

    def _stop_active(self) -> None:
        """复用进程组清理；停止失败时保留句柄及槽位供 close 再次清理。"""
        if self.process is not None:
            try:
                stop_process(self.process)
            finally:
                # on_start 抛出异常时，start 尚未发送，仍须关闭父进程持有的管道。
                if self.process.stdin is not None and not self.process.stdin.closed:
                    self.process.stdin.close()
            self.process = None
        self.slots.release(self.lease)
        self.lease = None

    def close(self) -> None:
        self._stop_active()
