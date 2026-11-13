"""源仿真器候选版本管理。

候选只包含源实现，不包含目标配置、policy 或实验结果。
本模块不启动仿真、不运行 Agent 代码、不评价候选效果。

约定：
- root 是宿主侧私有目录，不向 Agent 或 worker 提供可写挂载。
- source_dir 由可信执行层指定，不接受 Agent 提供的任意宿主路径。
- 执行层负责在复制期间阻止源目录继续被修改。
- worker 使用候选的独立运行副本，不直接修改归档。
- 单个会话由一个 CandidateStore 管理。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
from pathlib import Path

from .files import write_json


class CandidateNotFound(ValueError):
    """候选编号无效或候选不存在。"""


def _copy_directory(source_fd: int, destination: Path) -> None:
    """通过目录描述符复制，拒绝符号链接及特殊文件。

    面向当前 Linux 运行环境；不跟随 Agent 控制的符号链接。
    """
    destination.mkdir()

    for name in sorted(os.listdir(source_fd)):
        if name in {".git", "__pycache__"} or name.endswith(".pyc"):
            continue

        info = os.stat(
            name,
            dir_fd=source_fd,
            follow_symlinks=False,
        )
        target = destination / name

        if stat.S_ISDIR(info.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=source_fd,
            )
            try:
                _copy_directory(child_fd, target)
            finally:
                os.close(child_fd)

        elif stat.S_ISREG(info.st_mode):
            file_fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=source_fd,
            )
            try:
                opened = os.fstat(file_fd)
                if not stat.S_ISREG(opened.st_mode):
                    raise ValueError("Source entry changed to a nonregular file")

                # fd 由 finally 统一关闭。
                with (
                    os.fdopen(file_fd, "rb", closefd=False) as source,
                    target.open("xb") as output,
                ):
                    shutil.copyfileobj(source, output)

                # 保留普通权限及执行位，不复制特殊权限位。
                target.chmod(opened.st_mode & 0o777)
            finally:
                os.close(file_fd)

        else:
            raise ValueError(
                f"Source contains a symlink or special file: {name!r}"
            )


def _copy_source(source: Path, destination: Path) -> None:
    """打开源目录后复制，源目录本身也不能是符号链接。"""
    source_fd = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        _copy_directory(source_fd, destination)
    finally:
        os.close(source_fd)


class CandidateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_index = 1

    def _allocate(self) -> tuple[str, Path]:
        """分配新编号，不覆盖已有或保存失败的候选目录。"""
        while True:
            candidate_id = f"c{self._next_index:06d}"
            self._next_index += 1
            directory = self.root / candidate_id

            try:
                directory.mkdir()
            except FileExistsError:
                continue

            return candidate_id, directory

    def _directory(self, candidate_id: str) -> Path:
        if (
            not isinstance(candidate_id, str)
            or re.fullmatch(r"c[0-9]{6,}", candidate_id) is None
        ):
            raise CandidateNotFound("Invalid candidate ID")

        return self.root / candidate_id

    def capture(
        self,
        source_dir: str | Path,
        *,
        request_id: str,
    ) -> dict:
        """保存源实现，返回不含宿主路径的候选记录。"""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

        source = Path(source_dir).expanduser().absolute()

        # 仅用于检测目录重叠；实际复制仍拒绝符号链接。
        resolved = source.resolve()
        if (
            self.root.is_relative_to(resolved)
            or resolved.is_relative_to(self.root)
        ):
            raise ValueError("Source and candidate storage must not overlap")

        candidate_id, directory = self._allocate()
        record_path = directory / "candidate.json"

        record = {
            "candidate_id": candidate_id,
            "request_id": request_id,
            "created_at": time.time(),
            "status": "capturing",
        }
        write_json(record_path, record)

        try:
            _copy_source(source, directory / "source")
        except BaseException:
            # 保留部分文件和编号，不自动删除或重试。
            record["status"] = "capture_failed"
            write_json(record_path, record)
            raise

        record["status"] = "ready"
        write_json(record_path, record)
        return record

    def get(self, candidate_id: str) -> dict:
        """查询候选记录，包括未完成或保存失败的记录。"""
        directory = self._directory(candidate_id)

        try:
            text = (directory / "candidate.json").read_text(
                encoding="utf-8"
            )
        except FileNotFoundError as exc:
            raise CandidateNotFound("Candidate does not exist") from exc

        return json.loads(text)

    def source_path(self, candidate_id: str) -> Path:
        """取得已完整保存的源实现路径，仅供宿主执行层使用。"""
        record = self.get(candidate_id)
        if record["status"] != "ready":
            raise ValueError("Candidate capture is incomplete")

        return self._directory(candidate_id) / "source"

    def select(self, candidate_id: str) -> bool:
        """记录最终选择；不存在或保存不完整时返回 False。"""
        try:
            record = self.get(candidate_id)
        except CandidateNotFound:
            return False

        if record["status"] != "ready":
            return False

        write_json(
            self.root / "selected.json",
            {"candidate_id": candidate_id},
        )
        return True