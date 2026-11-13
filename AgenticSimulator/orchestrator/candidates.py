"""源仿真器候选版本管理。

候选只包含源实现，不包含目标配置、policy 或实验结果。
本模块不启动仿真、不运行 Agent 代码、不评价候选效果。

约定：
- root 是宿主侧私有目录，不向 Agent 或 worker 提供可写挂载。
- source_dir 由可信执行层指定，不接受 Agent 提供的任意宿主路径。
- 执行层负责在复制期间阻止源目录继续被修改。
- worker 使用候选的独立运行副本，不直接修改归档。
- 单个会话由一个 CandidateStore 管理。

“候选”就是某次请求保存的源仿真器代码版本，类似给当时的代码拍一张快照。
目录示例：root/c000001/candidate.json 保存登记信息，
root/c000001/source/ 保存代码，root/selected.json 记录最终选中的编号。
状态顺序为 capturing（复制中）→ ready（完整）或 capture_failed（复制失败）。
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
    # fd 是操作系统为已打开文件/目录分配的整数句柄。
    # 相对于已打开的目录操作，比反复拼接并解析完整路径更容易控制访问边界。
    destination.mkdir()

    # 排序使遍历顺序稳定；版本库元数据和 Python 缓存不属于候选源码。
    for name in sorted(os.listdir(source_fd)):
        if name in {".git", "__pycache__"} or name.endswith(".pyc"):
            continue

        # 不跟随符号链接：如果条目是“指向别处的快捷方式”，应检查链接本身。
        info = os.stat(
            name,
            dir_fd=source_fd,
            follow_symlinks=False,
        )
        target = destination / name

        if stat.S_ISDIR(info.st_mode):
            # O_DIRECTORY 要求打开的是目录，O_NOFOLLOW 拒绝最终条目为链接。
            # dir_fd 指定 name 相对于哪个已打开目录寻找。
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
            # 普通文件只读打开；O_NONBLOCK 避免条目意外变为 FIFO 时阻塞打开。
            file_fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=source_fd,
            )
            try:
                # 检查与打开之间条目可能变化，因此对真正打开的对象再检查一次。
                # 这些检查不能替代执行层在复制期间阻止 Agent 修改源目录的约定。
                opened = os.fstat(file_fd)
                if not stat.S_ISREG(opened.st_mode):
                    raise ValueError("Source entry changed to a nonregular file")

                # fd 由 finally 统一关闭。
                # rb 读取原始字节，xb 新建文件且拒绝覆盖，不改变源码内容。
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
        # 即使复制失败，也不能泄漏目录句柄；部分复制结果由调用方保留。
        os.close(source_fd)


class CandidateStore:
    """管理候选编号、代码快照及最终选择，不负责评判哪个版本更好。"""

    def __init__(self, root: str | Path):
        """准备宿主归档目录；已有目录保留，编号冲突在分配时跳过。"""
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_index = 1

    def _allocate(self) -> tuple[str, Path]:
        """分配新编号，不覆盖已有或保存失败的候选目录。"""
        while True:
            # 例如 c000001；即使某个编号复制失败，也不回收或覆盖它。
            candidate_id = f"c{self._next_index:06d}"
            self._next_index += 1
            directory = self.root / candidate_id

            try:
                directory.mkdir()
            except FileExistsError:
                # 包括此前运行留下的目录和不完整快照，均跳过以保留证据。
                continue

            return candidate_id, directory

    def _directory(self, candidate_id: str) -> Path:
        """只接受 c 加至少六位数字，避免把任意路径当成候选编号。"""
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

        # absolute 保留路径中的链接供后续拒绝；resolve 则用于下面的重叠检查。
        source = Path(source_dir).expanduser().absolute()

        # 仅用于检测目录重叠；实际复制仍拒绝符号链接。
        # 两个目录不能互相包含，否则可能把归档复制进自身，或污染待复制源码。
        resolved = source.resolve()
        if (
            self.root.is_relative_to(resolved)
            or resolved.is_relative_to(self.root)
        ):
            raise ValueError("Source and candidate storage must not overlap")

        candidate_id, directory = self._allocate()
        record_path = directory / "candidate.json"

        # 先写 capturing，再复制文件；外部查看时能发现尚未完整保存的版本。
        # 记录只含编号、时间和状态，不含宿主路径，可由执行层用于公开响应。
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

        # 只有整个复制流程正常返回，才把候选标记为可用。
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
            # 把底层“文件不存在”转换为本模块统一的“候选不存在”。
            raise CandidateNotFound("Candidate does not exist") from exc

        return json.loads(text)

    def source_path(self, candidate_id: str) -> Path:
        """取得已完整保存的源实现路径，仅供宿主执行层使用。"""
        record = self.get(candidate_id)
        # 不允许拿未完成或失败的半份源码开展后续实验。
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

        # 只保存选择的编号，不复制代码、不运行实验，也不计算候选效果。
        write_json(
            self.root / "selected.json",
            {"candidate_id": candidate_id},
        )
        return True
