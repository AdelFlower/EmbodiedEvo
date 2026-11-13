"""宿主侧 GPU 槽位管理。

沿用外层框架的文件锁方案，只依赖 Python 标准库。
适用于当前 Linux 运行环境。

约定：
- GPU 编号采用宿主机物理编号。
- 需要相互协调的调度器使用相同锁目录和每卡容量。
- 锁目录位于宿主侧，不向 Agent 或 worker 开放。
- executor 确认 worker 已停止后，才能释放槽位。

“槽位”可以理解为一张 GPU 上允许同时开工的名额。每个名额对应一个
锁文件，谁先成功锁住文件，谁获得这个名额；文件存在本身不代表已占用。
这里不运行 GPU 程序，只负责分配名额。
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TextIO


@dataclass(frozen=True)
class GpuLease:
    """一次槽位占用；由宿主 executor 持有。"""

    # frozen=True 禁止重新赋值这些字段；lease 是持有名额的凭证。
    gpu: str
    slot: int
    # 锁依赖打开的文件句柄维持；打印或比较 lease 时忽略这个内部字段。
    _handle: TextIO = field(repr=False, compare=False)


class GpuSlots:
    """通过文件锁限制每张 GPU 的并发 worker 数量。

    per_gpu=1 表示参与本调度协议的 worker 独占使用该卡。
    不探测显存，不阻止未使用本协议的外部进程占用 GPU。
    """

    def __init__(
        self,
        gpus: Sequence[int | str],
        per_gpu: int = 1,
        *,
        directory: str | Path | None = None,
    ):
        # 单个字符串也能被逐字符遍历，必须先拒绝，否则 "12" 会被当成两张卡。
        if isinstance(gpus, (str, bytes)):
            raise ValueError("gpus must be a sequence of physical GPU indices")

        # 统一成数字字符串：例如整数 2 和字符串 "02" 都规范为 "2"。
        normalized = []
        for gpu in gpus:
            value = str(gpu).strip()
            if not value.isascii() or not value.isdigit():
                raise ValueError(
                    "GPU indices must be nonnegative physical GPU indices"
                )
            normalized.append(str(int(value)))

        if not normalized:
            raise ValueError("GPU pool must not be empty")

        # 规范化后再去重，避免同一张卡以不同写法重复进入池中。
        if len(set(normalized)) != len(normalized):
            raise ValueError("GPU indices must be unique")

        if type(per_gpu) is not int or per_gpu < 1:
            raise ValueError("per_gpu must be a positive integer")

        self.gpus = tuple(normalized)
        self.per_gpu = per_gpu

        # 默认以用户编号隔离锁目录。同一用户下需协调的调度器必须用相同目录。
        if directory is None:
            directory = (
                Path("/tmp")
                / f"agenticsimulator-gpu-slots-{os.getuid()}"
            )

        self.directory = Path(directory).expanduser().resolve()
        # 0700 表示新建的目录仅允许所属用户访问。
        self.directory.mkdir(
            mode=0o700,
            parents=True,
            exist_ok=True,
        )

    def acquire(self) -> GpuLease | None:
        """立即尝试申请槽位，无空位时返回 None。

        优先使用各张卡的第一个槽位，再使用第二个槽位。
        等待、超时和取消由 executor 管理。
        """
        # 先遍历槽位、再遍历 GPU，尽量先把任务分散到不同卡上。
        for slot in range(self.per_gpu):
            for gpu in self.gpus:
                path = self.directory / f"gpu-{gpu}-slot-{slot}.lock"
                handle = path.open("a", encoding="utf-8")

                try:
                    # LOCK_EX：独占锁；LOCK_NB：拿不到锁立即报错，不在这里等待。
                    fcntl.flock(
                        handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    # 当前名额已有人使用，关闭本次句柄后继续寻找其他名额。
                    handle.close()
                    continue
                except BaseException:
                    handle.close()
                    raise

                # 成功后不要关闭 handle，交给 lease 持有，直到 release 时释放。
                return GpuLease(
                    gpu=gpu,
                    slot=slot,
                    _handle=handle,
                )

        return None

    @staticmethod
    def release(lease: GpuLease | None) -> None:
        """释放槽位；允许传入 None 或重复释放同一 lease。"""
        # 关闭文件句柄即可释放 flock 锁，无需删除锁文件。
        if lease is not None:
            lease._handle.close()
