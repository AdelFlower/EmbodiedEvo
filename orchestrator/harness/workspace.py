"""Copy only declared adapter templates and host-allowlisted source descriptions."""

from pathlib import Path
import shutil

TEMPLATE_FILES = ("instruction.md", "api/README.md", "rewards/default_reward.py")
DIRECTORIES = ("api", "source_reference", "observations", "environments", "rewards", "tools", "notes", "submission")


def _regular_file(path):
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("workspace allowlist requires regular files without symlinks")


def prepare_workspace(workspace, template_dir, source_reference_files):
    workspace, template_dir = Path(workspace), Path(template_dir)
    references = [Path(p) for p in source_reference_files]
    templates = [template_dir / name for name in TEMPLATE_FILES]
    for path in templates + references:
        _regular_file(path)
    if any(path.suffix != ".md" for path in references) or len({p.name for p in references}) != len(references):
        raise ValueError("source reference allowlist requires unique Markdown filenames")
    if any(p.is_symlink() for p in (workspace, *workspace.parents)):
        raise ValueError("workspace destination cannot contain symlinks")
    if workspace.exists() and not workspace.is_dir():
        raise ValueError("workspace destination must be a directory")
    if workspace.exists() and any(p.is_symlink() or not p.is_dir() or p.relative_to(workspace).as_posix() not in DIRECTORIES
                                  for p in workspace.rglob("*")):
        raise ValueError("workspace must be fresh (empty declared directories are accepted)")
    workspace.mkdir(parents=True, exist_ok=True)
    for name in DIRECTORIES:
        (workspace / name).mkdir(exist_ok=True)
    for name, path in zip(TEMPLATE_FILES, templates):
        shutil.copyfile(path, workspace / name)
    for path in references:
        shutil.copyfile(path, workspace / "source_reference" / path.name)
    (workspace / "notes/PROGRESS.md").write_text("# Progress\n\nRecord current evidence, hypotheses, decisions and pending requests here.\n")
