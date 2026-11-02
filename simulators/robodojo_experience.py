"""Preserve the RoboDojo experience when using its pinned Isaac Sim image."""

from __future__ import annotations

import tomllib
from pathlib import Path

URDF_PIN = '"isaacsim.asset.importer.urdf" = {version = "2.4.31", exact = true}'


def prepare_experience(source: Path, destination: Path, installed_extension: Path) -> Path:
    """Adapt the known URDF patch-version mismatch without changing vendor files."""
    if not installed_extension.is_file():
        return source
    original = source.read_text(encoding="utf-8")
    if URDF_PIN not in original:
        return source
    installed = tomllib.loads(installed_extension.read_text(encoding="utf-8"))["package"]["version"]
    if installed != "2.4.30":
        return source

    # Kit permits repeated [settings] tables, so retain its text instead of
    # parsing and reserializing the experience with a strict TOML parser.
    adapted = original.replace(URDF_PIN, URDF_PIN.replace("2.4.31", "2.4.30"))
    adapted = adapted.replace("${app}", str(source.parent.resolve()))
    destination.write_text(adapted, encoding="utf-8")
    print(
        "[EmbodiedEvo] Isaac Sim experience compatibility: "
        "isaacsim.asset.importer.urdf requested 2.4.31; using installed 2.4.30. "
        f"Source: {source}; selected experience: {destination}"
    )
    return destination
