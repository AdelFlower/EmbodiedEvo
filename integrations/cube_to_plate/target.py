"""Host-private target schema and repository-path boundary."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, repr=False)
class TargetRef:
    _spec: dict = field(repr=False)


def load_target_spec(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPOSITORY) and not path.is_relative_to(REPOSITORY / "local"):
        raise ValueError("target spec must be outside the repository or under local/")
    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError:
        raise ValueError("target spec: invalid JSON") from None
    if (not isinstance(spec, dict) or set(spec) != {"kind", "cube_material", "cube_mass_kg"}
            or spec["kind"] != "cube_to_plate_target_v1"):
        raise ValueError("target spec: invalid schema")
    material = spec["cube_material"]
    if not isinstance(material, dict) or set(material) != {"static_friction", "dynamic_friction", "restitution"}:
        raise ValueError("target spec: invalid material schema")
    values = [spec["cube_mass_kg"], material["static_friction"], material["dynamic_friction"], material["restitution"]]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        raise ValueError("target spec: finite numbers required")
    if not all(v > 0 for v in values[:3]) or not 0 <= values[3] <= 1:
        raise ValueError("target spec: invalid numeric domain")
    return TargetRef(spec)
