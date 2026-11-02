"""Sandbox for submitted session programs.

Keep persistent globals, physics budgets and the four primitives.
Remove simulator objects and filesystem/import capabilities from submitted code.
Use f-strings or percent formatting; format/format_map methods are unavailable.
Class patterns support explicit keyword attributes, not positional attributes.
"""

from __future__ import annotations

import ast
import builtins
import math
from types import SimpleNamespace

import numpy as np


def numerical_numpy():
    names = [
        "abs",
        "all",
        "allclose",
        "any",
        "arange",
        "arccos",
        "arcsin",
        "arctan",
        "arctan2",
        "argmax",
        "argmin",
        "argsort",
        "array",
        "asarray",
        "bool_",
        "clip",
        "concatenate",
        "cos",
        "cross",
        "deg2rad",
        "diag",
        "diff",
        "dot",
        "empty",
        "exp",
        "eye",
        "float32",
        "float64",
        "full",
        "hstack",
        "hypot",
        "identity",
        "inf",
        "int32",
        "int64",
        "interp",
        "isclose",
        "isfinite",
        "isnan",
        "linspace",
        "log",
        "max",
        "maximum",
        "mean",
        "median",
        "meshgrid",
        "min",
        "minimum",
        "nan",
        "ndarray",
        "newaxis",
        "nonzero",
        "ones",
        "pi",
        "rad2deg",
        "ravel",
        "repeat",
        "reshape",
        "round",
        "sign",
        "sin",
        "sort",
        "sqrt",
        "stack",
        "std",
        "sum",
        "tan",
        "tile",
        "transpose",
        "uint8",
        "unique",
        "vstack",
        "where",
        "zeros",
        "zeros_like",
        "ones_like",
    ]
    values = {name: getattr(np, name) for name in names}
    values["linalg"] = SimpleNamespace(
        **{
            name: getattr(np.linalg, name)
            for name in [
                "norm",
                "inv",
                "pinv",
                "solve",
                "lstsq",
                "svd",
                "eig",
                "eigh",
                "det",
                "matrix_rank",
            ]
        }
    )
    values["random"] = SimpleNamespace(
        **{
            name: getattr(np.random, name)
            for name in ["default_rng", "uniform", "normal", "randint", "choice", "seed"]
        }
    )
    return SimpleNamespace(**values)


NUMPY = numerical_numpy()
MATH = SimpleNamespace(**{name: getattr(math, name) for name in dir(math)
                          if not name.startswith("_")})


def numerical_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in {"numpy", "math"}:
        raise ImportError("Only numpy and math are available in robot snippets")
    module = NUMPY if name == "numpy" else MATH
    # A plain ``import numpy as np`` statement passes fromlist=None.
    for attribute in fromlist or ():
        _validate_attribute(attribute)
        if attribute != "*" and not hasattr(module, attribute):
            raise ImportError(f"Unavailable numerical import: {attribute}")
    return module


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "print",
        "range",
        "reversed",
        "round",
        "RuntimeError",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "ValueError",
        "zip",
        "ZeroDivisionError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AssertionError",
    ]
}
SAFE_BUILTINS["__import__"] = numerical_import


def _validate_attribute(name: str) -> None:
    # String replacement fields can resolve attributes outside the submitted AST.
    if name.startswith("_") or name in {"tofile", "dump", "reset", "format", "format_map"}:
        raise ValueError(f"Unavailable attribute in robot snippet: {name}")


def validate_snippet(code: str) -> None:
    tree = ast.parse(code, filename="<interactive-snippet>")
    forbidden = {
        "env",
        "APIS",
        "INPUTS",
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "vars",
        "dir",
        "breakpoint",
        "input",
        "help",
        "memoryview",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith("_")):
            raise ValueError(f"Unavailable name in robot snippet: {node.id}")
        if isinstance(node, ast.Attribute):
            _validate_attribute(node.attr)
        if isinstance(node, ast.MatchClass):
            # Positional patterns obtain unchecked attribute names from __match_args__.
            if node.patterns:
                raise ValueError("Positional class patterns are unavailable in robot snippets")
            for name in node.kwd_attrs:
                _validate_attribute(name)
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module]
            if any(n not in {"numpy", "math"} for n in names) or getattr(node, "level", 0):
                raise ValueError("Only numpy and math imports are supported")
            if isinstance(node, ast.ImportFrom):
                for imported in node.names:
                    _validate_attribute(imported.name)
