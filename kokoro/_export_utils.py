"""Shared helpers for export scripts that load Kokoro model code without kokoro package __init__.

Export CLIs load kokoro/model.py via importlib so misaki/pipeline are not pulled in.
load_kokoro_for_export encapsulates that with a unique suffix per script for sys.modules.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_path(repo_root: Path, path_rel: str, name: str) -> ModuleType:
    """Load a Python file as a module (same semantics as export scripts)."""
    p = (Path(repo_root) / path_rel).resolve()
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def load_kokoro_for_export(
    repo_root: Path | str | None = None,
    *,
    suffix: str = "",
) -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load istftnet, modules, and model with patched relative imports.

    suffix is appended to each module name (e.g. '_duration', '_for_examples', or ''
    for kokoro_istftnet / kokoro_modules / kokoro_model).

    If repo_root is omitted, it is the repository root (parent of the kokoro package).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    else:
        repo_root = Path(repo_root)

    istftnet_name = f"kokoro_istftnet{suffix}"
    modules_name = f"kokoro_modules{suffix}"
    model_name = f"kokoro_model{suffix}"

    kokoro_istftnet = load_module_from_path(repo_root, "kokoro/istftnet.py", istftnet_name)
    sys.modules[istftnet_name] = kokoro_istftnet

    kokoro_modules_src = (repo_root / "kokoro/modules.py").read_text()
    kokoro_modules_src = kokoro_modules_src.replace(
        "from .istftnet import AdainResBlk1d",
        f"from {istftnet_name} import AdainResBlk1d",
    )
    kokoro_modules = importlib.util.module_from_spec(importlib.util.spec_from_loader(modules_name, loader=None))
    kokoro_modules.__dict__[istftnet_name] = kokoro_istftnet
    kokoro_modules.__dict__["__name__"] = modules_name
    sys.modules[modules_name] = kokoro_modules
    exec(kokoro_modules_src, kokoro_modules.__dict__)

    kokoro_model_src = (repo_root / "kokoro/model.py").read_text()
    kokoro_model_src = kokoro_model_src.replace(
        "from .istftnet import Decoder",
        f"from {istftnet_name} import Decoder",
    )
    kokoro_model_src = kokoro_model_src.replace(
        "from .modules import CustomAlbert, ProsodyPredictor, TextEncoder",
        f"from {modules_name} import CustomAlbert, ProsodyPredictor, TextEncoder",
    )
    kokoro_model = importlib.util.module_from_spec(importlib.util.spec_from_loader(model_name, loader=None))
    kokoro_model.__dict__[istftnet_name] = kokoro_istftnet
    kokoro_model.__dict__[modules_name] = kokoro_modules
    kokoro_model.__dict__["__name__"] = model_name
    sys.modules[model_name] = kokoro_model
    exec(kokoro_model_src, kokoro_model.__dict__)

    return kokoro_istftnet, kokoro_modules, kokoro_model
