"""Config 로더: _base_ 상속을 해석하고 deep-merge 한다.

model config(cnn.yaml / vit.yaml)는 base.yaml을 상속하고 architecture 블록만
덮어쓴다. 이 로더가 base를 먼저 읽고 그 위에 model config를 얹어 병합한다.
"""
from __future__ import annotations
import copy
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_name = cfg.pop("_base_", None)
    if base_name is not None:
        base_path = path.parent / base_name
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg
