"""Loads the YAML scenario file and builds node specs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .contracts import NodeSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_nodes(cfg: dict[str, Any]) -> list[NodeSpec]:
    """Edge nodes get ids 0..E-1, the cloud is the last node."""
    nodes = []
    for i, e in enumerate(cfg["topology"]["edges"]):
        nodes.append(NodeSpec(node_id=i, tier="edge", price_core_hour=0.0, **e))
    c = dict(cfg["topology"]["cloud"])
    nodes.append(NodeSpec(node_id=len(nodes), tier="cloud", **c))
    return nodes
