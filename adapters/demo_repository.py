from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_ROOT / "demo_data"


def _read_json(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_demo() -> dict[str, Any]:
    blocks = _read_json("evidence_blocks.json")
    return {
        "samples": _read_json("sample_merged.json"),
        "formal_samples": _read_json("formal_samples.json"),
        "results": _read_json("result.json"),
        "meta": _read_json("run_meta.json"),
        "blocks": blocks,
        "blocks_by_id": {str(block.get("block_id")): block for block in blocks},
        "result_text": (DATA_DIR / "result.txt").read_text(encoding="utf-8"),
    }


def flatten_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        if len(value) > 8 and value[0] in {"P", "T", "H"} and "_" in value:
            found.append(value)
    elif isinstance(value, list):
        for item in value:
            found.extend(flatten_ids(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "evidence_block_id",
                "evidence_block_ids",
                "composition",
                "monomers",
                "conflict",
                "conflicts",
                "all_candidates",
            }:
                found.extend(flatten_ids(item))
    return list(dict.fromkeys(found))


def property_record(result: dict[str, Any], property_name: str) -> dict[str, Any]:
    raw = result.get("properties_raw") or {}
    value = raw.get(property_name)
    return value if isinstance(value, dict) else {}