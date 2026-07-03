"""
Local dataset loader.

This is the ONLY place that touches the raw JSON files. Everything else
(tools, scoring, validation) reads through DataStore so we have a single
source of truth and the model can never "invent" a record -- every id
either exists here or it doesn't.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


class DataStore:
    def __init__(self, dataset_dir: str = _DATASET_DIR):
        self.dataset_dir = dataset_dir
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        files = {
            "businesses.json": None,
            "professionals.json": None,
            "opportunities.json": None,
        }
        for fname in files:
            path = os.path.join(self.dataset_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                rid = rec["id"]
                if rid in self._entities:
                    # Keep both but flag collision -- shouldn't happen with our IDs.
                    raise ValueError(f"Duplicate id across dataset files: {rid}")
                self._entities[rid] = rec

    def all(self) -> List[Dict[str, Any]]:
        return list(self._entities.values())

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self._entities.get(entity_id)

    def exists(self, entity_id: str) -> bool:
        return entity_id in self._entities

    def by_entity_type(self, entity_type: str) -> List[Dict[str, Any]]:
        return [r for r in self._entities.values() if r.get("entity_type") == entity_type]


# Module-level singleton, cheap to build (small dataset), but tests can
# still construct their own DataStore(dataset_dir=...) for isolation.
default_store = DataStore()
