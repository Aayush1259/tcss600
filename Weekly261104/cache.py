"""Disk cache: every model response is stored once, keyed by the exact request.

Adapted from the oracle-gap harness (routing/cache.py). Re-running any
script replays cached responses (including their measured latency and
token usage) instead of calling the model again, so no cloud call is paid
twice and every reported number can be regenerated from the cache.

Back this folder up: it is the raw evidence behind every number.
"""
import hashlib
import json
from pathlib import Path


class DiskCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, payload: dict):
        path = self._path(self.key(payload))
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)["response"]

    def put(self, payload: dict, response: dict) -> None:
        path = self._path(self.key(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"request": payload, "response": response}, f,
                      ensure_ascii=False)
        tmp.replace(path)  # atomic: a crash never leaves a half-written entry
