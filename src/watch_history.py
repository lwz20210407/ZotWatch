"""Delivery history: stage on generation, commit only after successful publication/email."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .author_watch import work_key
from .topic_matching import normalize_text


def history_keys(work):
    return {"work:" + work_key(work), "title:" + hashlib.sha256(normalize_text(work.title).encode()).hexdigest()}


class WatchHistory:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "state.json"
        self.pending = self.directory / "pending.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"version": 1, "sent": {}}
        if self.state.get("version") != 1 or not isinstance(self.state.get("sent"), dict):
            raise ValueError("Invalid watch history; refusing to silently reset it")

    def filter(self, works):
        return [w for w in works if not history_keys(w).intersection(self.state["sent"])]

    def stage(self, works):
        state = {"version": 1, "sent": dict(self.state["sent"])}
        stamp = datetime.now(timezone.utc).isoformat()
        for work in works:
            for key in history_keys(work):
                state["sent"][key] = stamp
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def commit(self):
        if not self.pending.exists():
            raise RuntimeError("No pending delivery history")
        json.loads(self.pending.read_text(encoding="utf-8"))
        self.pending.replace(self.path)
