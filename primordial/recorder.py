"""State snapshot recording for clips and analysis."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

try:
    import orjson

    def _dumps(obj: object) -> bytes:
        return orjson.dumps(obj)

    def _dump_to_file(obj: object, path: str) -> None:
        with open(path, "wb") as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

except ImportError:
    import json

    def _dumps(obj: object) -> bytes:
        return json.dumps(obj).encode()

    def _dump_to_file(obj: object, path: str) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


if TYPE_CHECKING:
    from .config import RecorderConfig
    from .world import World


class Recorder:
    """Records simulation state snapshots for later replay and analysis."""

    def __init__(self, config: RecorderConfig):
        self.config = config
        self.snapshots: list[dict] = []
        self.stats_history: list[dict] = []
        self.session_id = f"session_{int(time.time())}"

        os.makedirs(config.output_dir, exist_ok=True)

    def record(self, world: World) -> None:
        """Record a snapshot of the current world state."""
        if len(self.snapshots) >= self.config.max_snapshots:
            return

        snapshot = world.to_dict()
        self.snapshots.append(snapshot)
        self.stats_history.append(snapshot["stats"])

    def should_record(self, tick: int) -> bool:
        """Check if we should record at this tick."""
        return tick % self.config.snapshot_interval == 0

    def save_clip(self, name: str | None = None, start: int = 0, end: int | None = None) -> str:
        """Save a range of snapshots as a clip file.

        Args:
            name: Clip name. Auto-generated if None.
            start: Start index in snapshots list.
            end: End index (exclusive). None = all remaining.

        Returns:
            Path to saved clip file.
        """
        if name is None:
            name = f"clip_{self.session_id}_{start}"

        clip_data = {
            "name": name,
            "session_id": self.session_id,
            "snapshot_count": 0,
            "tick_range": [0, 0],
            "snapshots": [],
        }

        selected = self.snapshots[start:end]
        if selected:
            clip_data["snapshots"] = selected
            clip_data["snapshot_count"] = len(selected)
            clip_data["tick_range"] = [selected[0]["tick"], selected[-1]["tick"]]

        path = os.path.join(self.config.output_dir, f"{name}.json")
        _dump_to_file(clip_data, path)
        return path

    def save_stats(self, name: str | None = None) -> str:
        """Save aggregated statistics history."""
        if name is None:
            name = f"stats_{self.session_id}"

        path = os.path.join(self.config.output_dir, f"{name}.json")
        _dump_to_file(self.stats_history, path)
        return path

    def clear(self) -> None:
        """Clear recorded data (but keep stats)."""
        self.snapshots.clear()
