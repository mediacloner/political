"""
Lightweight file-based live status for cross-process debate tracking.

The orchestrator writes status updates to output/.live_status.json.
The dashboard polls this file to detect CLI-launched debates.
"""

import json
import time
from pathlib import Path

STATUS_FILE = Path("output/.live_status.json")


def _write(data: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(STATUS_FILE)  # atomic on POSIX


def publish_start(topic: str, max_rounds: int, time_limit_minutes: int) -> None:
    _write({
        "active": True,
        "topic": topic,
        "max_rounds": max_rounds,
        "time_limit_minutes": time_limit_minutes,
        "phase": "research",
        "round": 0,
        "agent": None,
        "started_at": time.time(),
        "updated_at": time.time(),
    })


def publish_phase(phase: str, round_num: int = 0, agent: str = None) -> None:
    status = read()
    if not status:
        return
    status["phase"] = phase
    status["round"] = round_num
    status["agent"] = agent
    status["updated_at"] = time.time()
    _write(status)


def publish_done(finish_reason: str, total_rounds: int) -> None:
    _write({
        "active": False,
        "phase": "done",
        "finish_reason": finish_reason,
        "round": total_rounds,
        "updated_at": time.time(),
    })


def clear() -> None:
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()


def read() -> dict | None:
    if not STATUS_FILE.exists():
        return None
    try:
        return json.loads(STATUS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
