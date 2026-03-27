"""
File-based live status for cross-process debate tracking.

The orchestrator writes the full debate state to output/.live_debate.json
after every turn. The dashboard polls this to render CLI-launched debates
in real time with full detail (turns, research, scores).
"""

import json
import time
from pathlib import Path

STATUS_FILE = Path("output/.live_status.json")
DEBATE_FILE = Path("output/.live_debate.json")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str))
    tmp.replace(path)  # atomic on POSIX


def publish_start(topic: str, max_rounds: int, time_limit_minutes: int) -> None:
    status = {
        "active": True,
        "topic": topic,
        "max_rounds": max_rounds,
        "time_limit_minutes": time_limit_minutes,
        "phase": "research",
        "round": 0,
        "agent": None,
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    _write(STATUS_FILE, status)
    # Initialize empty debate file
    _write(DEBATE_FILE, {
        "topic": topic,
        "max_rounds": max_rounds,
        "phase": "research",
        "round_num": 0,
        "turns": [],
        "us_claims": [],
        "china_claims": [],
        "verdict": "",
        "finished": False,
        "finish_reason": "",
        "started_at": time.time(),
    })


def publish_phase(phase: str, round_num: int = 0, agent: str = None) -> None:
    status = read()
    if not status:
        return
    status["phase"] = phase
    status["round"] = round_num
    status["agent"] = agent
    status["updated_at"] = time.time()
    _write(STATUS_FILE, status)
    # Update debate file phase too
    debate = read_debate()
    if debate:
        debate["phase"] = phase
        debate["round_num"] = round_num
        _write(DEBATE_FILE, debate)


def publish_turn(state_dict: dict) -> None:
    """Write the full debate state after each turn."""
    debate = read_debate() or {}
    debate.update({
        "turns": state_dict.get("turns", []),
        "us_claims": state_dict.get("us_claims", []),
        "china_claims": state_dict.get("china_claims", []),
        "evidence_cited": state_dict.get("evidence_cited", {}),
        "round_num": state_dict.get("round_num", 0),
        "updated_at": time.time(),
    })
    _write(DEBATE_FILE, debate)


def publish_scores(round_num: int, scores: dict) -> None:
    """Append quality scores for a round."""
    debate = read_debate()
    if not debate:
        return
    if "scores" not in debate:
        debate["scores"] = {}
    debate["scores"][str(round_num)] = scores
    _write(DEBATE_FILE, debate)


def publish_verdict(verdict: str) -> None:
    debate = read_debate()
    if debate:
        debate["verdict"] = verdict
        debate["phase"] = "done"
        _write(DEBATE_FILE, debate)


def publish_done(finish_reason: str, total_rounds: int) -> None:
    _write(STATUS_FILE, {
        "active": False,
        "phase": "done",
        "finish_reason": finish_reason,
        "round": total_rounds,
        "updated_at": time.time(),
    })
    debate = read_debate()
    if debate:
        debate["finished"] = True
        debate["finish_reason"] = finish_reason
        debate["phase"] = "done"
        _write(DEBATE_FILE, debate)


def clear() -> None:
    for f in (STATUS_FILE, DEBATE_FILE):
        if f.exists():
            f.unlink()


def read() -> dict | None:
    if not STATUS_FILE.exists():
        return None
    try:
        return json.loads(STATUS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def read_debate() -> dict | None:
    if not DEBATE_FILE.exists():
        return None
    try:
        return json.loads(DEBATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
