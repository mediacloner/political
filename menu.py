#!/usr/bin/env python3
"""
Politics AI Swarm — terminal control menu.

Usage:
    python menu.py
"""

import os
import sys
import json
import copy
import time
import subprocess
import yaml
from pathlib import Path

# Auto-activate .venv if running under system Python
_VENV = Path(__file__).parent / ".venv"
_VENV_PY = _VENV / "bin" / "python"
if _VENV_PY.exists() and not sys.prefix.startswith(str(_VENV)):
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)

sys.path.insert(0, str(Path(__file__).parent))

# ANSI
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
BLU="\033[94m"; GRN="\033[92m"; YLW="\033[93m"
RED="\033[91m"; CYN="\033[96m"; MAG="\033[95m"; ORG="\033[33m"


def clr():
    os.system("cls" if os.name == "nt" else "clear")


def hdr():
    print(f"{B}{BLU}")
    print("  ╔════════════════════════════════════════════╗")
    print("  ║    Politics AI Swarm — Control Menu       ║")
    print("  ╚════════════════════════════════════════════╝")
    print(f"{R}")


def ask(prompt, default=None):
    sfx = f" [{default}]" if default is not None else ""
    val = input(f"  {CYN}{prompt}{sfx}:{R} ").strip()
    return val if val else default


def ask_int(prompt, default=None):
    v = ask(prompt, str(default) if default is not None else None)
    try:
        return int(v) if v else default
    except ValueError:
        return default


def ask_yn(prompt, default="n"):
    v = ask(f"{prompt} (y/n)", default)
    return (v or "").lower() == "y"


def ok(msg):
    print(f"  {GRN}✓{R}  {msg}")


def err(msg):
    print(f"  {RED}✗{R}  {msg}")


def info(msg):
    print(f"  {DIM}{msg}{R}")


def pause():
    input(f"\n  {DIM}[Enter to continue]{R}")


# ── Actions ───────────────────────────────────────────────────────────────────

def check_status():
    print(f"\n{B}System Status{R}\n")
    try:
        from src.orchestrator import load_config
        config = load_config("config/settings.yaml")
        from src.tabby_client import TabbyClient
        client = TabbyClient(base_url=config["tabbyapi"]["url"],
                             api_key=config["tabbyapi"].get("api_key", ""))
        alive = client.is_alive()
        dot = f"{GRN}●{R}" if alive else f"{RED}●{R}"
        status = f"{GRN}ONLINE{R}" if alive else f"{RED}OFFLINE{R}"
        print(f"  {dot} TabbyAPI  {config['tabbyapi']['url']}  →  {status}")

        txdir = Path(config["output"]["transcripts_dir"])
        txcount = len(list(txdir.glob("*.json"))) if txdir.exists() else 0
        audio_dir = Path(config["output"]["audio_dir"])
        audio_count = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
        print(f"\n  Transcripts: {txcount}   Audio files: {audio_count}")

        print(f"\n{B}  Models:{R}")
        for key, mcfg in config["models"].items():
            path = Path(mcfg["path"].replace("~", str(Path.home())))
            mark = f"{GRN}✓{R}" if path.exists() else f"{RED}✗ not found{R}"
            print(f"  [{key:<5}] {mcfg['name']:<50}  {mark}")
    except Exception as e:
        err(str(e))


PROJECT_DIR  = Path(__file__).parent
PROJECT_VENV = PROJECT_DIR / ".venv"

TABBY_DIR  = PROJECT_DIR / "tabbyAPI"
TABBY_VENV = TABBY_DIR / "venv"          # path start.py / update_deps.sh expect


def _project_python() -> str:
    for name in ("python", "python3"):
        p = PROJECT_VENV / "bin" / name
        if p.exists():
            return str(p)
    return str(PROJECT_VENV / "bin" / "python")


def _ensure_project_venv() -> bool:
    """Create .venv and install requirements.txt if not already done."""
    if any((PROJECT_VENV / "bin" / n).exists() for n in ("python", "python3")):
        return True
    print(f"\n  {YLW}Creating project .venv …{R}")
    result = subprocess.run([sys.executable, "-m", "venv", str(PROJECT_VENV)])
    if result.returncode != 0:
        err("python3 -m venv failed")
        return False
    ok("Project virtual environment created")
    # Always install Flask (needed for dashboard); then attempt full requirements
    subprocess.run([_project_python(), "-m", "pip", "install", "-q", "flask", "pyyaml", "requests"])
    req = PROJECT_DIR / "requirements.txt"
    if req.exists():
        print(f"  {YLW}Installing requirements.txt (heavy deps may take a while) …{R}")
        subprocess.run([_project_python(), "-m", "pip", "install", "-q", "-r", str(req)])
    ok("Dependencies installed")
    return True


def _venv_python() -> str:
    for name in ("python", "python3"):
        p = TABBY_VENV / "bin" / name
        if p.exists():
            return str(p)
    return str(TABBY_VENV / "bin" / "python")  # will fail with a clear error


def _tabby_env() -> dict:
    """OS environment with the venv bin/ prepended so bare 'pip' resolves to venv pip."""
    env = os.environ.copy()
    env["PATH"] = str(TABBY_VENV / "bin") + ":" + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(TABBY_VENV)
    env.pop("PYTHONHOME", None)
    return env


def _ensure_venv() -> bool:
    """Create tabbyAPI/venv if it doesn't exist yet. Returns True on success."""
    if any((TABBY_VENV / "bin" / n).exists() for n in ("python", "python3")):
        return True
    if not TABBY_DIR.exists():
        err("tabbyAPI/ directory not found — run: git submodule update --init")
        return False
    print(f"\n  {YLW}Creating tabbyAPI/venv …{R}")
    result = subprocess.run([sys.executable, "-m", "venv", str(TABBY_VENV)])
    if result.returncode != 0:
        err("python3 -m venv failed")
        return False
    ok("Virtual environment created")
    # Reset first_run_done so start.py re-installs deps into the fresh venv
    opts_path = TABBY_DIR / "start_options.json"
    if opts_path.exists():
        try:
            opts = json.loads(opts_path.read_text())
            opts["first_run_done"] = False
            opts_path.write_text(json.dumps(opts))
        except Exception:
            pass  # non-fatal; start.py will handle it
    return True


def start_tabbyapi():
    if not _ensure_venv():
        return
    print(f"\n  Starting TabbyAPI in background (first run installs deps)…")
    try:
        subprocess.Popen(
            [_venv_python(), "start.py"],
            cwd=str(TABBY_DIR),
            env=_tabby_env(),
            start_new_session=True,
        )
        ok("TabbyAPI started — press [s] in ~10 s to verify")
    except Exception as e:
        err(str(e))


def ensure_tabbyapi() -> bool:
    """Start TabbyAPI if not running, wait up to 90 s for it to respond."""
    from src.orchestrator import load_config
    from src.tabby_client import TabbyClient
    config = load_config("config/settings.yaml")
    client = TabbyClient(base_url=config["tabbyapi"]["url"],
                         api_key=config["tabbyapi"].get("api_key", ""))

    if client.is_alive():
        return True

    if not _ensure_venv():
        return False

    print(f"\n  {YLW}TabbyAPI offline — starting (first run installs deps, may take ~2 min)…{R}")
    try:
        subprocess.Popen(
            [_venv_python(), "start.py"],
            cwd=str(TABBY_DIR),
            env=_tabby_env(),
            start_new_session=True,
        )
    except Exception as e:
        err(f"Failed to launch TabbyAPI: {e}")
        return False

    print(f"  Waiting", end="", flush=True)
    for _ in range(45):          # up to 90 seconds
        time.sleep(2)
        print(".", end="", flush=True)
        if client.is_alive():
            print(f" {GRN}ready!{R}")
            return True

    print()
    err("TabbyAPI did not respond after 90 s")
    info("Check the process output: cd tabbyAPI && venv/bin/python start.py")
    return False


def _load_core():
    from src.orchestrator import load_config, load_personas, Orchestrator
    config = load_config("config/settings.yaml")
    personas = load_personas("config/personas.yaml")
    return config, personas, Orchestrator


def run_debate(skip_research=False, quick=False):
    print(f"\n{B}Debate Configuration{R}\n")

    topic = ask("Topic", "Should the EU partner with NASA or CNSA for a lunar base?")
    if not topic:
        err("Topic required")
        return

    if quick:
        max_rounds, time_limit, skip_research = 1, 5, True
        print(f"  {DIM}Quick test: 1 round, 5 min, no research{R}")
    else:
        max_rounds = ask_int("Max rounds", 8)
        time_limit = ask_int("Time limit (min)", 20)

    print(f"\n  US Persona:")
    print(f"    [1] Default (Ambassador V. Marsh / Strategist)")
    print(f"    [2] Director James Harrington (Lobbyist)")
    print(f"    [3] Dr. Sarah Chen (Analyst)")
    us_choice = ask("Choice", "1")
    us_persona = {"1": None, "2": "lobbyist", "3": "analyst"}.get(us_choice)

    print(f"\n  China Persona:")
    print(f"    [1] Default (Dir-Gen Wei Changming / Director)")
    print(f"    [2] Chairman Liu Peng (Enterprise Rep)")
    print(f"    [3] Prof. Zhang Yifei (Economist)")
    cn_choice = ask("Choice", "1")
    china_persona = {"1": None, "2": "enterprise_rep", "3": "economist"}.get(cn_choice)

    print(f"\n  EU Persona:")
    print(f"    [1] Default (Commissioner Elise Fontaine / Strategist)")
    print(f"    [2] Admiral Henrik Sørensen (Hawk)")
    print(f"    [3] Dr. Marie Leclerc (Economist)")
    eu_choice = ask("Choice", "1")
    eu_persona = {"1": None, "2": "hawk", "3": "economist"}.get(eu_choice)

    podcast = False
    if not quick:
        podcast = ask_yn("Generate podcast?", "n")

    if not ensure_tabbyapi():
        return

    config, personas, Orchestrator = _load_core()
    if skip_research:
        config = copy.deepcopy(config)
        config["research"]["enabled"] = False

    print(f"\n  {GRN}Starting: {topic}{R}\n")
    try:
        orch = Orchestrator(config=config, personas=personas,
                            us_persona=us_persona, china_persona=china_persona,
                            eu_persona=eu_persona)
        orch.run_debate(topic=topic, max_rounds=max_rounds,
                        time_limit_minutes=time_limit, produce_podcast=podcast)
    except RuntimeError as e:
        err(str(e))
        info("Start TabbyAPI first with [t]")
    except KeyboardInterrupt:
        print(f"\n  {YLW}Interrupted{R}")


def list_transcripts(show=True):
    from src.orchestrator import load_config
    config = load_config("config/settings.yaml")
    txdir = Path(config["output"]["transcripts_dir"])
    if not txdir.exists():
        if show:
            print(f"  {YLW}No transcripts directory{R}")
        return [], config
    files = sorted(txdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        if show:
            print(f"  {YLW}No transcripts found{R}")
        return [], config
    if show:
        print(f"\n{B}  Saved Transcripts{R}\n")
        for i, f in enumerate(files[:20], 1):
            try:
                d = json.loads(f.read_text())
                from datetime import datetime
                ts = datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
                topic = d.get("topic", "?")[:58]
                rounds = d.get("round_num", 0)
                reason = d.get("finish_reason", "")
                print(f"  [{i:>2}] {BLU}{topic}{R}")
                print(f"       {DIM}{rounds} rounds · {reason} · {ts}{R}")
            except Exception:
                pass
    return files[:20], config


def view_transcript():
    files, _ = list_transcripts()
    if not files:
        return
    idx = ask_int("Select number", 1)
    if not idx or not (1 <= idx <= len(files)):
        return
    d = json.loads(files[idx - 1].read_text())
    print(f"\n{B}{'='*62}{R}")
    print(f"{B}  {d.get('topic')}{R}")
    print(f"  {DIM}{d.get('round_num')} rounds · {d.get('finish_reason')}{R}")
    print(f"{B}{'='*62}{R}\n")
    colors = {"us": BLU, "china": ORG, "judge": MAG}
    labels = {"us": "🇺🇸 US", "china": "🇨🇳 China", "judge": "🇪🇺 Judge"}
    for turn in d.get("turns", []):
        c = colors.get(turn["agent"], R)
        print(f"{B}{c}{labels.get(turn['agent'], turn['agent'])}  Round {turn['round']}{R}")
        for line in turn.get("content", "").split("\n"):
            print(f"  {line}")
        print()
    if d.get("verdict"):
        print(f"{B}{MAG}🇪🇺 Final Verdict{R}")
        for line in d["verdict"].split("\n"):
            print(f"  {line}")


def generate_podcast():
    files, config = list_transcripts()
    if not files:
        return
    idx = ask_int("Select transcript number", 1)
    if not idx or not (1 <= idx <= len(files)):
        return
    f = files[idx - 1]
    d = json.loads(f.read_text())
    print(f"\n  Topic: {BLU}{d.get('topic')}{R}")
    info("Optional: server-side paths to voice reference .wav files (leave blank to skip)")
    vus = ask("US voice ref (.wav)", "")
    vcn = ask("China voice ref (.wav)", "")
    vju = ask("Judge voice ref (.wav)", "")
    voice_refs = {k: v for k, v in [("us", vus), ("china", vcn), ("judge", vju)] if v}

    if not ensure_tabbyapi():
        return

    from src.context.debate_state import DebateState
    from src.tts.podcast import PodcastProducer
    from src.tabby_client import TabbyClient

    _, personas, _ = _load_core()
    from src.orchestrator import load_config
    config = load_config("config/settings.yaml")
    client = TabbyClient(base_url=config["tabbyapi"]["url"],
                         api_key=config["tabbyapi"].get("api_key", ""))
    state = DebateState.from_dict(d)
    print(f"\n  {GRN}Generating podcast…{R}\n")
    try:
        result = PodcastProducer(config, personas).produce(
            state, client, voice_refs or None, source_transcript=f.name)
        if result:
            ok(f"Podcast saved: {result}")
        else:
            print(f"  {YLW}Script saved — audio skipped (install Fish Speech for audio){R}")
    except Exception as e:
        err(str(e))


def _load_script(path: Path) -> dict:
    """Load and validate a debate script YAML file."""
    with open(path) as f:
        script = yaml.safe_load(f)
    required = ["topic"]
    for key in required:
        if key not in script:
            raise ValueError(f"Script missing required field: {key}")
    return script


def list_scripts():
    scripts_dir = PROJECT_DIR / "scripts"
    if not scripts_dir.exists():
        print(f"  {YLW}No scripts/ directory{R}")
        return []
    files = sorted(scripts_dir.glob("*.yaml"))
    if not files:
        print(f"  {YLW}No scripts found in scripts/{R}")
        return []
    print(f"\n{B}  Debate Scripts{R}\n")
    for i, f in enumerate(files, 1):
        try:
            s = _load_script(f)
            topic = s["topic"][:60]
            personas_info = s.get("personas", {})
            p_str = ", ".join(f"{k}={v}" for k, v in personas_info.items())
            rounds = s.get("rounds", 8)
            research = "research" if s.get("research", True) else "no research"
            podcast = " + podcast" if s.get("podcast", False) else ""
            print(f"  {B}{CYN}[{i:>2}]{R}  {BLU}{topic}{R}")
            print(f"       {DIM}{rounds} rounds · {research}{podcast} · {p_str}{R}")
        except Exception as e:
            print(f"  {B}{CYN}[{i:>2}]{R}  {RED}{f.name}: {e}{R}")
    return files


def run_script():
    files = list_scripts()
    if not files:
        return
    idx = ask_int("Select script", 1)
    if not idx or not (1 <= idx <= len(files)):
        return

    script = _load_script(files[idx - 1])
    topic = script["topic"]
    max_rounds = script.get("rounds", 8)
    time_limit = script.get("time_limit_minutes", 20)
    skip_research = not script.get("research", True)
    podcast = script.get("podcast", False)
    personas = script.get("personas", {})

    us_persona = personas.get("us")
    china_persona = personas.get("china")
    eu_persona = personas.get("judge")

    # Allow defaults (None) when persona is "default" or not specified
    if us_persona == "default":
        us_persona = None
    if china_persona == "default":
        china_persona = None
    if eu_persona == "default":
        eu_persona = None

    print(f"\n  {B}Script:{R} {files[idx - 1].name}")
    print(f"  {B}Topic:{R}  {BLU}{topic}{R}")
    print(f"  {B}Config:{R} {max_rounds} rounds, {time_limit} min"
          f"{', no research' if skip_research else ''}"
          f"{', podcast' if podcast else ''}")
    print()

    if not ensure_tabbyapi():
        return

    config, personas_data, Orchestrator = _load_core()
    if skip_research:
        config = copy.deepcopy(config)
        config["research"]["enabled"] = False

    print(f"\n  {GRN}Starting: {topic}{R}\n")
    try:
        orch = Orchestrator(config=config, personas=personas_data,
                            us_persona=us_persona, china_persona=china_persona,
                            eu_persona=eu_persona)
        orch.run_debate(topic=topic, max_rounds=max_rounds,
                        time_limit_minutes=time_limit, produce_podcast=podcast)
    except RuntimeError as e:
        err(str(e))
        info("Start TabbyAPI first with [t]")
    except KeyboardInterrupt:
        print(f"\n  {YLW}Interrupted{R}")


def launch_dashboard():
    if not _ensure_project_venv():
        return
    port = ask_int("Dashboard port", 8000)
    print(f"\n  {GRN}Dashboard →  http://127.0.0.1:{port}{R}")
    info("Press Ctrl+C to stop\n")
    try:
        subprocess.run([_project_python(), str(PROJECT_DIR / "dashboard.py"), "--port", str(port)])
    except KeyboardInterrupt:
        print(f"\n  {YLW}Dashboard stopped{R}")


# ── Menu definition ───────────────────────────────────────────────────────────

MENU = [
    ("s", "System status",                         check_status),
    ("t", "Start TabbyAPI server",                  start_tabbyapi),
    (None, None, None),
    ("1", f"Full run      {DIM}research → debate → verdict → podcast (optional){R}", lambda: run_debate()),
    ("2", f"Debate only   {DIM}skip web research{R}",           lambda: run_debate(skip_research=True)),
    ("3", f"Quick test    {DIM}1 round, no research{R}",        lambda: run_debate(quick=True)),
    (None, None, None),
    ("4", f"Run script     {DIM}load a pre-configured debate scenario{R}", run_script),
    ("5", f"List scripts   {DIM}show available scripts{R}",              lambda: list_scripts()),
    (None, None, None),
    ("6", "List transcripts",                       list_transcripts),
    ("7", "View transcript",                        view_transcript),
    ("8", "Generate podcast from transcript",       generate_podcast),
    (None, None, None),
    ("9", "Launch web dashboard",                   launch_dashboard),
    ("q", "Quit",                                   None),
]


def main():
    while True:
        clr()
        hdr()
        for key, label, _ in MENU:
            if key is None:
                print()
            else:
                print(f"  {B}{CYN}[{key}]{R}  {label}")
        print()
        choice = input(f"  {CYN}>{R} ").strip().lower()
        print()
        for key, _, fn in MENU:
            if key == choice:
                if fn is None:
                    print("  Goodbye.")
                    sys.exit(0)
                try:
                    fn()
                except KeyboardInterrupt:
                    print(f"\n  {YLW}Interrupted{R}")
                break
        else:
            print(f"  {RED}Unknown option '{choice}'{R}")
        pause()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--script":
        script_path = Path(sys.argv[2])
        if not script_path.exists():
            err(f"Script not found: {script_path}")
            sys.exit(1)
        script = _load_script(script_path)
        topic = script["topic"]
        max_rounds = script.get("rounds", 8)
        time_limit = script.get("time_limit_minutes", 20)
        skip_research = not script.get("research", True)
        podcast = script.get("podcast", False)
        personas = script.get("personas", {})
        us_p = personas.get("us") if personas.get("us") != "default" else None
        cn_p = personas.get("china") if personas.get("china") != "default" else None
        eu_p = personas.get("judge") if personas.get("judge") != "default" else None

        if not ensure_tabbyapi():
            sys.exit(1)
        config, personas_data, Orchestrator = _load_core()
        if skip_research:
            config = copy.deepcopy(config)
            config["research"]["enabled"] = False
        print(f"\n  {GRN}Script: {script_path.name} — {topic}{R}\n")
        orch = Orchestrator(config=config, personas=personas_data,
                            us_persona=us_p, china_persona=cn_p, eu_persona=eu_p)
        orch.run_debate(topic=topic, max_rounds=max_rounds,
                        time_limit_minutes=time_limit, produce_podcast=podcast)
    else:
        main()
