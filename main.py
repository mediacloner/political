#!/usr/bin/env python3
"""
Politics AI Swarm — CLI entrypoint.

Usage examples:
  python main.py --topic "Should ESA partner with NASA or CNSA for a lunar base?"
  python main.py --topic "..." --rounds 6 --time-limit 30
  python main.py --topic "..." --us-persona lobbyist --china-persona economist
  python main.py --topic "..." --podcast
  python main.py --check   # Verify TabbyAPI is reachable
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator, load_config, load_personas


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-agent geopolitical debate system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--topic", "-t",
        required=False,
        help="Debate topic (required unless --check)",
    )
    p.add_argument(
        "--rounds", "-r",
        type=int,
        default=None,
        help="Maximum debate rounds (default from config)",
    )
    p.add_argument(
        "--time-limit",
        type=int,
        default=None,
        metavar="MINUTES",
        help="Time limit in minutes (default from config)",
    )
    p.add_argument(
        "--us-persona",
        choices=["strategist", "lobbyist", "analyst"],
        default=None,
        help="US delegation persona (default: strategist)",
    )
    p.add_argument(
        "--china-persona",
        choices=["director", "enterprise_rep", "economist"],
        default=None,
        help="China delegation persona (default: director)",
    )
    p.add_argument(
        "--eu-persona",
        choices=["strategist", "hawk", "economist"],
        default=None,
        help="EU delegation persona (default: strategist)",
    )
    p.add_argument(
        "--podcast",
        action="store_true",
        help="Produce audio podcast after debate (requires Fish Speech)",
    )
    p.add_argument(
        "--voice-us",
        default=None,
        metavar="WAV",
        help="Path to US voice reference clip (.wav, 10-30s)",
    )
    p.add_argument(
        "--voice-china",
        default=None,
        metavar="WAV",
        help="Path to China voice reference clip",
    )
    p.add_argument(
        "--voice-judge",
        default=None,
        metavar="WAV",
        help="Path to EU judge voice reference clip",
    )
    p.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Check that TabbyAPI is reachable and exit",
    )
    return p.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    personas = load_personas("config/personas.yaml")

    if args.check:
        from src.tabby_client import TabbyClient
        client = TabbyClient(
            base_url=config["tabbyapi"]["url"],
            api_key=config["tabbyapi"].get("api_key", ""),
        )
        if client.is_alive():
            print(f"TabbyAPI is reachable at {config['tabbyapi']['url']}")
            sys.exit(0)
        else:
            print(f"TabbyAPI NOT reachable at {config['tabbyapi']['url']}")
            print("Start it with: cd tabbyAPI && python3 main.py")
            sys.exit(1)

    if not args.topic:
        print("Error: --topic is required")
        sys.exit(1)

    voice_refs = {}
    if args.voice_us:
        voice_refs["us"] = args.voice_us
    if args.voice_china:
        voice_refs["china"] = args.voice_china
    if args.voice_judge:
        voice_refs["judge"] = args.voice_judge

    orchestrator = Orchestrator(
        config=config,
        personas=personas,
        us_persona=args.us_persona,
        china_persona=args.china_persona,
        eu_persona=args.eu_persona,
    )

    state = orchestrator.run_debate(
        topic=args.topic,
        max_rounds=args.rounds,
        time_limit_minutes=args.time_limit,
        produce_podcast=args.podcast,
        voice_refs=voice_refs if voice_refs else None,
    )


if __name__ == "__main__":
    main()
