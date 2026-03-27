"""
Orchestrator — the main debate loop.

Flow:
  Phase 1: Research & initial proposals (US → China)
  Phase 2: Iterative debate loop (Judge questions → US → China → score → repeat)
  Phase 3: Final verdict (Judge)
  Phase 4: Podcast production (optional)
"""

import time
import yaml
from pathlib import Path
from typing import Optional

from src.tabby_client import TabbyClient
from src.context.debate_state import DebateState
from src.context.context_manager import ContextManager
from src.agents.us_agent import USAgent
from src.agents.china_agent import ChinaAgent
from src.agents.eu_judge import EUJudge
from src.research.web_search import search_and_extract, format_research_for_prompt
from src.evaluation.repetition_detector import RepetitionDetector
from src.evaluation.quality_scorer import QualityScorer
from src.rag.retriever import RAGRetriever
from src.tts.podcast import PodcastProducer


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_personas(personas_path: str = "config/personas.yaml") -> dict:
    with open(personas_path) as f:
        return yaml.safe_load(f)


class Orchestrator:
    def __init__(
        self,
        config: dict,
        personas: dict,
        us_persona: str = None,
        china_persona: str = None,
    ):
        self.config = config
        self.personas = personas

        # TabbyAPI client
        tabby_cfg = config["tabbyapi"]
        self.client = TabbyClient(
            base_url=tabby_cfg["url"],
            api_key=tabby_cfg.get("api_key", ""),
            timeout=tabby_cfg.get("timeout", 120),
        )

        self.debate_cfg = config["debate"]
        self.research_cfg = config["research"]
        self.model_cfgs = config["models"]

        # Will be initialized per debate
        self._us_agent: Optional[USAgent] = None
        self._china_agent: Optional[ChinaAgent] = None
        self._judge: Optional[EUJudge] = None
        self._us_persona = us_persona
        self._china_persona = china_persona

        self._ctx_manager: Optional[ContextManager] = None
        self._repetition = RepetitionDetector(
            threshold=self.debate_cfg["repetition_threshold"],
            max_consecutive=self.debate_cfg["repetition_max_consecutive"],
        )
        self._quality = QualityScorer()
        self._rag = RAGRetriever()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_debate(
        self,
        topic: str,
        max_rounds: int = None,
        time_limit_minutes: int = None,
        produce_podcast: bool = False,
        voice_refs: dict = None,
    ) -> DebateState:
        max_rounds = max_rounds or self.debate_cfg["default_max_rounds"]
        time_limit = (time_limit_minutes or self.debate_cfg["default_time_limit_minutes"]) * 60

        print(f"\n{'='*60}")
        print(f"DEBATE: {topic}")
        print(f"Max rounds: {max_rounds} | Time limit: {time_limit//60} min")
        print(f"{'='*60}\n")

        # Verify TabbyAPI is reachable
        if not self.client.is_alive():
            raise RuntimeError(
                f"TabbyAPI not reachable at {self.client.base_url}. "
                "Start TabbyAPI first: cd tabbyAPI && python3 main.py"
            )

        # Initialize agents and state
        state = DebateState(topic=topic)
        self._init_agents(topic)
        self._ctx_manager = ContextManager(self.config, self.client)
        self._repetition.reset()

        start_time = time.time()

        # ------ Phase 1: Research & Initial Proposals ------------------
        print("\n--- Phase 1: Research & Initial Proposals ---\n")
        self._phase_research(state)

        # ------ Phase 2: Debate Loop -----------------------------------
        print("\n--- Phase 2: Debate Loop ---\n")
        for round_num in range(1, max_rounds + 1):
            elapsed = time.time() - start_time
            if elapsed >= time_limit * 0.9:  # 90% of time → stop loop
                state.finish_reason = "time_limit"
                print(f"  [loop] time limit approaching ({elapsed/60:.1f} min), exiting loop")
                break

            state.round_num = round_num
            print(f"\n  === Round {round_num} ===")

            # Judge asks a targeted question
            question = self._judge_question_turn(state)

            # US responds
            self._debater_turn(
                agent=self._us_agent,
                agent_key="us",
                state=state,
                turn_objective=self._us_agent.get_turn_objective(round_num),
                extra=self._devil_advocate_injection(round_num) if self._should_inject_devil(round_num, state) else "",
            )

            # China responds
            self._debater_turn(
                agent=self._china_agent,
                agent_key="china",
                state=state,
                turn_objective=self._china_agent.get_turn_objective(round_num),
                extra=self._devil_advocate_injection(round_num) if self._should_inject_devil(round_num, state) else "",
            )

            # Post-round: summarize, score, check termination
            self._ctx_manager.maybe_summarize(state)
            self._score_round(state, round_num)

            # Add to RAG if enabled (10+ rounds)
            if round_num >= 10:
                for t in state.get_turns_for_round(round_num):
                    self._rag.add_turn(t.agent, round_num, t.content)

            # Check early termination
            if self._repetition.should_terminate():
                state.finish_reason = "stagnation"
                print("  [loop] repetition threshold reached — ending debate early")
                break
            if self._quality.should_terminate():
                state.finish_reason = "stagnation"
                print("  [loop] quality stagnation detected — ending debate early")
                break
        else:
            state.finish_reason = "rounds_exhausted"

        state.finished = True

        # ------ Phase 3: Verdict ---------------------------------------
        print("\n--- Phase 3: Final Verdict ---\n")
        self._verdict_turn(state)

        # ------ Phase 4: Podcast (optional) ----------------------------
        if produce_podcast:
            print("\n--- Phase 4: Podcast Production ---\n")
            producer = PodcastProducer(self.config, self.personas)
            self._load_judge_model()
            producer.produce(state, self.client, voice_refs)

        # Save outputs
        if self.config["output"]["save_debate_state"]:
            state.save(self.config["output"]["transcripts_dir"])

        print(f"\n{'='*60}")
        print(f"Debate complete. Reason: {state.finish_reason}")
        print(f"Total rounds: {state.round_num}")
        print(f"Total time: {(time.time()-start_time)/60:.1f} min")
        if self._quality.average_scores():
            print(f"Avg quality: {self._quality.average_scores()}")
        print(f"{'='*60}\n")

        return state

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_research(self, state: DebateState) -> None:
        """Phase 1: Each debater researches and stakes out an opening position."""
        if not self.research_cfg["enabled"]:
            print("  [research] disabled in config, skipping")
            return

        for agent_key, agent in [("us", self._us_agent), ("china", self._china_agent)]:
            model_cfg = self.model_cfgs[agent_key]
            self._swap(agent_key)

            print(f"  [{agent_key}] researching...", flush=True)
            research_results = self._run_research(state.topic, agent)

            objective = agent.get_turn_objective(0)
            research_block = format_research_for_prompt(research_results)

            system = agent.build_system_prompt()
            if research_block:
                system += f"\n\n{research_block}"

            messages = self._ctx_manager.build_debater_turn_messages(
                system_prompt=system,
                state=state,
                turn_objective=f"Opening position: {objective}",
            )

            raw = self.client.chat(
                messages,
                temperature=model_cfg["temperature"],
                top_p=model_cfg.get("top_p", 0.95),
                max_tokens=model_cfg["max_tokens"],
            )

            argument, thinking = agent.parse_response(raw)
            turn = state.add_turn(agent_key, argument, thinking)
            print(f"  [{agent_key}] opening: {argument[:120]}...", flush=True)

            # Extract claims
            claims, evidence = agent.extract_claims(argument, self.client)
            for c in claims:
                state.add_claim(agent_key, c)
            for e in evidence:
                state.add_evidence(agent_key, e)

    def _debater_turn(
        self,
        agent,
        agent_key: str,
        state: DebateState,
        turn_objective: str,
        extra: str = "",
    ) -> None:
        model_cfg = self.model_cfgs[agent_key]
        self._swap(agent_key)

        system = agent.build_system_prompt()
        messages = self._ctx_manager.build_debater_turn_messages(
            system_prompt=system,
            state=state,
            turn_objective=turn_objective,
            extra_instruction=extra,
        )

        raw = self.client.chat(
            messages,
            temperature=model_cfg["temperature"],
            top_p=model_cfg.get("top_p", 0.95),
            max_tokens=model_cfg["max_tokens"],
        )

        argument, thinking = agent.parse_response(raw)
        turn = state.add_turn(agent_key, argument, thinking)

        # Repetition check
        is_rep, sim = self._repetition.check(agent_key, argument)
        turn.is_repetitive = is_rep
        flag = " [REPETITIVE]" if is_rep else ""
        print(f"  [{agent_key}] r{state.round_num}: {argument[:100]}...{flag}", flush=True)

        # Extract claims
        claims, evidence = agent.extract_claims(argument, self.client)
        for c in claims:
            state.add_claim(agent_key, c)
        for e in evidence:
            state.add_evidence(agent_key, e)

    def _judge_question_turn(self, state: DebateState) -> str:
        """Judge generates a targeted question for the weakest argument."""
        self._load_judge_model()
        messages = self._judge.build_question_messages(state)

        raw = self.client.chat(
            messages,
            temperature=self.model_cfgs["judge"]["temperature"],
            top_p=self.model_cfgs["judge"].get("top_p", 0.9),
            max_tokens=300,
        )
        question, thinking = self._judge.parse_question(raw)
        state.add_turn("judge", question, thinking)
        print(f"  [judge] question: {question[:120]}...", flush=True)
        return question

    def _verdict_turn(self, state: DebateState) -> None:
        """Judge delivers the final structured verdict."""
        self._load_judge_model()
        messages = self._judge.build_verdict_messages(state)

        raw = self.client.chat(
            messages,
            temperature=self.model_cfgs["judge"]["temperature"],
            top_p=self.model_cfgs["judge"].get("top_p", 0.9),
            max_tokens=self.model_cfgs["judge"]["max_tokens"],
        )
        verdict, thinking = self._judge.parse_verdict(raw)
        state.verdict = verdict
        state.add_turn("judge", verdict, thinking)
        print(f"\n  [verdict]\n{verdict[:300]}...", flush=True)

    def _score_round(self, state: DebateState, round_num: int) -> None:
        """Score the current round using the judge model."""
        self._load_judge_model()
        turns = state.get_turns_for_round(round_num)
        scores = self._quality.score_round(turns, self.client)
        for turn in turns:
            if turn.agent in scores:
                turn.quality_score = scores[turn.agent]
        if scores:
            print(f"  [quality] r{round_num}: {scores}", flush=True)

    # ------------------------------------------------------------------
    # Model swap helpers
    # ------------------------------------------------------------------

    def _swap(self, agent_key: str) -> None:
        model_cfg = self.model_cfgs[agent_key]
        self.client.swap_model(
            model_name=model_cfg["name"],
            model_path=model_cfg["path"],
            max_seq_len=model_cfg["max_seq_len"],
        )

    def _load_judge_model(self) -> None:
        self._swap("judge")

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def _run_research(self, topic: str, agent) -> list[dict]:
        """Generate search queries and fetch results for an agent."""
        # Build query from topic + agent perspective
        perspective = "US" if hasattr(agent, "agent_key") and agent.agent_key == "us" else "Chinese"
        queries = [
            f"{topic} {perspective} perspective 2025 2026",
            f"{topic} latest developments evidence",
        ]
        results = []
        for query in queries[:2]:
            found = search_and_extract(
                query=query,
                max_results=self.research_cfg["max_results_per_query"],
                max_chars=self.research_cfg["max_content_chars"],
                tavily_api_key=self.research_cfg.get("tavily_api_key", ""),
                jina_enabled=self.research_cfg.get("jina_enabled", True),
            )
            results.extend(found)
        return results[:self.research_cfg["max_results_per_query"]]

    # ------------------------------------------------------------------
    # Anti-collapse helpers
    # ------------------------------------------------------------------

    def _should_inject_devil(self, round_num: int, state: DebateState) -> bool:
        interval = self.debate_cfg.get("devil_advocate_every_n", 4)
        if round_num % interval != 0:
            return False
        # Only inject if agreement is actually converging
        return len(state.points_of_agreement) > len(state.points_of_contention)

    @staticmethod
    def _devil_advocate_injection(round_num: int) -> str:
        return (
            "SPECIAL INSTRUCTION: The debate has been converging. You MUST introduce "
            "a controversial counterpoint or reframe that challenges the emerging consensus. "
            "Identify an angle IGNORED by both sides and make it central to your argument."
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_agents(self, topic: str) -> None:
        self._us_agent = USAgent(self.personas, self.model_cfgs["us"], topic, self._us_persona)
        self._china_agent = ChinaAgent(self.personas, self.model_cfgs["china"], topic, self._china_persona)
        self._judge = EUJudge(self.personas, self.model_cfgs["judge"], topic)
