"""China Delegation agent."""

from src.agents.base_agent import BaseAgent


class ChinaAgent(BaseAgent):
    role = "debater"
    agent_key = "china"

    def __init__(self, personas: dict, model_cfg: dict, topic: str, persona_name: str = None):
        persona_name = persona_name or personas["china"]["default_persona"]
        persona = personas["china"]["personas"][persona_name]
        super().__init__(persona, model_cfg, topic)
        self.agent_name = persona_name

    def get_turn_objective(self, round_num: int, opponent_last_turn=None) -> str:
        if opponent_last_turn is None:
            return (
                "Present the opening Chinese position on the debate topic. "
                "Emphasize concrete execution capabilities, cost-effectiveness, and non-interference principles. "
                "Identify the strongest argument for the Chinese perspective."
            )
        return (
            f"Round {round_num}: Identify the single weakest assumption in the US argument "
            "and dismantle it using first-principles logic and specific evidence. "
            "Counter with a distinct Chinese alternative that the US cannot easily match."
        )
