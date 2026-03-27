"""US Delegation agent."""

from src.agents.base_agent import BaseAgent


class USAgent(BaseAgent):
    role = "debater"
    agent_key = "us"

    def __init__(self, personas: dict, model_cfg: dict, topic: str, persona_name: str = None):
        persona_name = persona_name or personas["us"]["default_persona"]
        persona = personas["us"]["personas"][persona_name]
        super().__init__(persona, model_cfg, topic)
        self.agent_name = persona_name

    def get_turn_objective(self, round_num: int, opponent_last_turn=None) -> str:
        if opponent_last_turn is None:
            return (
                "Present the opening US position on the debate topic. "
                "Cite specific data, programs, or agreements. "
                "Identify the strongest argument for the US perspective."
            )
        return (
            f"Round {round_num}: Identify the single weakest point in China's last argument "
            "and attack it with specific evidence. Assert the US position forcefully. "
            "Do not merely rebut — advance your own argument."
        )
