from typing import List, Dict, Any, Tuple
from rl.agent import Agent
from rl.policies import Policy
from rl.environment import Environment
from rl.llm import LLM


class LLMAgent:
    def __init__(self, prompts: List[str], initial_value: int, policy: Policy, name: str):
        self.prompts = prompts
        self.agent = Agent(len(prompts), initial_value, policy)
        self.name = name

        self.llm = LLM()
        self.last_action = None

    def add_message(self, environment: Environment) -> Tuple[str, int]:
        prompt_idx = self.agent.get_action()
        prompt = self.prompts[prompt_idx]
        self.last_action = prompt_idx
        environment.set_prompt(prompt)
        code = self.llm.generate_text(environment)
        environment.add_message(self.name, code)

    def reward(self, reward: int):
        self.agent.update(self.last_action, reward)
