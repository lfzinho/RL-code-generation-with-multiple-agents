from typing import List, Tuple
from pydantic import BaseModel
from rl.agent import Agent
from rl.policies import Policy
from rl.environment import Environment
from rl.llm import LLM


class LLMAgent:
    def __init__(self, environment: Environment, prompts: List[str], initial_value: int, policy: Policy, name: str):
        self.environment = environment
        self.prompts = prompts
        self.agent = Agent(len(prompts), initial_value, policy)
        self.name = name

        self.llm = LLM()
        self.last_action = None

    def add_message(self) -> Tuple[str, int]:
        prompt_idx = self.agent.get_action()
        prompt = self.prompts[prompt_idx]
        self.last_action = prompt_idx
        self.environment.set_prompt(prompt)
        message = self.llm.generate_text(self.environment)
        message = self.mark_name_on_message(message)
        self.environment.add_message(message, self.name)

    def reward(self, reward: int):
        self.agent.update(self.last_action, reward)
    
    def mark_name_on_message(self, message: dict):
        message['content'] = f"Sent by {self.name}: \n\n{message['content']}"
        return message
