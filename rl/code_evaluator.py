from rl.environment import Environment
from rl.llm import LLM


class CodeEvaluator:
    def __init__(self, environment: Environment, prompt: str, name: str = "Code Evaluator"):
        self.environment = environment
        self.prompt = prompt
        self.name = name

        self.llm = LLM()
    
    def evaluate_code(self):
        reward, message = self.llm.evaluate_code(self.environment, self.prompt)
        self.environment.add_message(self.name, message)
        return reward
