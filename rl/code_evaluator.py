from rl.environment import Environment, Message
from rl.llm import LLM


class CodeEvaluator:
    def __init__(self, environment: Environment, prompt: str, name: str = "Code Evaluator"):
        self.environment = environment
        self.prompt = prompt
        self.name = name

        self.llm = LLM()
    
    def evaluate_code(self):
        evaluation = self.llm.evaluate_code(self.environment, self.prompt)
        reward = self.extract_numerical_evaluation(evaluation)
        self.environment.add_message(self.name, evaluation)
        return reward
    
    def extract_numerical_evaluation(self, evaluation: str) -> int:
        return int(evaluation.split()[-1])
