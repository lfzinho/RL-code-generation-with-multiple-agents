import numpy as np
import instructor
from openai import OpenAI
from instructor.exceptions import InstructorRetryException
from pydantic import BaseModel, Field
from rl.environment import Environment


class CodeEvaluation(BaseModel):
    correctness_grade: int = Field(..., description="A grade from 0 to 100 for the correctness of the code.")
    explanation_correctness: str = Field(..., description="A brief explanation of the correctness grade.")
    readability_grade: int = Field(..., description="A grade from 0 to 100 for the readability of the code.")
    explanation_readability: str = Field(..., description="A brief explanation of the readability grade.")

    def get_mean_grade(self) -> int:
        return np.mean([self.correctness_grade, self.readability_grade])
    
    def get_answer(self) -> str:
        return f"""**Correctness:** {self.correctness_grade}
\n**Grade Explanation:** {self.explanation_correctness}

**Readability:** {self.readability_grade}
\n**Grade Explanation:** {self.explanation_readability}"""


class CodeEvaluator:
    default_model = "gemma2:2b"
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        mode=instructor.Mode.JSON,
    )
    max_retries = 3

    def __init__(self, environment: Environment, prompt: str, name: str = "Code Evaluator"):
        self.environment = environment
        self.prompt = prompt
        self.name = name
    
    def evaluate_code(self):
        last_code_msg = self.environment.get_last_message(owner="Coder")
        reduced_environment = [{"role": "User", "content": self.prompt + "\n" + last_code_msg['content']}]
        reward, message = self.call_instructor_for_evaluation(reduced_environment)
        message = self.mark_name_on_message(message)
        self.environment.add_message(message, self.name)
        return reward
    
    def mark_name_on_message(self, message: dict):
        message['content'] = f"Sent by {self.name}: \n\n{message['content']}"
        return message

    def call_instructor_for_evaluation(self, messages: list) -> CodeEvaluation:
        response = self.client.chat.completions.create(
            model=self.default_model,
            messages=messages,
            response_model=CodeEvaluation,
            max_retries=self.max_retries,
        )
        return response.get_mean_grade(), {'role':'assistant','content':response.get_answer()}
