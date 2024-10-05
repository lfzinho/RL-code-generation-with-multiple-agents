import sys
import instructor
import numpy as np
from instructor.exceptions import InstructorRetryException
from openai import OpenAI
from pydantic import BaseModel, Field
from rl.environment import Environment
from IPython.display import display, Markdown


class CodeEvaluation(BaseModel):
    correctness_grade: int = Field(..., description="A grade from 0 to 100 that represents the correctness of the code")
    readability_grade: int = Field(..., description="A grade from 0 to 100 that represents the readability of the code")
    reason: str = Field(..., description="A reason for the grades given")

    def get_min_grade(self) -> int:
        return min(self.correctness_grade, self.readability_grade)
    
    def get_answer(self) -> str:
        return f"Correctness: {self.correctness_grade}, Readability: {self.readability_grade} - {self.reason}"

class LLM:
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        mode=instructor.Mode.JSON,
    )
    max_retries = 5

    @classmethod
    def generate_text(cls, environment: Environment, role: str, response_model: BaseModel) -> str:
        response = None
        retries = 0
        while response is None:
            try:
                response = cls.client.chat.completions.create(
                    model="gemma2:2b",
                    messages=environment.get_state(),
                    response_model=response_model,
                    max_retries=0,
                    strict=False
                )
                return response.get_answer()
            except InstructorRetryException as e:
                retries += 1
                print(f"Retrying... ({retries}/{cls.max_retries})", end="\r")
                if retries >= cls.max_retries:
                    print("Max retries reached. Exiting.")
                    for message in e.messages:
                        display(Markdown(f"**{message['role']}**: {message['content']}"))
                    sys.exit(1)

    
    @classmethod
    def evaluate_code(cls, environment: Environment, prompt: str) -> int:
        last_code_msg = environment.get_last_message(from_coder=True)

        response = cls.client.chat.completions.create(
            model="gemma2:2b",
            messages=[{"role": "user", "content": prompt}, {"role": "Coder", "content": last_code_msg['content']}],
            response_model=CodeEvaluation,
        )
        return response.get_min_grade(), response.get_answer()