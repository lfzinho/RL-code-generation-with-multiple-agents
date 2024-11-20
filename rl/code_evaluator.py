import numpy as np
import pandas as pd
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

    def __init__(self, environment: Environment, prompt: str, name: str = "Code Evaluator", clean_csv_path: str = ""):
        self.environment = environment
        self.prompt = prompt
        self.name = name
        self.clean_csv_path = clean_csv_path 

    def evaluate_code(self, csv_path: str) -> int:
        # Cleaning assessment
        cleanliness_score = self.evaluate_cleanliness(csv_path)

        # Code Quality Assessment Using LLM
        last_code_msg = self.environment.get_last_message(owner="Coder")
        reduced_environment = [{"role": "User", "content": self.prompt + "\n" + last_code_msg['content']}]
        reward, message = self.call_instructor_for_evaluation(reduced_environment)

        # Adds feedback to the environment
        message = self.mark_name_on_message(message)
        self.environment.add_message(message, self.name)

        return cleanliness_score

    def evaluate_cleanliness(self, csv_path: str) -> int:
        df_dirty = pd.read_csv(csv_path)
        df_clean = pd.read_csv(self.clean_csv_path)
        
        score = 0
        
        # Checks for NaNs in modified CSV
        if not df_dirty.isna().any().any():
            score += 1 
        
        # Checks that there are no empty cells in the modified CSV
        if not (df_dirty == "").any().any():
            score += 1 
        
        # Checks if column types are consistent with the clean CSV
        if all(df_dirty.dtypes == df_clean.dtypes):
            score += 1 
        
        return score

    def mark_name_on_message(self, message: dict):
        message['content'] = f"Sent by {self.name}: \n\n{message['content']}"
        return message

    def call_instructor_for_evaluation(self, messages: list) -> tuple[int, dict]:
        response = self.client.chat.completions.create(
            model=self.default_model,
            messages=messages,
            response_model=CodeEvaluation,
            max_retries=self.max_retries,
        )
        return response.get_mean_grade(), {'role': 'assistant', 'content': response.get_answer()}    
