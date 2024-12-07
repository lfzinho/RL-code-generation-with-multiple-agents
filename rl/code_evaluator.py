import numpy as np
import pandas as pd
import instructor
from openai import OpenAI
from instructor.exceptions import InstructorRetryException
from pydantic import BaseModel, Field
from rl.environment import Environment

CSV_PATH = "csv_data/imdb_sample_10.csv"

class CodeEvaluation(BaseModel):
    is_code_functional: bool = Field(..., description="A boolean indicating if the code is functional.")
    is_code_consise: bool = Field(..., description="A boolean indicating if the code is consise.")
    is_code_easily_readable: bool = Field(..., description="A boolean indicating if the code is easily readable.")
    is_code_documented: bool = Field(..., description="A boolean indicating if the code is documented.")
    is_csv_path_correct: bool = Field(..., description=f"A boolean indicating if the csv path is correct. The correct path is '{CSV_PATH}'.")
    is_code_all_grouped: bool = Field(..., description="A boolean indicating if the code is all grouped. If the code is split into multiple cells, this should be False.")
    is_code_saving_csv: bool = Field(..., description="A boolean indicating if the code is saving the csv file.")
    overall_grade: int = Field(..., description="A score from 0 to 100 indicating the overall code quality. This should consider the boolean values just as well as other factors not covered by them.")
    explanation: str = Field(..., description="A summary of the evaluation. Why are the scores true or false? What can be improved?")

    def get_mean_grade(self) -> int:
        return (np.mean([
            self.is_code_functional,
            self.is_code_consise,
            self.is_code_easily_readable,
            self.is_code_documented,
            self.is_csv_path_correct,
            self.is_code_all_grouped,
            self.is_code_saving_csv,
        ]) + self.overall_grade/100) / 2 * 100
    
    def get_answer(self) -> str:
        return f"""**Code Functional**: {self.is_code_functional}
**Code Consise**: {self.is_code_consise}
**Code Easily Readable**: {self.is_code_easily_readable}
**Code Documented**: {self.is_code_documented}
**CSV Path Correct**: {self.is_csv_path_correct}
**Code All Grouped**: {self.is_code_all_grouped}
**Code Saving CSV**: {self.is_code_saving_csv}

**Overall Grade**: {self.overall_grade}
**Explanation**: {self.explanation}"""
    

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

    def evaluate_code(self) -> int:
        # Cleaning assessment
        #cleanliness_score = self.evaluate_cleanliness(csv_path)

        # Code Quality Assessment Using LLM
        last_code_msg = self.environment.get_last_message(owner="Coder")
        reduced_environment = [{"role": "User", "content": self.prompt + "\n" + last_code_msg['content']}]
        rewards, message = self.call_instructor_for_evaluation(reduced_environment)

        # Adds feedback to the environment
        message = self.mark_name_on_message(message)
        self.environment.add_message(message, self.name)

        return rewards

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
