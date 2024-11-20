import ollama
from rl.environment import Environment

import openai

openai.base_url = "http://localhost:11434/v1"
openai.api_key = 'ollama'

class LLM:
    default_model = "gemma2:2b"

    @classmethod
    def generate_text(cls, environment: Environment) -> dict:
        total_environment = environment.get_state()
        return cls.generate_llm_text(total_environment)
    
    @classmethod
    def generate_llm_text(cls, messages: list) -> dict:
        return ollama.chat(
            model=cls.default_model,
            messages=messages,
        )['message']
        