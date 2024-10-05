from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

import instructor


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


class CodeResponse(BaseModel):
    python_code: str
    

# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

resp = client.chat.completions.create(
    model="gemma2:2b",
    messages=[{'role': 'System', 'content': 'Give the worst python solution for this problem'}, {'role': 'User', 'content': 'Write a piece of code for exemplifying the dijkstra algorithm.'}],
    response_model=CodeResponse,
)
print(resp.model_dump_json(indent=2))