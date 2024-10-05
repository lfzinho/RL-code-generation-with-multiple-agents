from typing import Dict, List, OrderedDict


class Environment:
    def __init__(self, messages: List[Dict] = []):
        self.prompt = None
        self.messages = messages

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def get_state(self) -> List[Dict]:
        state = self.messages.copy()
        state.append(
            {
                "role": "System",
                "content": self.prompt
            }
        )
        return state

    def get_last_message(self, from_coder=False, from_reviewer=False) -> Dict:
        if from_coder:
            for message in reversed(self.messages):
                if "Coder" in message['role']:
                    return message
            return None
        elif from_reviewer:
            for message in reversed(self.messages):
                if "Reviewer" in message['role']:
                    return message
            return None      
        else:  
            return self.messages[-1]

    def add_message(self, role: str, message: str):
        self.messages.append(
            {
                "role": role,
                "content": message
            }
        )

    def print_messages(self):
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
