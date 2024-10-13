from typing import Dict, List, OrderedDict


class Environment:
    def __init__(self):
        self.prompt = None
        self.messages = []
        self.messages_owners = []

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def get_state(self) -> List[Dict]:
        state = self.messages.copy()
        state.append(
            {
                "role": "User",
                "content": self.prompt
            }
        )
        return state

    def get_last_message(self, owner: str = None) -> Dict:
        if owner is None:
            return self.messages[-1]
        for i, message_owner in enumerate(reversed(self.messages_owners)):
            if message_owner == owner:
                return self.messages[-(i + 1)]
        return None

    def add_message(self, message: Dict, agent_name: str):
        self.messages.append(message)
        self.messages_owners.append(agent_name)
