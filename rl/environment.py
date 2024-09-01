from typing import Dict, List, OrderedDict


class Message:
    def __init__(self, agent_name: str, content: str):
        self.agent_name = agent_name
        self.content = content
    
    def __str__(self):
        return f"{self.agent_name}: {self.content}"


class Environment:
    def __init__(self, messages: List[Message] = []):
        self.prompt = None
        self.messages = messages
    
    def set_prompt(self, prompt: str):
        self.prompt = prompt
    
    def get_state(self) -> str:
        state = f"Prompt: {self.prompt}\n"
        for message in self.messages:
            state += f"{message}\n"
        return state
    
    def get_last_message(self, from_coder=False, from_reviewer=False) -> Message:
        if from_coder:
            for message in reversed(self.messages):
                if "Coder" in message.agent_name:
                    return message
            return None
        elif from_reviewer:
            for message in reversed(self.messages):
                if "Reviewer" in message.agent_name:
                    return message
            return None      
        else:  
            return self.messages[-1]
    
    def add_message(self, agent_name: str, message: str):
        self.messages.append(Message(agent_name, message))
    
    def print_messages(self):
        for message in self.messages:
            print(message)

