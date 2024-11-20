from rl.environment import Environment
from rl.llm import LLM
from IPython.display import display, Markdown


class Coder:
    """A class for keeping track of average conversation starting and ending 
    rewards for each prompt.
    """
    def __init__(self, prompts_props: list[dict], environment: Environment = None):
        self.name = "Coder"
        self.environment = environment
        self.prompts = prompts_props
        self.start_rewards = {prompt_props['index']: [] for prompt_props in prompts_props}
        self.end_rewards = {prompt_props['index']: [] for prompt_props in prompts_props}
        self.current_prompt = None

        self.llm = LLM()
    
    def add_message(self, prompt: dict):
        """Add a message to the conversation and update the current prompt.
        
        Parameters
        ----------
        message : str
            The message to be added to the conversation
        """
        self.current_prompt = prompt['index']
        self.environment.set_prompt(prompt['prompt'])
        message = self.llm.generate_text(self.environment)
        # Register the initial message
        initial_prompt_message = {
            "role": "User",
            "content": prompt['prompt']
        }
        display(Markdown(f"**Initial prompt**: {prompt['prompt']}"))
        self.environment.add_message(initial_prompt_message, 'User')
        # Register the response from the coder
        display(Markdown(f"**Coder**: {message['content']}"))
        message = self.mark_name_on_message(message)
        self.environment.add_message(message, self.name)

    def mark_name_on_message(self, message: dict):
        message['content'] = f"Sent by {self.name}: \n\n{message['content']}"
        return message
    
    def first_reward(self, reward: int):
        """Register the reward for the first message in the conversation.
        
        Parameters
        ----------
        reward : int
            The reward to be registered for the first message
        """
        self.start_rewards[self.current_prompt].append(reward)
    
    def final_reward(self, reward: int):
        """Register the reward for the last message in the conversation.
        
        Parameters
        ----------
        reward : int
            The reward to be registered for the last message
        """
        self.end_rewards[self.current_prompt].append(reward)