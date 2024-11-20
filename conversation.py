import tqdm
import pandas as pd
from coder import Coder
from rl.llm_agent import LLMAgent
from rl.environment import Environment
from rl.code_evaluator import CodeEvaluator
from rl.policies import EpsilonGreedyPolicy
from rl.utils import compute_delta_grade, is_terminate_grade
from csv_tools.csv_changer import change_csv

CSV_PATH = "csv_tools/data.csv"
csv_data = pd.read_csv(CSV_PATH).to_string()

def_env = Environment()
evaluator = CodeEvaluator(
    environment=def_env,
    prompt="Evaluate the python code bellow: give grades from 0 to 100 in reliability and clarity. Briefly explain your grades.",
    name="Code Evaluator"
)


def create_coder(prompts_props: list[dict]) -> Coder:
    """Create a coder agent.

    Keep in mind that the coder agent is not an RL agent, but a simple holder for the rewards given
    the initial prompt and the conversation.
    """
    global def_env
    return Coder(prompts_props, def_env)

def create_reviewer(prompts: list[str]) -> LLMAgent:
    """Create a reviewer agent.
    
    Parameters
    ----------
    prompts : list[str]
        List of prompts to be used by the reviewer agent.

    Returns
    -------
    LLMAgent
        The reviewer agent that will be used to evaluate the conversation
    """
    global def_env
    return LLMAgent(
        environment=def_env,
        prompts=prompts,
        initial_value=100,
        policy=EpsilonGreedyPolicy(0.1),
        name="Reviewer"
    )

def create_refiner(prompts: list[str]) -> LLMAgent:
    """Create a refiner agent.

    Parameters
    ----------
    prompts : list[str]
        List of prompts to be used by the refiner agent.

    Returns
    -------
    LLMAgent
        The refiner agent that will be used to refine the conversation
    """
    global def_env
    return LLMAgent(
        environment=def_env,
        prompts=prompts,
        initial_value=100,
        policy=EpsilonGreedyPolicy(0.1),
        name="Code Refiner"
    )

def start_conversation(
        coder: Coder, 
        coder_prompt: dict, 
        reviewer: LLMAgent, 
        refiner: LLMAgent, 
        max_turns: int = 5,
        csv_path: str = "", 
        clean_csv_path: str = ""
    ) -> Environment:
    """Start a conversation between the coder, reviewer, refiner and evaluator.

    Parameters
    ----------
    coder : Coder
        The coder agent that will be used to generate the initial code.
    coder_prompt : str
        The prompt to be used by the coder agent.
    reviewer : LLMAgent
        The reviewer agent that will be used to evaluate the conversation.
    refiner : LLMAgent
        The refiner agent that will be used to refine the conversation.
    max_turns : int, optional
        The maximum number of turns that the conversation will last, by default 5.

    Returns
    -------
    Environment
        The final environment after the conversation.
    """
    global evaluator, csv_data
    environment = Environment()
    
    # Set the clean CSV path in the evaluator
    evaluator.clean_csv_path = clean_csv_path
    
    # Set the environment for all agents
    for agent in [coder, reviewer, refiner, evaluator]:
        agent.environment = environment
    
    # Adding csv in the first coder prompt
    coder_prompt["prompt"] += f"\n\nHere is the initial CSV content:\n{csv_data}"
    coder.add_message(coder_prompt)
    
    # Start the conversation
    last_grade = None
    for turn in tqdm.tqdm(range(max_turns), desc="Conv. turns", position=1, leave=False):
        # Evaluates the code and status of the CSV
        grade = evaluator.evaluate_code(csv_path)
        
        # Check if the score is enough to close
        if grade >= 3:
            print("CSV limpo! Terminando conversa.")
            break
        # If it is the first turn, reward the coder
        elif last_grade is None:
            coder.first_reward(grade)
        # If it is not the first turn, reward refiner and reviewer    
        else:
            delta_grade = compute_delta_grade(last_grade, grade)
            refiner.reward(delta_grade)
            reviewer.reward(delta_grade)
        
        # Update the CSV with the generated code
        last_code_msg = environment.get_last_message(owner="Coder")["content"]
        change_csv(last_code_msg, csv_path)
        
        # Reads the updated CSV and adds it to the next prompts for Reviewer and Refiner
        csv_data = pd.read_csv(csv_path).to_string()
        next_prompt = f"\n\nHere is the updated CSV content:\n{csv_data}"
        reviewer.add_message(next_prompt)
        refiner.add_message(next_prompt)

        last_grade = grade

    # Reward the Coder with the latest review
    coder.final_reward(last_grade)
    
    return environment    
