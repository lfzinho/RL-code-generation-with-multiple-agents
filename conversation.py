import tqdm
from coder import Coder
from rl.llm_agent import LLMAgent
from rl.environment import Environment
from rl.code_evaluator import CodeEvaluator
from rl.policies import EpsilonGreedyPolicy
from rl.utils import compute_delta_grade, is_terminate_grade


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
        coder_prompt:str, 
        reviewer: LLMAgent, 
        refiner: LLMAgent, 
        max_turns:int=5
    ):
    global evaluator
    environment = Environment()
    # Set the environment for all agents
    for agent in [coder, reviewer, refiner, evaluator]:
        agent.environment = environment
    # Start the conversation
    last_grade = None
    coder.add_message(coder_prompt)
    for turn in tqdm.tqdm(range(max_turns), desc="Conv. turns", position=1, leave=False):
        # Code is evaluated
        grade = evaluator.evaluate_code()
        # If grade is terminate, break the loop
        if is_terminate_grade(grade):
            break
        # If it is the first turn, reward the coder
        elif last_grade is None:
            coder.first_reward(grade)
        # If it is not the first turn, reward refiner and reviewer
        else:
            delta_grade = compute_delta_grade(last_grade, grade)
            refiner.reward(delta_grade)
            reviewer.reward(delta_grade)
        # Keeps the conversation going
        reviewer.add_message()
        refiner.add_message()
        last_grade = grade
    # Reward the coder with the last grade
    coder.final_reward(last_grade)
