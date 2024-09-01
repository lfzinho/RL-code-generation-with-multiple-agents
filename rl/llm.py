from rl.environment import Environment, Message


class LLM:
    @staticmethod
    def generate_text(environment: Environment) -> str:
        return f"This is a generated text with the prompt: {environment.prompt}"
    
    @staticmethod
    def evaluate_code(environment: Environment, prompt: str) -> int:
        environment.set_prompt(prompt)
        code_message = environment.get_last_message(from_coder=True)
        review_message = environment.get_last_message(from_reviewer=True)
        
        # Placeholder code to evaluate the code
        grade = 0
        text_values = {
            "Best": 5,
            "Good": 4,
            "Neutral": 3,
            "Bad": 2,
            "Worst": 1
        }
        for message in [code_message, review_message]:
            if message is None:
                continue
            for text, value in text_values.items():
                if text in message.content:
                    grade += value
                    break
        
        return f"The code was graded as {grade}"