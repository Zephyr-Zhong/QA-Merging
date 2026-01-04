# for qwen instruct
def get_math_task_prompt_thinking():
    # qwen3-think (for thinking mode)
    problem_prompt = (
        "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}./think<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )
    return problem_prompt


def get_math_task_prompt_nothink():
    # qwen3-think (for non-thinking mode)
    problem_prompt = (
        "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}./no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return problem_prompt