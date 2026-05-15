# for qwen3 instruct
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

# for qwen25 thinking & non-thinking prompts
def get_qwen25_math_task_prompt_thinking():
    # r1-distilled-qwen
    problem_prompt = (
        "<|User|>{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|Assistant|><think>\n"
    )
    return problem_prompt

def get_qwen25_math_task_prompt_nothink():
    problem_prompt = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return problem_prompt