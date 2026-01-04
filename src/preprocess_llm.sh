#!/bin/bash
# ==================== Qwen2.5-1.5B models ====================
#resize/store llm experts
# python resize_model_tokens.py --model_name Qwen/Qwen2.5-Math-1.5B
# python resize_model_tokens.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# # split model in layers
# python split_model.py --model_name DeepSeek-R1-Distill-Qwen-1.5B
# python split_model.py --model_name Qwen2.5-Math-1.5B


# ==================== Qwen3-4B models ====================
#resize/store llm experts
# python resize_model_tokens.py --model_name "Qwen/Qwen3-4B-Base"
python resize_model_tokens.py --model_name "Qwen/Qwen3-4B-Thinking-2507"
python resize_model_tokens.py --model_name "Qwen/Qwen3-4B-Instruct-2507"

# # split model in layers
# python split_model.py --model_name Qwen3-4B-Base
python split_model.py --model_name Qwen3-4B-Thinking-2507
python split_model.py --model_name Qwen3-4B-Instruct-2507
