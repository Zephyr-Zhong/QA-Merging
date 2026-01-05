#!/bin/bash

# Pretrained model: --language_model_name
# DeepSeek-R1-Distill-Qwen-1.5B
# Qwen2.5-Math-1.5B
# Qwen3-4B-Base
# Qwen3-4B-Instruct-2507
# Qwen3-4B-Thinking-2507
# --do_contrastive \

python ./merge_sequential_llm.py 	--val_shot 64 \
									--batch_size 16 \
									--do_contrastive \
									--lr 0.001 \
									--epochs 100 \
									--do_short_cot \
									--do_long_cot \
									--language_model_name Qwen3-4B-Thinking-2507 \
									--tokenizer_model_name Qwen3-4B-Thinking-2507 \
									--save_path RPAM_cl_w_1000_MI_0.5


