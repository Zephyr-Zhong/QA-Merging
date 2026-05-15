#!/bin/bash

# from paper: lr:{0.1, 0.01}, epoch:{50, 100, 200}
# --init_merge_coef: for short-CoT model merging contribution

python ./merge_sequential_llm.py 	--val_shot 64 \
									--batch_size 16 \
									--do_contrastive \
									--lr 0.01 \
									--epochs 50 \
									--do_short_cot \
									--do_long_cot \
									--top_k_ratio 0.3 \
									--init_merge_coef 0.5 \
									--language_model_name DeepSeek-R1-Distill-Qwen-1.5B \
									--tokenizer_model_name DeepSeek-R1-Distill-Qwen-1.5B \
									--save_path contrastive_t_100_MI_0.5 \
									--ranked_layers_json_path ./layers_analysis/qwen_25/layer_analysis_outputs_a_0.5_g_1_layer/ranked_layers.json



