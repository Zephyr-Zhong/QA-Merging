import random
import pandas as pd
import random
import json
from utils import load_eval_data

random.seed(0)



if __name__ == "__main__":
    mix_original_dataset_from_source = False  # whether to mix original dataset
    mix_sampled_dataset = False  # whether to mix sampled dataset
    mix_original_dataset_from_propressed = True 
    
    # ================= Original dataset from source =================
    # mix original dataset [GSM8K, MATH, AIME24]
    source_data_base_path = "DATASET_PATH_FOR_GSM8K_MATH_AIME24"
    gsm8k_path = source_data_base_path + "gsm8k/test.jsonl"
    math_path = source_data_base_path + "math/test.jsonl"
    aime_path = source_data_base_path + "aime24/test.jsonl"
    
    output_path = "./propressed_data/original_mix_mathematic_problems.jsonl"
    if mix_original_dataset_from_source:
        print("Mixing original datasets...")
        # load data
        gsm8k_data = pd.read_json(gsm8k_path, lines=True)
        math_data = pd.read_json(math_path, lines=True)
        aime_data = pd.read_json(aime_path, lines=True)
        data = pd.concat([gsm8k_data, math_data, aime_data], ignore_index=True)

        # save mixed dataset to jsonl
        data.to_json(output_path, orient='records', lines=True)


    # ================= Original dataset from propressed =================
    # mix original dataset [GSM8K, MATH, AIME24]
    base_path = "./propressed_data/"
    long_cot_path = base_path + "long_cot_math.jsonl"
    short_cot_path = base_path + "short_cot_math.jsonl"
    
    output_path = "./propressed_data/original_mix_mathematic_problems.jsonl"
    if mix_original_dataset_from_propressed:
        print("Mixing original datasets from propressed ...")
        # load data
        long_cot_data = pd.read_json(long_cot_path, lines=True)
        short_cot_data = pd.read_json(short_cot_path, lines=True)
        data = pd.concat([long_cot_data, short_cot_data], ignore_index=True)

        # save mixed dataset to jsonl
        data.to_json(output_path, orient='records', lines=True)


    # ================= LM sampling dataset =================
    # sampled dataset [GSM8K, MATH, AIME24]
    sampled_base_path = "SAMPLED_DATASETS_PATH"
    long_cot_gsm8k_path = sampled_base_path + "Qwen3-4B-Thinking-2507_12/math_eval_16384/gsm8k/test_qwen3-think_-1_seed0_t0.6_topp0.95_topk20_s0_e-1_n12.jsonl"
    long_cot_math_path = sampled_base_path + "Qwen3-4B-Thinking-2507_12/math_eval_32768/math/test_qwen3-think_-1_seed0_t0.6_topp0.95_topk20_s0_e-1_n12.jsonl"
    long_cot_aime_path = sampled_base_path + "Qwen3-4B-Thinking-2507_12/math_eval_81920/aime24/test_qwen3-think_-1_seed0_t0.6_topp0.95_topk20_s0_e-1_n12.jsonl"
    short_cot_gsm8k_path = sampled_base_path + "Qwen3-4B-Instruct-2507_12/math_eval_16384/gsm8k/test_qwen3-instruct-2507_-1_seed0_t0.7_topp0.80_topk20_s0_e-1_n12.jsonl"
    short_cot_math_path = sampled_base_path + "Qwen3-4B-Instruct-2507_12/math_eval_16384/math/test_qwen3-instruct-2507_-1_seed0_t0.7_topp0.80_topk20_s0_e-1_n12.jsonl"
    short_cot_aime_path = sampled_base_path + "Qwen3-4B-Instruct-2507_12/math_eval_16384/aime24/test_qwen3-instruct-2507_-1_seed0_t0.7_topp0.80_topk20_s0_e-1_n12.jsonl"

    long_cot_output_path = "./propressed_data/Qwen3-4B-Thinking-2507/mix_mathematic_problems.json"
    short_cot_output_path = "./propressed_data/Qwen3-4B-Instruct-2507/mix_mathematic_problems.json"
    if mix_sampled_dataset:
        print("Mixing sampled datasets...")
        # load data
        long_cot_data = load_eval_data({"gsm8k": long_cot_gsm8k_path, "math": long_cot_math_path, "aime": long_cot_aime_path})
        short_cot_data = load_eval_data({"gsm8k": short_cot_gsm8k_path, "math": short_cot_math_path, "aime": short_cot_aime_path})

        # save mixed dataset to json
        long_cot_data.to_json(long_cot_output_path, orient="index")
        short_cot_data.to_json(short_cot_output_path, orient="index")