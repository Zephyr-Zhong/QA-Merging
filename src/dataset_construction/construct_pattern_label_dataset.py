import random
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_eval_data
import pandas as pd

random.seed(0)


import numpy as np


def pick_shortest_correct_response(responses, correctness, solution_lengths):
    # Filter out the correct responses and their corresponding lengths
    correct_responses = [(resp, length) for resp, corr, length in zip(responses, correctness, solution_lengths) if corr == 1]
    
    if not correct_responses:
        assert False, "No correct responses available"
    
    # Find the shortest correct response
    shortest_response = min(correct_responses, key=lambda x: x[1])

    # return the response text of the shortest correct response
    return shortest_response[0] 


    
# Construct PL dataset in Long-CoT and Short-CoT based on threshold and training samples
def construct_pairwise_json_data(data, gain_threshold, data_size):
    long_cot_samples = []
    short_cot_samples = []
    L2S_samples = []

    for item in data:
        gain = item['gain']
        item.pop('gain')
        if gain <= 0:
            # chosen model: short_cot
            short_cot_samples.append(item)
        elif gain > gain_threshold:
            # chosen model: long_cot
            long_cot_samples.append(item)
        else:
            assert False, "gain should not be zero"

    print(f"Short CoT samples: {len(short_cot_samples)}, Percent: {len(short_cot_samples)/len(data)*100:.2f}%")
    print(f"Long CoT samples: {len(long_cot_samples)}, Percent: {len(long_cot_samples)/len(data)*100:.2f}%")
    print(f"L2S samples: {len(L2S_samples)}, Percent: {len(L2S_samples)/len(data)*100:.2f}%")
    print("-"*20)

    return short_cot_samples, long_cot_samples

def save_to_jsonl(data, output_path):
    df = pd.DataFrame(data)
    # save dataframe as jsonl
    df.to_json(output_path, orient='records', lines=True)


if __name__ == "__main__":
    model = "qwen25" # "qwen25" OR "qwen3"
    gain_threshold = 0
    save_dataset = True
    

    if model == "qwen25":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        long_cot_data_path = "./propressed_data/DeepSeek-R1/mix_mathematic_problems.json"
        short_cot_data_path = "./propressed_data/Qwen2.5-Math-1.5B/mix_mathematic_problems.json"
        save_folder = "./propressed_data/qwen25_labeled/"
    elif model == "qwen3":  
        model_path = "Qwen/Qwen3-4B-Thinking-2507"
        long_cot_data_path = "./propressed_data/Qwen3-4B-Thinking-2507/mix_mathematic_problems.json"
        short_cot_data_path = "./propressed_data/Qwen3-4B-Instruct-2507/mix_mathematic_problems.json"
        save_folder = "./propressed_data/qwen3_labeled/"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    long_cot_data = load_eval_data(long_cot_data_path)
    short_cot_data = load_eval_data(short_cot_data_path)

    # default sampling 12 sampels for each problem
    sampling_size = len(long_cot_data[0]['pred'])
    print("sampling_size:", sampling_size)
    assert len(long_cot_data) == len(short_cot_data)
    print("num problems:",len(long_cot_data))

    negative = 0
    positive = 0
    negative_gain = []
    positive_gain = []

    long_acc_random = 0
    short_acc_random = 0

    valid_count = 0

    all_gain = []

    selected_data = []

    total_acc_long = 0
    total_length_long = 0

    total_acc_short = 0
    total_length_short = 0

    total_acc_optimal_long = 0
    total_length_optimal_long = 0

    total_acc_optimal_short = 0
    total_length_optimal_short = 0

    max_length_inc_ratio = 10

    acc_counters = [0,0,0] # >0 =0 <0
    long_lengths = []
    accuracy_diffs = []

    for group_index in tqdm(range(len(long_cot_data))):
        long_group = long_cot_data[group_index]
        short_group = short_cot_data[group_index]

        long_answers = long_group['pred']
        short_answers = short_group['pred']

        long_correctness = long_group['score']
        short_correctness = short_group['score']

        # calculate lengths
        long_solutions = [solution for solution in long_group['responses']]
        short_solutions = [solution for solution in short_group['responses']]

        long_solution_lengths = [len(tokenizer(solution)['input_ids']) for solution in long_solutions]
        short_solution_lengths = [len(tokenizer(solution)['input_ids']) for solution in short_solutions]

        long_accuracy = sum(long_correctness) / len(long_correctness)
        short_accuracy = sum(short_correctness) / len(short_correctness)

        long_avg_length = sum(long_solution_lengths) / len(long_solution_lengths)
        short_avg_length = sum(short_solution_lengths) / len(short_solution_lengths)

        
        relative_accuracy_gain = long_accuracy - short_accuracy - 1/(2*sampling_size) #/ short_accuracy if short_accuracy != 0 else (long_accuracy - 1/sampling_size) / (1/sampling_size)
        relative_length_increnment = (long_avg_length - short_avg_length) / short_avg_length

        if relative_accuracy_gain > 0:
            gain = relative_accuracy_gain / relative_length_increnment
        else:
            gain = relative_accuracy_gain * (relative_length_increnment/max_length_inc_ratio)

        # a special case
        if long_accuracy == 0 or short_accuracy == 0:
            continue
        if short_avg_length > long_avg_length:
            continue
        
        valid_count += 1
        
        acc_diff = long_accuracy - short_accuracy
        acc_counters[0] += 1 if acc_diff > 0 else 0
        acc_counters[1] += 1 if acc_diff == 0 else 0
        acc_counters[2] += 1 if acc_diff < 0 else 0

        long_lengths.append(long_avg_length)
        accuracy_diffs.append(acc_diff)


        total_acc_long += long_correctness[0]
        total_length_long += long_solution_lengths[0]

        total_acc_short += short_correctness[0]
        total_length_short += short_solution_lengths[0]

        if gain > gain_threshold:
            positive += 1
            positive_gain.append(gain)
            total_acc_optimal_long += long_correctness[0]
            total_length_optimal_long += long_solution_lengths[0]
        else:
            # gain <= gain_threshold
            negative += 1
            negative_gain.append(gain)
            total_acc_optimal_short += short_correctness[0]
            total_length_optimal_short += short_solution_lengths[0]
        
        best_response_short = pick_shortest_correct_response(short_solutions, short_correctness, short_solution_lengths)
        best_response_long = pick_shortest_correct_response(long_solutions, long_correctness, long_solution_lengths)

        all_gain.append(gain)
        selected_data.append({
            "instruction": long_group['question'],
            "output": long_group['gt_cot'],
            "gain": gain,
            "best_response_long": best_response_long,
            "best_response_short": best_response_short
        })


    short_cot_samples, long_cot_samples = construct_pairwise_json_data(selected_data, gain_threshold, 64)

    # save dataset
    if save_dataset:
        long_cot_output_path = save_folder + "long_cot_math_"+str(gain_threshold)+"_response_2.jsonl"
        short_cot_output_path = save_folder + "short_cot_math_"+str(gain_threshold)+"_response_2.jsonl"
        save_to_jsonl(short_cot_samples, short_cot_output_path)
        save_to_jsonl(long_cot_samples, long_cot_output_path)