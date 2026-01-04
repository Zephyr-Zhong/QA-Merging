import random
from transformers import AutoTokenizer
import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_eval_data
import pandas as pd

random.seed(0)

    
def construct_pairwise_json_data(data):
    long_cot_samples = []
    short_cot_samples = []

    for item in data:
        gain = item['gain']
        item.pop('gain')
        print(gain)
        if gain <= 0:
            short_cot_samples.append(item)
        elif gain > 0:
            long_cot_samples.append(item)
        else:
            assert False, "gain should not be zero"

    print(f"Short CoT samples: {len(short_cot_samples)}, Percent: {len(short_cot_samples)/len(data)*100:.2f}%")
    print(f"Long CoT samples: {len(long_cot_samples)}, Percent: {len(long_cot_samples)/len(data)*100:.2f}%")
    print("-"*20)
    return short_cot_samples, long_cot_samples

def save_to_jsonl(data, output_path):
    df = pd.DataFrame(data)
    # save dataframe as jsonl
    df.to_json(output_path, orient='records', lines=True)

if __name__ == "__main__":
    model_path = "YOUR MODEL PATH HERE"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    long_cot_data_path = "./propressed_data/Qwen3-4B-Thinking-2507/mix_mathematic_problems.json"
    short_cot_data_path = "./propressed_data/Qwen3-4B-Instruct-2507/mix_mathematic_problems.json"

    long_cot_data = load_eval_data(long_cot_data_path)
    short_cot_data = load_eval_data(short_cot_data_path)

    # default sampling 12 sampels for each problem
    sampling_size = len(long_cot_data[0]['responses'])
    print("sampling_size:", sampling_size)

    assert len(long_cot_data) == len(short_cot_data)

    print("num problems:",len(long_cot_data))
    # sys.exit()
    negative = 0
    postive = 0

    long_acc_random = 0
    short_acc_random = 0

    valid_count = 0

    all_gain = []

    selected_data = []

    total_acc_long = 0
    total_length_long = 0

    total_acc_short = 0
    total_length_short = 0

    total_acc_optimal = 0
    total_length_optimal = 0

    max_length_inc_ratio = 10

    acc_counters = [0,0,0] # >0 =0 <0
    long_lengths = []
    accuracy_diffs = []

    for group_index in range(len(long_cot_data)):
    # for group_index in range(50):

        long_group = long_cot_data[group_index]
        short_group = short_cot_data[group_index]
        print(long_group.keys())

        # calculate correctness
        ground_truth_answer = long_group['gt']
        
        # skip multiple choice questions
        if ground_truth_answer in ["A", "B", "C", "D", "E", "F", "\\text{A}", "\\text{B}", "\\text{C}", "\\text{D}", "\\text{E}", "\\text{F}","\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}", "\\boxed{E}", "\\boxed{F}"]:
            continue
        if ground_truth_answer =="None" or ground_truth_answer == "":
            continue

        long_answers = long_group['pred']
        short_answers = short_group['pred']

        long_correctness = long_group['score']
        short_correctness = short_group['score']

        long_acc_random += long_correctness[0]
        short_acc_random += short_correctness[0]

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
        if long_accuracy == 0 and short_accuracy == 0:
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

        if gain > 0:
            postive += 1
            total_acc_optimal += long_correctness[0]
            total_length_optimal += long_solution_lengths[0]

        if gain <= 0:
            negative += 1
            total_acc_optimal += short_correctness[0]
            total_length_optimal += short_solution_lengths[0]
            

        all_gain.append(gain)
        selected_data.append({
            "instruction": long_group['question'],
            "output": long_group['gt_cot'],
            "gain": gain
        })

        
        if long_avg_length > 4000 and gain < 0:
            print("Problem Index:", group_index)
            print("long_accuracy:",long_accuracy)
            print("short_accuracy:",short_accuracy)
            print("long_avg_length:",long_avg_length)
            print("short_avg_length:",short_avg_length)
            print("relative_accuracy_gain:",relative_accuracy_gain)
            print("relative_length_increnment:",relative_length_increnment)
            print("gain:",gain)
            print("-"*20)


    print("acc_counters:", [a / valid_count for a in acc_counters])

    np.set_printoptions(suppress=True)
    gain_avg = sum(all_gain) / len(all_gain)
    gain_abs_avg = sum([e if e>=0 else -e for e in all_gain]) / len(all_gain)
    print("gain_avg:", gain_avg)
    print("gain abs avg:", gain_abs_avg)
    print("gain max:", max(all_gain))
    print("gain min:", min(all_gain))

    print("[-] negative counts:", negative, "\n[+] postive counts:", postive)
    print("[-] negative gain:",negative / (valid_count))
    print("[+] postive gain:",postive / (valid_count))

    print("total_acc_long:", total_acc_long / valid_count, "total_length_long:", total_length_long / valid_count)
    print("total_acc_short:", total_acc_short / valid_count, "total_length_short:", total_length_short / valid_count)
    print("total_acc_optimal:", total_acc_optimal / valid_count, "total_length_optimal:", total_length_optimal / valid_count)

    # preference models dataset
    long_cot_output_path = "./propressed_data/long_cot_math.jsonl"
    short_cot_output_path = "./propressed_data/short_cot_math.jsonl"
    short_cot_samples, long_cot_samples = construct_pairwise_json_data(selected_data)

    save_to_jsonl(short_cot_samples, short_cot_output_path)
    save_to_jsonl(long_cot_samples, long_cot_output_path)