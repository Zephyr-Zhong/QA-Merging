import os
import sys
import json
from pathlib import Path

import numpy as np
import transformers
import re

from transformers import AutoTokenizer

transformers.utils.logging.set_verbosity_error()

start_of_think_token_id=151667 # <think>
end_of_think_token_id=151668 # </think>

def get_avg_length(model_path, result_path):
    result_path = Path(result_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    keywords = ['wait', 're-examine', 'double-check', 'let me check', 'recap', 'let me just verify', 'let me just check']
    pattern = r'|'.join(re.escape(keyword) for keyword in keywords)

    for dataset_name in os.listdir(result_path):
        print(dataset_name)

        dataset_dir = result_path / dataset_name
        for file_name in os.listdir(dataset_dir):
            if not file_name.endswith('.jsonl'):
                continue
            print(f'\t{file_name}', end='')
            length_list = []
            kw_freq_list = []
            think_length_list = []
            level_length_map = {}
            think_length_level_map = {}
            level_acc_map = {}
            level_reflection_map = {}
            level_thinking_map = {}
            reflection_cnt = 0
            thinking_cnt = 0
            with open(dataset_dir / file_name) as f:
                for line_data in f:
                    line_data = json.loads(line_data)
                    for i in range(len(line_data['code'])):
                        encoded_code = tokenizer(line_data['code'][i])['input_ids']
                        length = len(encoded_code)
                        length_list.append(length)
                        # keywords for reflection
                        keywords_match = re.findall(pattern, line_data["code"][i], re.IGNORECASE)
                        if len(keywords_match) > 0:
                            kw_freq_list.append(len(keywords_match))
                            reflection_cnt += 1

                        # find <think> token in response
                        # if start_of_think_token_id in encoded_code and end_of_think_token_id in encoded_code:
                        if end_of_think_token_id in encoded_code:
                            # start_idx = encoded_code.index(start_of_think_token_id)
                            end_idx = encoded_code.index(end_of_think_token_id) + 1
                            think_length = end_idx
                            think_length_list.append(think_length)
                            thinking_cnt += 1
                            if 'level' in line_data:
                                think_length_level_map.setdefault(line_data['level'], []).append(think_length)
                                level_thinking_map.setdefault(line_data['level'], []).append(1)
                        

                        # MATH level-based stats
                        if 'level' in line_data:
                            level_length_map.setdefault(line_data['level'], []).append(length)
                            level_acc_map.setdefault(line_data['level'], []).append(int(line_data["score"][i]))
                            level_reflection_map.setdefault(line_data['level'], []).append(int(len(keywords_match) > 0))
            # print(f'\t{round(np.mean(length_list), 2)}[{reflection_cnt}]; [{round(np.mean(kw_freq_list), 1) if kw_freq_list else 0}]')
            print(f'\t{round(np.mean(length_list), 2)}[{reflection_cnt}]; [#Tokens Max: {round(np.max(length_list), 2)}; Min: {round(np.min(length_list), 2)}]')
            if thinking_cnt == 0:
                think_length_list = [0]
            print(f'\tThinking length [counts]: {round(np.mean(think_length_list), 2)}[{thinking_cnt}]')
            if level_length_map:
                for level, level_length_list in sorted(level_length_map.items()):
                    number_q_level = len(level_acc_map[level])
                    print(
                        f'\t\tlevel-{level}: {round(np.mean(level_length_list),3)}  ;\tAcc: {round(np.mean(level_acc_map[level]) * 100, 1)}  ;\t{round(np.mean(level_reflection_map[level]), 3)}')
                    if level in think_length_level_map:
                        print(
                            f'\t\t\t\tThinking Tokens: {np.sum(think_length_level_map[level])/number_q_level};\t[{round(np.sum(level_thinking_map[level])/number_q_level, 3)}];\t{number_q_level}')
                    


def get_avg_acc(result_path):
    result_path = Path(result_path)

    for dataset_name in os.listdir(result_path):
        print(dataset_name)
        dataset_dir = result_path / dataset_name
        for file_name in os.listdir(dataset_dir):
            if not file_name.endswith('.jsonl'):
                continue
            print(f'\t{file_name}', end='')
            acc = 0
            samples_num = 0
            with open(dataset_dir / file_name) as f:
                for line_data in f:
                    line_data = json.loads(line_data)

                    acc += sum(line_data["score"])
                    samples_num += len(line_data["score"])
            print(f'\tAverage Acc: {round(acc/samples_num*100, 2)}')

if __name__ == '__main__':
    _, _model_path, _result_path = sys.argv
    get_avg_length(_model_path, _result_path)
    get_avg_acc(_result_path)
