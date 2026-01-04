import numpy as np
import pandas as pd

from torch.utils.data import Subset
import datasets
import transformers
from utils.prompt_construct_utils import *

class LLMDataLoader:
    def __init__(self, tokenizer: transformers.AutoTokenizer):
        self.tokenizer = tokenizer
        self.math500_path = "math_code_data/MATH_train.jsonl"
        self.gsm8k_path = "math_code_data/gsm8k_test.jsonl"
        self.long_cot_path = "math_code_data/long_cot_math.jsonl"
        self.short_cot_path = "math_code_data/short_cot_math.jsonl"
        self.max_len = 0

    def encode(self, examples: dict, max_seq_length: int = 1024):
        inputs = {}
        ins_token = self.tokenizer(examples['instruction'], max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs['input_ids'] = ins_token['input_ids']
        inputs['attention_mask'] = ins_token['attention_mask']
        self.max_len = max(self.max_len, inputs['attention_mask'][0].sum().item())   
        return inputs
        
    def load_dataset(self, dataset_name: str, max_seq_length: int = 1024, val_shot: int = 64):
        # train: 64, other is test
        if dataset_name == "long_cot":

            math_data = pd.read_json(self.long_cot_path, lines=True)
            data_df = math_data[['instruction', 'output']]

            data_df['instruction'] = data_df['instruction'].apply(lambda x: get_math_task_prompt_thinking().format(instruction=x))
        elif dataset_name == "short_cot":

            math_data = pd.read_json(self.short_cot_path, lines=True)
            data_df = math_data[['instruction', 'output']]

            data_df['instruction'] = data_df['instruction'].apply(lambda x: get_math_task_prompt_nothink().format(instruction=x))
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        dataset = datasets.Dataset.from_pandas(data_df)
        dataset = dataset.map(self.encode, batched=True)

        permuted_indices = np.random.RandomState(seed=0).permutation(len(dataset)).tolist()
        num_train_data = val_shot
        train_dataset = Subset(dataset=dataset, indices=permuted_indices[:num_train_data])
        test_dataset = Subset(dataset=dataset, indices=permuted_indices[num_train_data:])
        return train_dataset, test_dataset

