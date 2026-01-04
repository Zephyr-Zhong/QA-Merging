# we want to resize the model tokenizer to 32001 tokens and model's embedding

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from utils.utils import smart_tokenizer_and_embedding_resize
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--llm_version", type=str, default="v1.0")
    args = parser.parse_args()

    path = f'{args.model_name}' # path for pretrained model
    model_name = path.partition('/')[2]
    print(model_name)
    path2 = f'./MergeLM_models/{model_name}/{args.llm_version}' # path for saving model

    compute_dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 trust_remote_code=True,
                                                 device_map={"":0},
                                                 torch_dtype=compute_dtype)

    print(tokenizer.vocab_size)

    if not os.path.exists(path2):
        os.makedirs(path2)
    
    # resize tokenizer and embedding size
    if "Llama-2" in model_name:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model,
            tokenizer=tokenizer,
        )
    elif "Qwen" in model_name:
        pass
    
    model.generation_config.do_sample = True
    model.save_pretrained(path2)
    tokenizer.save_pretrained(path2)
    print(f'=== Completed - resize_model_tokens: {args.model_name} ===')