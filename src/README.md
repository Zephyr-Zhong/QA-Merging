## Environment
see in environment.yaml

## Datasets
The processed data is in ./src/math_code_data

## Base Models
Downloading base models from Huggingface.

## Preprocess Base Models
Run
```shell
preprocess_llm.sh
```

## Modify transformers package
Add 
```python
        self.causal_mask = attention_mask
        self.position_ids = position_ids
        self.past_key_values = past_key_values
        self.output_attentions = output_attentions
        self.use_cache = use_cache
        # judge whether the attribute self has layers
        if not hasattr(self, "layers"):
                return hidden_states
```
in lib/python3.10/site-packages/transformers/models/qwen3/modeling_qwen3.py: Qwen3Model.forawrd before (*transformers==4.52.4*)
```python
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
```

## Training
Merging Long-CoT and Short-CoT:
```shell
sh qa_merge.sh
```