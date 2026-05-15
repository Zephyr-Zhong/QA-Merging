import os
import gc
import json
import random
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
import logging
from src.representation_correct.interpolation_methods import linear_pareto_interpolation

log = logging.getLogger(__name__)

PoolingMode = Literal["last_token", "mean", "all_tokens", "generation_tokens"]


# -----------------------------------------------------------------------------
# Core ReACT closed-form solver
# -----------------------------------------------------------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def compute_react_weights(
    A_merged: torch.Tensor,
    A_specialist: torch.Tensor,
    beta: float = 0.1,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A_merged.ndim != 2 or A_specialist.ndim != 2:
        raise ValueError(
            f"Expected [N, D] tensors, got {A_merged.shape} and {A_specialist.shape}."
        )
    if A_merged.shape != A_specialist.shape:
        raise ValueError(
            f"Merged and specialist representations must have same shape, got "
            f"{A_merged.shape} and {A_specialist.shape}."
        )

    A_merged = A_merged.to(dtype=dtype)
    A_specialist = A_specialist.to(dtype=dtype)

    if normalize:
        A_merged = l2_normalize(A_merged)
        A_specialist = l2_normalize(A_specialist)

    X = A_merged.t()     # [D, N]
    Y = A_specialist.t() # [D, N]

    C = X @ X.t()        # [D, D]
    B = Y @ X.t()        # [D, D]

    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    W_orth = U @ Vh

    eye = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    reg_inv = C + beta * eye
    target = B + beta * W_orth

    W = torch.linalg.solve(reg_inv.t(), target.t()).t()
    return W, C


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
class CoTResponseReplayDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        response_field,
        template_name="qwen3-think",
        instruction_field="instruction",
        max_length=4096,
    ):
        self.items = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        for ex in rows:
            instruction = str(ex[instruction_field])
            response = str(ex[response_field])

            prompt = format_qwen_prompt(instruction, template_name)
            full_text = prompt + response + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            enc = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).to("cuda")
            attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long).to("cuda")

            generation_mask = torch.zeros_like(input_ids, dtype=torch.bool).to("cuda")
            start = min(len(prompt_ids), input_ids.numel())
            generation_mask[start:] = True

            if generation_mask.sum().item() == 0:
                continue

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "generation_mask": generation_mask,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


QWEN_CHAT_TEMPLATES = {
    "r1-distilled-qwen": (
        "<|User|>{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|Assistant|><think>\n"
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "qwen3-think": (
        "<|im_start|>user {input}\nPlease reason step by step, and put your final answer within \\boxed{{}}./think<|im_end|>\n<|im_start|>assistant <think>"
    )
}


def format_qwen_prompt(instruction: str, template_name: str = "qwen3-think") -> str:
    if template_name not in QWEN_CHAT_TEMPLATES:
        raise KeyError(
            f"Unknown template_name={template_name}. Available: {list(QWEN_CHAT_TEMPLATES.keys())}"
        )
    return QWEN_CHAT_TEMPLATES[template_name].format(input=instruction)


def make_text_collator(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch):
        max_len = max(x["input_ids"].numel() for x in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        generation_mask = torch.zeros((bsz, max_len), dtype=torch.bool)

        for i, item in enumerate(batch):
            L = item["input_ids"].numel()
            input_ids[i, :L] = item["input_ids"]
            attention_mask[i, :L] = item["attention_mask"]
            generation_mask[i, :L] = item["generation_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "generation_mask": generation_mask,
        }

    return collate_fn


# -----------------------------------------------------------------------------
# Hidden-state extraction (layer-wise)
# -----------------------------------------------------------------------------
@dataclass
class HiddenExtractionConfig:
    layer_indices: Optional[List[int]] = None  # decoder-layer indices, e.g. [0,1,...,L-1]
    pooling: PoolingMode = "last_token"
    normalize: bool = True
    max_tokens_per_batch_for_all_tokens: int = 4096


@torch.no_grad()
def extract_hidden_representations_multi_layer(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cuda",
    config: HiddenExtractionConfig = HiddenExtractionConfig(),
) -> Dict[int, torch.Tensor]:
    model.eval()
    device = torch.device(device)
    reps_per_layer: Dict[int, List[torch.Tensor]] = {}
    resolved_layer_indices: Optional[List[int]] = None

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        num_decoder_layers = len(outputs.hidden_states) - 1
        if resolved_layer_indices is None:
            if config.layer_indices is None:
                resolved_layer_indices = list(range(num_decoder_layers))
            else:
                resolved_layer_indices = []
                for idx in config.layer_indices:
                    if idx < 0:
                        idx = num_decoder_layers + idx
                    if idx < 0 or idx >= num_decoder_layers:
                        raise IndexError(
                            f"Invalid decoder layer index {idx}; model has {num_decoder_layers} layers."
                        )
                    resolved_layer_indices.append(idx)
            for layer_idx in resolved_layer_indices:
                reps_per_layer[layer_idx] = []

        generation_mask = batch.get("generation_mask", None)
        if generation_mask is not None:
            generation_mask = generation_mask.to(device)

        for layer_idx in resolved_layer_indices:
            hidden = outputs.hidden_states[layer_idx + 1].float()  # skip embedding state

            if config.pooling == "last_token":
                last_idx = attention_mask.sum(dim=1).clamp_min(1) - 1
                batch_idx = torch.arange(hidden.size(0), device=hidden.device)
                pooled = hidden[batch_idx, last_idx]
            elif config.pooling == "mean":
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            elif config.pooling == "all_tokens":
                pooled = hidden[attention_mask.bool()]
                max_rows = config.max_tokens_per_batch_for_all_tokens
                if pooled.size(0) > max_rows:
                    perm = torch.randperm(pooled.size(0), device=pooled.device)[:max_rows]
                    pooled = pooled[perm]
            elif config.pooling == "generation_tokens":
                if generation_mask is None:
                    raise ValueError("generation_tokens requires generation_mask.")
                valid_mask = attention_mask.bool() & generation_mask.bool()
                pooled = hidden[valid_mask]
                max_rows = config.max_tokens_per_batch_for_all_tokens
                if pooled.size(0) > max_rows:
                    perm = torch.randperm(pooled.size(0), device=pooled.device)[:max_rows]
                    pooled = pooled[perm]
            else:
                raise ValueError(f"Unsupported pooling mode: {config.pooling}")

            if config.normalize:
                pooled = l2_normalize(pooled)

            reps_per_layer[layer_idx].append(pooled.cpu())
            del hidden, pooled

        print(f"Batch {i}: extracted {len(resolved_layer_indices)} layers")
        del outputs, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not reps_per_layer:
        raise RuntimeError("No representations were extracted. Check dataloader format.")

    return {layer_idx: torch.cat(parts, dim=0) for layer_idx, parts in reps_per_layer.items()}


# -----------------------------------------------------------------------------
# Qwen ReACT alignment algorithm
# -----------------------------------------------------------------------------
@dataclass
class QwenReACTConfig:
    num_calibration_samples: int = 128
    batch_size: int = 2
    max_length: int = 2048
    beta: float = 0.1
    seed: int = 42
    device: str = "cuda"
    layer_indices: Optional[List[int]] = None  # decoder-layer indices
    pooling: PoolingMode = "generation_tokens"
    normalize_hidden: bool = True
    solve_dtype: str = "float32"
    output_path: str = "./qwen_react_multilayer_artifact.pt"

    def torch_solve_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        if self.solve_dtype not in mapping:
            raise ValueError(f"Unsupported solve_dtype={self.solve_dtype}")
        return mapping[self.solve_dtype]


class QwenReACTAlignment:
    def __init__(
        self,
        merged_model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        task_model_paths: Dict[str, str],
        task_datasets: Dict[str, Dataset],
        config: QwenReACTConfig = QwenReACTConfig(),
    ):
        self.merged_model = merged_model
        self.tokenizer = tokenizer
        self.task_model_paths = task_model_paths
        self.task_datasets = task_datasets
        self.config = config

        missing = set(task_datasets.keys()) - set(task_model_paths.keys())
        if missing:
            raise ValueError(f"Missing specialist models for tasks: {missing}")

    def _make_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        collate_fn = make_text_collator(self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _extract_for_model(self, model: nn.Module, dataset: Dataset) -> Dict[int, torch.Tensor]:
        model.to(self.config.device)
        model.eval()
        loader = self._make_loader(dataset)
        extraction_config = HiddenExtractionConfig(
            layer_indices=self.config.layer_indices,
            pooling=self.config.pooling,
            normalize=self.config.normalize_hidden,
        )
        return extract_hidden_representations_multi_layer(
            model=model,
            dataloader=loader,
            device=self.config.device,
            config=extraction_config,
        )

    def compute_task_matrices(self, task_name: str, dataset: Dataset, specialist_model: nn.Module) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        if task_name not in self.task_datasets:
            raise KeyError(f"Unknown task_name={task_name}")

        print(f"\n[Qwen-ReACT multi-layer] Task: {task_name}")

        # for _, model in self.task_models.items():
        #     model.to("cpu")
        # self.merged_model.to(self.config.device)
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        print("  - Extracting merged-model hidden states for selected layers...")
        merged_reps = self._extract_for_model(self.merged_model, dataset)

        print("  - Extracting specialist-model hidden states for selected layers...")
        # specialist_model = self.task_models[task_name]
        specialist_model.to(self.config.device)
        specialist_reps = self._extract_for_model(specialist_model, dataset)

        # specialist_model.to("cpu")
        del specialist_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        layer_matrices: Dict[int, torch.Tensor] = {}
        layer_covariances: Dict[int, torch.Tensor] = {}
        common_layers = sorted(set(merged_reps.keys()) & set(specialist_reps.keys()))

        print("Calculating matrices for layers")
        for layer_idx in common_layers:
            A_merged = merged_reps[layer_idx]
            A_specialist = specialist_reps[layer_idx]
            n = min(A_merged.size(0), A_specialist.size(0))
            A_merged = A_merged[:n]
            A_specialist = A_specialist[:n]

            print(f"  - Solving W for decoder layer {layer_idx}: {n} reps, dim={A_merged.size(-1)}")
            W, C = compute_react_weights(
                A_merged.to(self.config.device),
                A_specialist.to(self.config.device),
                beta=self.config.beta,
                normalize=True,
                dtype=self.config.torch_solve_dtype(),
            )

            del A_merged, A_specialist
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            layer_matrices[layer_idx] = W.cpu()
            layer_covariances[layer_idx] = C.cpu()

            del W, C
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del merged_reps, specialist_reps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return layer_matrices, layer_covariances

    def run(self) -> Dict[str, Any]:
        transformation_matrices: Dict[str, Dict[int, torch.Tensor]] = {}
        covariances: Dict[str, Dict[int, torch.Tensor]] = {}

        print(f"\n=== Processing dataset ===")
        indices = list(range(self.config.num_calibration_samples))
        rng = random.Random(0)
        dataset_size_min = min(len(self.task_datasets["short_cot"]), len(self.task_datasets["long_cot"]))
        n = min(self.config.num_calibration_samples, dataset_size_min)
        indices = list(range(dataset_size_min))
        rng.shuffle(indices)
        dataset_long = Subset(self.task_datasets["long_cot"], indices[:n])
        dataset_short = Subset(self.task_datasets["short_cot"], indices[:n])
        dataset = torch.utils.data.ConcatDataset([dataset_long, dataset_short])

        for task_name in self.task_datasets.keys():
        #     dataset = self.task_datasets[task_name]
        #     if self.config.num_calibration_samples > 0:
        #         n = min(self.config.num_calibration_samples, len(dataset))
        #         rng = random.Random(self.config.seed)
        #         indices = list(range(len(dataset)))
        #         rng.shuffle(indices)
        #         dataset = Subset(dataset, indices[:n])
            specialist_model, _ = load_causal_lm_and_tokenizer(self.task_model_paths[task_name])
            W_by_layer, C_by_layer = self.compute_task_matrices(task_name, dataset, specialist_model)
            transformation_matrices[task_name] = W_by_layer
            covariances[task_name] = C_by_layer

        artifact = {
            "transformation_matrices": transformation_matrices,
            "C_mms": covariances,
            "config": asdict(self.config),
            "task_names": list(self.task_datasets.keys()),
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.config.output_path)), exist_ok=True)
        torch.save(artifact, self.config.output_path)
        return artifact


# -----------------------------------------------------------------------------
# Preference adaptation: assemble final corrector W_p for each layer
# -----------------------------------------------------------------------------
def add_preference_corrector_to_artifact(
    artifact: Dict[str, Any],
    preference_name: str,
    preference: Dict[str, float],
    orth_reg: float = 0.1,
) -> Dict[str, Any]:
    transformation_matrices = artifact["transformation_matrices"]  # task -> layer -> W
    covariances = artifact["C_mms"]                                # task -> layer -> C

    task_names: List[str] = []
    for task_name in transformation_matrices.keys():
        weight = float(preference.get(task_name, 0.0))
        if weight <= 0:
            continue
        if task_name not in covariances:
            raise KeyError(f"Missing covariance C for task '{task_name}'.")
        task_names.append(task_name)

    if not task_names:
        raise ValueError("At least one preference weight must be positive.")

    weights = torch.tensor([float(preference[name]) for name in task_names], dtype=torch.bfloat16)
    weights = weights / weights.sum()

    common_layers = sorted(set.intersection(*[set(transformation_matrices[name].keys()) for name in task_names]))
    if not common_layers:
        raise ValueError("No common layers found across selected tasks.")

    W_p_by_layer: Dict[int, torch.Tensor] = {}
    for layer_idx in common_layers:
        selected_matrices = {name: transformation_matrices[name][layer_idx] for name in task_names}
        selected_covariances = {name: covariances[name][layer_idx] for name in task_names}
        W_p = linear_pareto_interpolation(
            matrix_dict=selected_matrices,
            weights=weights,
            Cmm_matrices=selected_covariances,
            orth_reg=orth_reg,
        )
        W_p_by_layer[layer_idx] = W_p

    artifact.setdefault("preference_correctors", {})[preference_name] = W_p_by_layer
    artifact.setdefault("preferences", {})[preference_name] = preference
    return artifact


# -----------------------------------------------------------------------------
# Model/layer utilities
# -----------------------------------------------------------------------------
def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = [
        "model.layers",
        "base_model.model.layers",
        "transformer.h",
    ]

    for path in candidates:
        obj: Any = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj

    raise AttributeError(
        "Cannot locate decoder layers. Expected one of: model.layers, base_model.model.layers, transformer.h."
    )


def get_layer_terminal_linear(layer: nn.Module) -> nn.Linear:
    candidates = [
        "mlp.down_proj",            # Qwen/LLaMA/Mistral
        "feed_forward.w2",          # some architectures
        "mlp.c_proj",               # GPT-style MLP output
        "output.dense",             # BERT-ish fallback
    ]

    for path in candidates:
        obj: Any = layer
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            if not isinstance(obj, nn.Linear):
                raise TypeError(f"Resolved terminal layer '{path}' is not nn.Linear: {type(obj)}")
            return obj

    raise AttributeError(
        "Cannot locate the terminal linear for this decoder layer. "
        "Expected one of: mlp.down_proj, feed_forward.w2, mlp.c_proj, output.dense."
    )


def load_causal_lm_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: Union[str, torch.dtype] = "bfloat16",
):
    if isinstance(torch_dtype, str):
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _effective_W(W: torch.Tensor, strength: float, dtype: torch.dtype) -> torch.Tensor:
    W = W.to(dtype=dtype)
    D = W.shape[0]
    I = torch.eye(D, device=W.device, dtype=dtype)
    return I + strength * (W - I)


@torch.no_grad()
def load_react_into_decoder_layers(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    artifact: Dict[str, Any],
    corrector_name: str,
    output_path: str,
    strength: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if corrector_name in artifact.get("preference_correctors", {}):
        W_by_layer = artifact["preference_correctors"][corrector_name]
    else:
        W_by_layer = artifact["transformation_matrices"][corrector_name]

    layers = get_decoder_layers(model)

    for layer_idx, W in sorted(W_by_layer.items()):
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer index {layer_idx} out of range for model with {len(layers)} layers.")

        layer = layers[layer_idx]
        terminal_linear = get_layer_terminal_linear(layer)
        W_eff = _effective_W(W.to(terminal_linear.weight.device), strength=strength, dtype=dtype)

        old_weight = terminal_linear.weight.data
        new_weight = W_eff.to(old_weight.device, dtype=old_weight.dtype) @ old_weight
        terminal_linear.weight.data.copy_(new_weight.to(terminal_linear.weight.dtype))

        if terminal_linear.bias is not None:
            old_bias = terminal_linear.bias.data
            new_bias = old_bias @ W_eff.t().to(old_bias.device, dtype=old_bias.dtype)
            terminal_linear.bias.data.copy_(new_bias.to(terminal_linear.bias.dtype))

        print(f"Loaded layer {layer_idx} into terminal linear: {terminal_linear.__class__.__name__}")

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved multi-layer ReACT-loaded model to: {output_path}")



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Qwen25
    merged_path = "../NLG/save_merge_models/short_cot_long_cot/contrastive_t_100_MI_0.3/DeepSeek-R1-Distill-Qwen-1.5B/50-0.01-64-16-0.3"
    thinking_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    instruct_path = "Qwen/Qwen2.5-Math-1.5B"


    if "Qwen2.5" in instruct_path:
        based_model = "qwen25"
        dataset_think_template = "r1-distilled-qwen"
        dataset_instruct_template = "r1-distilled-qwen"
        long_json_path = "../../dataset_construction/propressed_data/qwen25_labeled/long_cot_math_0_response_2.jsonl"
        short_json_path = "../../dataset_construction/propressed_data/qwen25_labeled/short_cot_math_0_response_2.jsonl"
    else:
        based_model = "qwen3"
        dataset_think_template = "qwen3-think"
        dataset_instruct_template = "qwen3-think"
        long_json_path = "../../dataset_construction/propressed_data/qwen3_labeled/long_cot_math_0_response.jsonl"
        short_json_path = "../../dataset_construction/propressed_data/qwen3_labeled/short_cot_math_0_response.jsonl"


    merged_model, tokenizer = load_causal_lm_and_tokenizer(merged_path)

    long_dataset = CoTResponseReplayDataset(
        jsonl_path=long_json_path,
        tokenizer=tokenizer,
        response_field="best_response_long",
        template_name=dataset_think_template,
        instruction_field="instruction",
        max_length=2048,
    )

    short_dataset = CoTResponseReplayDataset(
        jsonl_path=short_json_path,
        tokenizer=tokenizer,
        response_field="best_response_short",
        template_name=dataset_instruct_template,
        instruction_field="instruction",
        max_length=1024,
    )

    task_datasets = {
        "long_cot": long_dataset,
        "short_cot": short_dataset,
    }

    task_model_paths = {
        "long_cot": thinking_path,
        "short_cot": instruct_path,
    }


    # load layer indices for computing W
    ranked_layers_json_path = "../NLG/layers_analysis/qwen_25/layer_analysis_outputs_a_0.5_g_1_layer/ranked_layers.json"

    top_k_ratio = 0.3
    with open(ranked_layers_json_path, "r", encoding="utf-8") as f:
        ranked_records = json.load(f)
    sorted_records = sorted(
        ranked_records,
        key=lambda rec: int(rec.get("rank", 10 ** 9))
    )
    top_k = math.ceil(len(sorted_records) * top_k_ratio)
    selected_layer_indices = sorted(
        int(rec["layer"]) - 1
        for rec in sorted_records[top_k:]
    )
    selected_layer_indices.append(len(sorted_records) - 1)   # always include the last layer

    # num_layers = len(get_decoder_layers(merged_model))
    config = QwenReACTConfig(
        num_calibration_samples=32,
        batch_size=1,
        max_length=2048,
        beta=0.1,
        layer_indices=selected_layer_indices,    # 仅计算部分层的 W
        pooling="generation_tokens",
        output_path="./transform_matrix/" + based_model + "/layer_wise/qwen_react_response_tokens_all_layers.pt",
    )

    generate_artifact = True
    if generate_artifact:
        aligner = QwenReACTAlignment(
            merged_model=merged_model,
            tokenizer=tokenizer,
            task_model_paths=task_model_paths,
            task_datasets=task_datasets,
            config=config,
        )
        artifact = aligner.run()
    else:
        log.info(f"Loading ReACT artifact from: {config.output_path}")
        artifact = torch.load(config.output_path, map_location="cpu")
        log.info("Artifact loaded successfully.")

    strength = 0.3
    for short_preference in [0.3]:
        merged_model, _ = load_causal_lm_and_tokenizer(merged_path)

        save_preference_name = "short_pref_" + str(short_preference).replace(".", "_")
        artifact = add_preference_corrector_to_artifact(
            artifact,
            preference_name=save_preference_name,
            preference={"long_cot": 1 - short_preference, "short_cot": short_preference},
            orth_reg=config.beta,
        )

        model_saved_path = "./saved_model/" + based_model + "/linear/strength_" + str(strength).replace(".", "_") + "/" + save_preference_name
        load_react_into_decoder_layers(
            merged_model,
            tokenizer,
            artifact,
            save_preference_name,
            model_saved_path,
            strength=strength,
            dtype=torch.bfloat16,
        )
    

if __name__ == "__main__":
    main()
