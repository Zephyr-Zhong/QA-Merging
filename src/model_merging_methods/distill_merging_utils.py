import copy
import random
import re
import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

compute_dtype = torch.bfloat16

def check_gpu():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2
        allocated_mem = torch.cuda.memory_allocated(i) / 1024 ** 2
        cached_mem = torch.cuda.memory_reserved(i) / 1024 ** 2
        print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {total_mem:.2f} MB")
        print(f"  Allocated memory: {allocated_mem:.2f} MB ({allocated_mem / total_mem * 100:.2f}%)")
        print(f"  Cached memory (reserved): {cached_mem:.2f} MB ({cached_mem / total_mem * 100:.2f}%)")
        print()


def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(model):
    orig_params = tuple(model.parameters())
    names = []
    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(model, names, params):
    for name, p in zip(names, params):
        set_attr(model, name.split("."), p)


def del_ex(model, exclude):
    new_model = copy.deepcopy(model)
    for param_name, param_value in model.named_parameters():
        exc = [re.match(regex, param_name) for regex in exclude]
        if any(exc):
            del_attr(new_model, param_name.split("."))
    return new_model


class MergedModel(nn.Module):
    def __init__(self, pretrained_model, models, granularity):
        super(MergedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.models = models
        self.granularity = granularity
        # =============== Change intialized merged ratio here
        self.init_merge_weights = 0.5

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.alphas = nn.ParameterList()
        for model in self.models:
            alpha = nn.ParameterList()
            if self.granularity == 'taskwise':
                alpha.append(nn.Parameter(
                    torch.tensor(self.init_merge_weights), requires_grad=True))
            elif self.granularity == 'layerwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(
                        torch.tensor(self.init_merge_weights), requires_grad=True))
            elif self.granularity == 'elementwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(torch.ones_like(
                        param) * self.init_merge_weights, requires_grad=True))
            else:
                raise NotImplementedError(
                    f'Invalid granularity: {self.granularity}')
            self.alphas.append(alpha)

        self.merged_model = copy.deepcopy(
            self.pretrained_model)
        _, self.names = make_functional(self.merged_model)

    def get_merged_model(self):
        merged_param = []
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            # for k in range(len(self.models)):
            k=0
            if self.granularity == 'taskwise':
                alpha = self.alphas[k][0]
            else:
                alpha = self.alphas[k][idx]
            param += alpha * \
                (dict(self.models[k].named_parameters())[
                    name] - pretrained_param)
            
            param += pretrained_param
            merged_param.append(param)

        load_weights(self.merged_model, self.names, merged_param)

        return self.merged_model

    def get_named_parameters(self):
        merged_param = {}
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                param += alpha * \
                    (dict(self.models[k].named_parameters())[
                        name] - pretrained_param)
            param += pretrained_param
            merged_param[name] = param
        return merged_param

    def forward(self, x):
        merged_model = self.get_merged_model()
        if isinstance(x, dict):
            return merged_model(**x)
        else:
            return merged_model(x)

    def turn_on_layer(self, layer_idx):
        layer_name = f'layer.{layer_idx}'
        assert self.granularity in ['layerwise', 'elementwise']
        for idx, (name, _) in enumerate(self.pretrained_model.named_parameters()):
            for k in range(len(self.models)):
                alpha = self.alphas[k][idx]
                if layer_name in name:
                    alpha.requires_grad = True
                else:
                    alpha.requires_grad = False

    def get_average_alpha(self):
        # return the average alpha for each model [model1_avg_alpha, model2_avg_alpha, ...]
        avg_alphas = []
        for k in range(len(self.models)):
            if self.granularity in ['layerwise', 'elementwise']:
                alpha_values = sum([torch.sum(alpha) for alpha in self.alphas[k]]).item()
                num_params = sum(p.numel() for p in self.alphas[k].parameters())
                avg_alpha = alpha_values / num_params
            avg_alphas.append(avg_alpha)
        return avg_alphas


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class LabeledDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_indices = []
        for i, dataset in enumerate(datasets):
            self.dataset_indices.extend(
                [(i, idx) for idx in range(len(dataset))])
        random.shuffle(self.dataset_indices)

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.dataset_indices[index]
        sample = self.datasets[dataset_idx][sample_idx]
        return sample, dataset_idx


def custom_collate_fn(batch):
    # Custom collate function to handle varying input sizes
    data = [item[0] for item in batch]
    source_loader = torch.tensor([item[1] for item in batch])
    return {'data': data, 'source_loader': source_loader}


def merge_data_loaders_from_trainers(trainers, batch_size=16, num_workers=0):
    # Extract datasets from the data loaders
    datasets = []
    for trainer in trainers:
        dataloader = trainer.get_train_dataloader()
        dataset = []
        for item in dataloader:
            dataset.append(trainer._prepare_inputs(item))
        datasets.append(dataset)

    # Create a merged dataset
    merged_dataset = LabeledDataset(datasets)

    # Create a new data loader from the merged dataset
    merged_loader = DataLoader(
        merged_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return merged_loader




class TransformedDataDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def transformed_data_collate_fn(batch):
    data = batch[0][0]
    source_loaders = batch[0][1]
    if len(batch[0]) > 2:
        attention_mask = batch[0][2]
    else:
        return {'data': data, 'source_loader': source_loaders}
    return {'data': data, 'source_loader': source_loaders, 'attention_mask': attention_mask}


def transform_data_loader_prelayer(data_loader, model, device, num_workers=0, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            source_loader = data['source_loader']

            output = model(x)

            # batchsize = 1
            transformed_data.append((output[0].cpu(), source_loader, output[1].cpu()))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader



def remove_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def load_pretrained_model(args):
    if 'Llama' in args.language_model_name or "Qwen" in args.language_model_name:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.path.join(args.cache_dir, args.language_model_name), args.llm_version),
                                                                                                    device_map=args.device,
                                                                                                    trust_remote_code=True,
                                                                                                    torch_dtype=compute_dtype)
        remove_grad(pretrained_model)

    return pretrained_model


def load_fine_tuned_model(args, dataset_name):
    if 'Llama' in args.language_model_name or "Qwen" in args.language_model_name:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.path.join(args.cache_dir, args.load_model_paths_dict[dataset_name]), args.llm_version), 
                                                                                                device_map=args.device,
                                                                                                trust_remote_code=True,
                                                                                                torch_dtype=compute_dtype)
        remove_grad(model)
    return model


def get_weight_map_llm(model_name, args):
    model_path = os.path.join(os.path.join(args.cache_dir, model_name), f'{args.llm_version}/split')
    weight_map = json.load(open(os.path.join(model_path, 'model_index.json')))
    return weight_map


def load_part_model(args, module_name, model_name):
    weight_map = get_weight_map_llm(model_name, args)
    model_path = os.path.join(os.path.join(args.cache_dir, model_name), f'{args.llm_version}/split')
    weight_path = os.path.join(model_path, weight_map[module_name])
    model = torch.load(weight_path, weights_only=False).to(args.device)
    remove_grad(model)
    return model


def load_merged_layers_llm(args, layer_idx):
    layer_pretrained = load_part_model(args, f'model.layers.{layer_idx}', args.language_model_name)

    layers = []
    for dataset in args.dataset_names:
        layer = load_part_model(args, f'model.layers.{layer_idx}', args.task_model_mapping_dict[dataset])
        layers.append(layer)

    merged_layers = MergedModel(layer_pretrained, layers, 'elementwise')
    return merged_layers, layers


# load models and merge them with merge_coef, return a initialized merged model for restore merged weights in trained layers
def load_avg_merged_model_llm(args, merge_coef=0.5):
    pre_model = load_pretrained_model(args)

    modules = ['model.embed_tokens.', 'model.norm.', 'lm_head.']

    num_layers = 36 # number of layers in model
    for i in range(num_layers):
        modules.append(f'model.layers.{i}.')

    for mod in modules:
        for name, param in pre_model.named_parameters():
            # flag = False
            if mod not in name:
                continue
            value = dict(pre_model.named_parameters())[name].clone()
            
            # for dataset in args.dataset_names:
            dataset = args.dataset_names[0]  # only merge one model in dataset
            model = load_part_model(args, mod[:-1], args.task_model_mapping_dict[dataset])

            value += (dict(model.named_parameters())[name[len(mod):]] - dict(pre_model.named_parameters())[name]) * merge_coef
            del model
            torch.cuda.empty_cache()
            set_attr(pre_model, name.split('.'), nn.Parameter(value, requires_grad=False))
            del value

    return pre_model


def load_avg_merged_model_pre_llm(args, merge_coef=0.5):
    pre_model = load_pretrained_model(args).model
    check_gpu()
    del pre_model.norm, pre_model.layers
    check_gpu()

    new_state_dict = {}

    # for dataset in args.dataset_names:
    dataset = args.dataset_names[0]  # only merge one model in dataset
    model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset])
    for name, param in model.named_parameters():
        new_param = (dict(model.named_parameters())[name] - dict(pre_model.named_parameters())[f'embed_tokens.{name}']) * merge_coef
        if new_state_dict.get(f'embed_tokens.{name}') is None:
            new_state_dict[f'embed_tokens.{name}'] = new_param
        else:
            new_state_dict[f'embed_tokens.{name}'] += new_param
    del model
    torch.cuda.empty_cache()

    for name, value in new_state_dict.items():
        set_attr(pre_model, name.split('.'), nn.Parameter(value + dict(pre_model.named_parameters())[name], requires_grad=False))
    return pre_model


def load_single_merged_model_pre_llm(args, dataset):
    pre_model = load_pretrained_model(args).model
    check_gpu()
    del pre_model.norm, pre_model.layers
    check_gpu()

    new_state_dict = {}

    model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset])
    for name, param in model.named_parameters():
        new_param = (dict(model.named_parameters())[name] - dict(pre_model.named_parameters())[f'embed_tokens.{name}'])
        if new_state_dict.get(f'embed_tokens.{name}') is None:
            new_state_dict[f'embed_tokens.{name}'] = new_param
        else:
            new_state_dict[f'embed_tokens.{name}'] += new_param
    del model
    torch.cuda.empty_cache()

    for name, value in new_state_dict.items():
        set_attr(pre_model, name.split('.'), nn.Parameter(value + dict(pre_model.named_parameters())[name], requires_grad=False))

    return pre_model


def transform_data_loader_prelayer_pertask_llm(data_loader, merged_model, models, device, num_workers=0, shuffle=True, batch_size=1):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            source_loader = data['source_loader']
            inputs = []

            # model_output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            model_output = merged_model(**x)
            inputs.append(model_output)
            for model in models:
                model_output = model(**x)
                inputs.append(model_output)

            # shape of inputs: [num_tasks+1, batch_size, seq_length, embedding_dim] -> [batch_size, num_tasks+1, seq_length, embedding_dim]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()

            # batchsize = 1
            transformed_data.append((inputs, source_loader))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_layer_pertask_llm(data_loader, merged_model, models, device, 
                                            pre_causal_mask, pre_cache_position, pre_position_embeddings):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            # shape of x: [batch_size, num_tasks+1, seq_length, embedding_dim] -> [num_tasks+1, batch_size, seq_length, embedding_dim]
            x = data['data'].to(device)
            x = x.permute(1, 0, 2, 3)

            source_loader = data['source_loader']

            inputs = []

            # output = merged_model(x[0], pre_causal_mask[0], pre_position_ids[0])[0]
            output = merged_model(
                    x[0], 
                    attention_mask=pre_causal_mask[0], 
                    cache_position=pre_cache_position[0], 
                    position_embeddings=pre_position_embeddings[0])[0]
            inputs.append(output)
            idx = source_loader.item()
            for model in models:
                output = model(
                            x[idx+1], 
                            attention_mask=pre_causal_mask[idx+1], 
                            cache_position=pre_cache_position[idx+1], 
                            position_embeddings=pre_position_embeddings[idx+1])[0]
                inputs.append(output)

            # shape of inputs: [num_tasks, batch_size, seq_length, embedding_dim] -> [batch_size, num_tasks, seq_length, embedding_dim]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()

            # batchsize = 1
            transformed_data.append((inputs, source_loader))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=True,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=0)

    return new_dataloader
