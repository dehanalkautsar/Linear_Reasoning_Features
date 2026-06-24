import torch
import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from bert_score import score
import statistics
from ast import literal_eval
import functools
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
from os.path import join
from utils import (
    load_prompt_template,
    set_act_modify_hooks,
    remove_hooks,
    generate_questions_in_hook,
    evaluation_on_dataset,
    compute_performance_on_reason_memory_subset,
    compute_performance_on_reason_subset,
    get_prediction,
    load_dataset,
    get_candidate_directions,
    get_candidate_directions_bloom,
    normalize_bloom_label,
    BLOOM_LABELS,
    BLOOM_LANG_MAP,
    bloom_taxo_path,
)

import sys
import re
import argparse
from collections import Counter

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Set paths dynamically for the model, output, and dataset directories.")
    
    # 添加命令行参数
    parser.add_argument('--model_dir', type=str, default="/mnt/workspace/workgroup/yhhong/transformers", help="Directory for the model")
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3-8B-Instruct', help="Name of the model")
    parser.add_argument('--output_dir', type=str, default='/mnt/workspace/Interp_Reasoning/outputs', help="Output directory")
    parser.add_argument('--dataset_dir', type=str, default='/mnt/workspace/Interp_Reasoning/dataset', help="Dataset directory")
    parser.add_argument('--Intervention', type=bool, default=False, help="Whether to perform features intervention")
    parser.add_argument('--dataset_name', type=str, default='ReasonMem', help="ReasonMem, BloomTaxo, GSM8k, PopQA, C-Eval-H, MGSM, GSM-symbolic")
    parser.add_argument('--lang', type=str, default='en', help="ReasonMem language suffix, e.g. en, ar, ch, ind")
    parser.add_argument('--hs_cache_dir', type=str, default='/mnt/workspace/workgroup/yhhong', help="hs_cache_dir")
    parser.add_argument('--scale', type=float, default=0.1, help="scale for intervention")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch size for evaluation")
    parser.add_argument('--metrics_out', type=str, default=None, help="Path to write metrics JSON")
    parser.add_argument('--answers_dir', type=str, default=None, help="Base directory to save model responses JSONs")
    parser.add_argument('--label_subset', type=str, default='all',
                        help="ReasonMem: all, reasoning, memorization. BloomTaxo: all or a label name.")
    parser.add_argument('--target_label', type=str, default=None,
                        help="BloomTaxo only: target label for one-vs-rest direction.")
    parser.add_argument('--direction_mode', type=str, default='mean_all', choices=("mean_all", "mean_others"),
                        help="BloomTaxo direction mode: mean_all or mean_others.")
    parser.add_argument('--layer_start', type=int, default=None,
                        help="Intervention start layer index (inclusive)")
    parser.add_argument('--layer_end', type=int, default=None,
                        help="Intervention end layer index (inclusive)")


    args = parser.parse_args()
    env_layer_start = os.environ.get("LAYER_START")
    env_layer_end = os.environ.get("LAYER_END")
    if args.layer_start is None and env_layer_start:
        try:
            args.layer_start = int(env_layer_start)
        except ValueError:
            raise ValueError(f"Invalid LAYER_START env value: {env_layer_start}")
    if args.layer_end is None and env_layer_end:
        try:
            args.layer_end = int(env_layer_end)
        except ValueError:
            raise ValueError(f"Invalid LAYER_END env value: {env_layer_end}")
    return args

def model_tag_from_name(model_name):
    return os.path.basename(model_name.rstrip('/'))


def model_family_from_name(model_name):
    tag = model_tag_from_name(model_name).lower()
    if "llama" in tag:
        return "llama"
    if "gemma" in tag:
        return "gemma"
    if "qwen" in tag:
        return "qwen"
    return re.sub(r"[^a-z0-9._-]+", "_", tag) or "model"


def normalize_answer_lang(lang):
    if not lang:
        return "default"
    lang = str(lang).strip().lower()
    return BLOOM_LANG_MAP.get(lang, lang)


def resolve_answers_base_dir(args):
    if args.answers_dir:
        return args.answers_dir
    env_dir = os.environ.get("ANSWERS_DIR")
    if env_dir:
        return env_dir
    return os.path.join(os.path.dirname(__file__), "answers")


def dataset_dir_name(dataset_name):
    lowered = str(dataset_name or "").strip().lower()
    if lowered == "reasonmem":
        return "reason_mem"
    if lowered == "bloomtaxo":
        return "bloom_taxo"
    return re.sub(r"[^a-z0-9._-]+", "_", lowered) or "dataset"


def normalize_label_tag(dataset_name, label_subset, target_label=None):
    label_raw = label_subset if label_subset is not None else "all"
    if dataset_name == "BloomTaxo":
        label_norm = normalize_bloom_label(label_raw) if label_raw != "all" else "all"
        if label_norm == "all" and target_label:
            label_norm = f"all_target_{normalize_bloom_label(target_label)}"
    else:
        label_norm = normalize_label(label_raw) if label_raw != "all" else "all"
    label_norm = str(label_norm).strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "_", label_norm) or "all"


def format_scale_dir(scale_value):
    try:
        return f"{float(scale_value):.2f}"
    except (TypeError, ValueError):
        return "scale"


def build_answers_path(args, dataset_name, label_subset, target_label=None, layer=None, model_layers_num=None):
    base_dir = resolve_answers_base_dir(args)
    if not base_dir:
        return None
    dataset_dir = dataset_dir_name(dataset_name)
    model_dir = model_family_from_name(args.model_name)
    lang_dir = normalize_answer_lang(args.lang)
    run_dir = "baseline" if not args.Intervention else format_scale_dir(args.scale)
    answers_dir = os.path.join(base_dir, dataset_dir, model_dir, lang_dir, run_dir)
    os.makedirs(answers_dir, exist_ok=True)

    label_tag = normalize_label_tag(dataset_name, label_subset, target_label=target_label)
    if layer is None:
        filename = f"responses_{label_tag}.json"
    else:
        width = len(str((model_layers_num - 1) if model_layers_num else layer))
        layer_str = str(layer).zfill(max(width, 1))
        filename = f"layer_{layer_str}_{label_tag}.json"
    return os.path.join(answers_dir, filename)


def write_answers(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def reasonmem_paths(dataset_dir, lang):
    dataset_filename = f"reason_mem_labels_mmlu_{lang}_validation.json"
    responses_filename = f"reason_mem_labels_mmlu_{lang}_responses.json"
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    responses_path = os.path.join(dataset_dir, responses_filename)
    return dataset_path, responses_path, dataset_filename


def preflight_checks(args, dataset_name):
    model_tag = model_tag_from_name(args.model_name)
    if dataset_name == 'ReasonMem':
        dataset_path, _, dataset_filename = reasonmem_paths(args.dataset_dir, args.lang)
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"ReasonMem dataset not found: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        label_counts = Counter([normalize_label(d.get('label', '')) for d in data])
        if label_counts.get('reasoning', 0) == 0:
            print(f"Warning: no Reasoning labels found in {dataset_filename}")
        if label_counts.get('memorization', 0) == 0 and label_counts.get('memory', 0) == 0:
            print(f"Warning: no Memory/Memorization labels found in {dataset_filename}")
        if args.label_subset not in ('all', 'reasoning', 'memorization'):
            raise ValueError(f"Invalid --label_subset: {args.label_subset}")
        if args.Intervention:
            lang_suffix = f"_{args.lang}" if args.lang else ""
            cache_path = os.path.join(args.hs_cache_dir, 'reasoning_representations_outputs',
                                      f'{model_tag}-base_hs_cache_no_cot_all{lang_suffix}.pt')
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(f"Hidden-state cache not found: {cache_path}")
    elif dataset_name == 'BloomTaxo':
        dataset_path = bloom_taxo_path(args.dataset_dir, args.lang)
        dataset_filename = os.path.basename(dataset_path)
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"BloomTaxo dataset not found: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        label_counts = Counter([normalize_bloom_label(d.get('label', '')) for d in data])
        for label in BLOOM_LABELS:
            if label_counts.get(label, 0) == 0:
                print(f"Warning: no {label} labels found in {dataset_filename}")
        label_subset_norm = normalize_bloom_label(args.label_subset) if args.label_subset != 'all' else 'all'
        if label_subset_norm != 'all' and label_subset_norm not in BLOOM_LABELS:
            raise ValueError(f"Invalid --label_subset for BloomTaxo: {args.label_subset}")
        if args.Intervention:
            target_label = args.target_label
            if not target_label and label_subset_norm != 'all':
                target_label = label_subset_norm
            target_label_norm = normalize_bloom_label(target_label) if target_label else None
            if target_label_norm not in BLOOM_LABELS:
                raise ValueError("BloomTaxo intervention requires --target_label or a valid --label_subset")
            lang_suffix = f"_{args.lang}" if args.lang else ""
            cache_path = os.path.join(args.hs_cache_dir, 'reasoning_representations_outputs',
                                      f'{model_tag}-base_hs_cache_no_cot_all_bloom_taxo{lang_suffix}.pt')
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(f"Hidden-state cache not found: {cache_path}")


def compute_accuracy(entries, indices=None):
    subset = [entries[i] for i in indices] if indices is not None else entries
    total = len(subset)
    correct = sum(1 for entry in subset if entry.get('model_predict_correctness') == True)
    accuracy = (correct / total) if total else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy}


def normalize_label(label):
    return str(label or "").strip().lower()


def is_reasoning(label):
    return normalize_label(label) == 'reasoning'


def is_memorization(label):
    return normalize_label(label) in ('memory', 'memorization')


def write_metrics(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

if torch.cuda.is_available():
    torch.cuda.manual_seed(8888)
    torch.cuda.manual_seed_all(8888)


torch.set_grad_enabled(False)
tqdm.pandas()


args = parse_args()

model_dir = args.model_dir
model_name = args.model_name
model_tag = model_tag_from_name(model_name)
output_dir = args.output_dir
dataset_dir = args.dataset_dir
dataset_name = args.dataset_name
scale = args.scale
batch_size = args.batch_size
label_subset = args.label_subset
target_label = args.target_label
direction_mode = args.direction_mode
reasonmem_lang = args.lang
lang_suffix = f"_{reasonmem_lang}" if reasonmem_lang else ""
reasonmem_dataset_path, reasonmem_responses_path, _ = reasonmem_paths(dataset_dir, reasonmem_lang)
if dataset_name == 'BloomTaxo':
    label_subset = normalize_bloom_label(label_subset) if label_subset != 'all' else 'all'
    if target_label:
        target_label = normalize_bloom_label(target_label)
    elif label_subset != 'all':
        target_label = label_subset

preflight_checks(args, dataset_name)


print(f"Model Directory: {model_dir}")
print(f"Model Name: {model_name}")
print(f"Output Directory: {output_dir}")
print(f"Dataset Directory: {dataset_dir}")
print(f"Whether to perform Intervention: {args.Intervention}")
print(f"Dataset Name: {dataset_name}")


model_path = join(model_dir, model_name)
if os.path.isdir(model_path):
    model_source = model_path
else:
    model_source = model_name if '/' in model_name else f"meta-llama/{model_name}"

available_gpu_count = torch.cuda.device_count()
if available_gpu_count == 0:
    raise RuntimeError("CUDA devices are required for this script.")

target_gpu_ids = list(range(min(4, available_gpu_count)))

max_memory = {}
for dev_id in target_gpu_ids:
    props = torch.cuda.get_device_properties(dev_id)
    total_gb = props.total_memory // (1024 ** 3)
    usable_gb = max(int(total_gb) - 2, 1)
    max_memory[dev_id] = f"{usable_gb}GiB"
max_memory["cpu"] = "120GiB"

hf_token = os.environ.get("HF_TOKEN")

model_kwargs = dict(
    torch_dtype=torch.float16,
    trust_remote_code=True
)
if len(target_gpu_ids) >= 2:
    model_kwargs["device_map"] = "auto"
    model_kwargs["max_memory"] = max_memory
if hf_token:
    model_kwargs["token"] = hf_token

model = AutoModelForCausalLM.from_pretrained(
    model_source,
    **model_kwargs
)

if len(target_gpu_ids) == 1:
    model.to(f'cuda:{target_gpu_ids[0]}')

# avoid SDPA edge cases with long/left-padded sequences during intervention
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
if hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "eager"

tokenizer_kwargs = dict(trust_remote_code=True)
if hf_token:
    tokenizer_kwargs["token"] = hf_token
tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"


model_layers_num = int(model.config.num_hidden_layers)
mlp_vector_num = int(model.config.intermediate_size)
mlp_dim_num = int(model.config.hidden_size)
layer_name = 'model.layers' 
mlp_name = 'mlp'
mlp_last_layer_name = 'down_proj'
attn_name = 'self_attn'
    

n_new_tokens = 200

# performal normal_evaluation
if not args.Intervention:

    if dataset_name == 'BloomTaxo':
        bloom_dataset_path = bloom_taxo_path(dataset_dir, reasonmem_lang)
        with open(bloom_dataset_path, 'r', encoding='utf-8') as f:
            ds_data = json.load(f)
        for entry in ds_data:
            if 'category' not in entry:
                if entry.get('Subject'):
                    entry['category'] = entry['Subject']
                elif entry.get('Group'):
                    entry['category'] = entry['Group']
                else:
                    entry['category'] = "default"
        if label_subset != 'all':
            ds_data = [entry for entry in ds_data
                       if normalize_bloom_label(entry.get('label')) == label_subset]

        print(f'****Running on {dataset_name} on {model_name} without Intervention')

        prompt_template, prompt_template_no_cot = load_prompt_template(
            ds_name=dataset_name,
            dataset_dir=dataset_dir,
            reasonmem_lang=reasonmem_lang
        )

        evaluation_on_dataset(model=model, tokenizer=tokenizer, val_sampled_data=ds_data,
                              prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot,
                              run_in_fewshot=True, run_in_cot=True, intervention=False,
                              ablation_dir=None, batch_size=batch_size, ds_name=dataset_name, scale=0.1)

        answers_path = build_answers_path(
            args,
            dataset_name,
            label_subset,
            target_label=target_label
        )
        write_answers(answers_path, ds_data)

        metrics_payload = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "label_subset": label_subset,
            "intervention": False,
            "scale": scale,
            "metrics": {
                "overall": compute_accuracy(ds_data)
            }
        }
        write_metrics(args.metrics_out, metrics_payload)
    else:
        #ds_data = load_dataset(ds_name = dataset_name, dataset_dir=dataset_dir, split='test')
        with open(reasonmem_dataset_path, 'r', encoding='utf-8') as f:
            ds_data = json.load(f)
        for entry in ds_data:
            if 'category' not in entry and 'Subject' in entry:
                entry['category'] = entry['Subject']
        if label_subset != 'all':
            if label_subset == 'reasoning':
                ds_data = [entry for entry in ds_data if is_reasoning(entry.get('label'))]
            elif label_subset == 'memorization':
                ds_data = [entry for entry in ds_data if is_memorization(entry.get('label'))]

        print(f'****Running on {dataset_name} on {model_name} without Intervention')

        prompt_template, prompt_template_no_cot = load_prompt_template(
            ds_name=dataset_name,
            dataset_dir=dataset_dir,
            reasonmem_lang=reasonmem_lang
        )


        evaluation_on_dataset(model = model, tokenizer = tokenizer, val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, run_in_fewshot=True, run_in_cot=True, 
                              intervention=False, ablation_dir=None, batch_size=batch_size, ds_name=dataset_name, scale=0.1)
        
        if dataset_name == 'ReasonMem':
            with open(reasonmem_responses_path, 'w', encoding='utf-8') as f:
                json.dump(ds_data, f, ensure_ascii=False, indent=4)

        answers_path = build_answers_path(
            args,
            dataset_name,
            label_subset,
            target_label=target_label
        )
        write_answers(answers_path, ds_data)
        
        if dataset_name != 'ReasonMem':
            compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=False, ds_name=dataset_name)
            metrics_payload = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "intervention": False,
                "scale": scale,
                "metrics": {
                    "overall": compute_accuracy(ds_data)
                }
            }
        else:
            with open(reasonmem_dataset_path, 'r', encoding='utf-8') as f:
                sampled_data = json.load(f)
            for entry in sampled_data:
                if 'category' not in entry and 'Subject' in entry:
                    entry['category'] = entry['Subject']

            reason_indices = [ix for ix, sample in enumerate(sampled_data) if is_reasoning(sample.get('label'))]
            memory_indices = [ix for ix, sample in enumerate(sampled_data) if is_memorization(sample.get('label'))]

            if label_subset == 'all':
                compute_performance_on_reason_memory_subset(val_sampled_data=ds_data, memory_indices=memory_indices, 
                                                    reason_indices=reason_indices, intervention=False)
                metrics_payload = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "label_subset": label_subset,
                    "intervention": False,
                    "scale": scale,
                    "metrics": {
                        "overall": compute_accuracy(ds_data),
                        "reason": compute_accuracy(ds_data, reason_indices),
                        "memory": compute_accuracy(ds_data, memory_indices)
                    }
                }
            elif label_subset == 'reasoning':
                reasoning_metrics = compute_accuracy(ds_data)
                print(f"***Original performance of Reasoning Subset Accuracy: {reasoning_metrics['accuracy']:.4f}")
                metrics_payload = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "label_subset": label_subset,
                    "intervention": False,
                    "scale": scale,
                    "metrics": {
                        "reason": reasoning_metrics
                    }
                }
            else:
                memory_metrics = compute_accuracy(ds_data)
                print(f"***Original performance of Memory Subset Accuracy: {memory_metrics['accuracy']:.4f}")
                metrics_payload = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "label_subset": label_subset,
                    "intervention": False,
                    "scale": scale,
                    "metrics": {
                        "memory": memory_metrics
                    }
                }

        write_metrics(args.metrics_out, metrics_payload)

        

elif args.Intervention:

    save_path = os.path.join(args.hs_cache_dir, 'reasoning_representations_outputs')

    if dataset_name == 'BloomTaxo':
        bloom_cache_path = os.path.join(
            save_path,
            f'{model_tag}-base_hs_cache_no_cot_all_bloom_taxo{lang_suffix}.pt'
        )
        loaded_dict = torch.load(bloom_cache_path)

        candidate_directions = get_candidate_directions_bloom(
            loaded_dict,
            model_layers_num,
            mlp_dim_num,
            target_label,
            direction_mode=direction_mode
        )

        bloom_dataset_path = bloom_taxo_path(dataset_dir, reasonmem_lang)
        with open(bloom_dataset_path, 'r', encoding='utf-8') as f:
            sampled_data = json.load(f)
        for entry in sampled_data:
            if 'category' not in entry:
                if entry.get('Subject'):
                    entry['category'] = entry['Subject']
                elif entry.get('Group'):
                    entry['category'] = entry['Group']
                else:
                    entry['category'] = "default"

        if label_subset == 'all':
            eval_data = list(sampled_data)
        else:
            eval_data = [entry for entry in sampled_data
                         if normalize_bloom_label(entry.get('label')) == label_subset]

        prompt_template, prompt_template_no_cot = load_prompt_template(
            ds_name=dataset_name,
            dataset_dir=dataset_dir,
            reasonmem_lang=reasonmem_lang
        )

        print(f'****Running on {dataset_name} on {model_name} with Features Intervention')

        per_layer_metrics = {}

        if args.layer_start is None and args.layer_end is None:
            layer_range = range(model_layers_num)
        else:
            start = 0 if args.layer_start is None else args.layer_start
            end = (model_layers_num - 1) if args.layer_end is None else args.layer_end
            if start < 0 or end < 0 or start > end or end >= model_layers_num:
                raise ValueError(f"Invalid layer range: start={start} end={end} for {model_layers_num} layers")
            layer_range = range(start, end + 1)

        for layer in layer_range:
            print(f'Doing Intervention in Layer {layer}')
            ablation_dir = candidate_directions[layer]
            ds_data = list(eval_data)

            evaluation_on_dataset(model=model, tokenizer=tokenizer, val_sampled_data=ds_data,
                                  prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot,
                                  ds_name=dataset_name, run_in_fewshot=True, run_in_cot=True,
                                  intervention=True, ablation_dir=ablation_dir, layer_name=layer_name,
                                  attn_name=attn_name, mlp_name=mlp_name, model_layers_num=model_layers_num,
                                  batch_size=batch_size, scale=scale)

            answers_path = build_answers_path(
                args,
                dataset_name,
                label_subset,
                target_label=target_label,
                layer=layer,
                model_layers_num=model_layers_num
            )
            write_answers(answers_path, ds_data)

            overall_metrics = compute_accuracy(ds_data)
            print(
                f"***Intervention Layer {layer} Accuracy: "
                f"{overall_metrics['accuracy']:.4f} "
                f"({overall_metrics['correct']}/{overall_metrics['total']})"
            )
            per_layer_metrics[str(layer)] = {"overall": overall_metrics}

        metrics_payload = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "label_subset": label_subset,
            "target_label": target_label,
            "direction_mode": direction_mode,
            "intervention": True,
            "scale": scale,
            "metrics_by_layer": per_layer_metrics
        }

        write_metrics(args.metrics_out, metrics_payload)
    else:
        # loading ReasonMem to get the direction

        # mmlu_pro_ds = load_dataset(ds_name = dataset_name, dataset_dir=dataset_dir, split='test')
        
        loaded_dict = torch.load(os.path.join(save_path, f'{model_tag}-base_hs_cache_no_cot_all{lang_suffix}.pt'))
        hs_cache_no_cot = loaded_dict['mmlu'] 

        with open(reasonmem_dataset_path, 'r', encoding='utf-8') as f:
              sampled_data = json.load(f)
        for entry in sampled_data:
            if 'category' not in entry and 'Subject' in entry:
                entry['category'] = entry['Subject']

        reason_indices = [ix for ix, sample in enumerate(sampled_data) if is_reasoning(sample.get('label'))]
        memory_indices = [ix for ix, sample in enumerate(sampled_data) if is_memorization(sample.get('label'))]

        candidate_directions = get_candidate_directions(hs_cache_no_cot, model_layers_num, mlp_dim_num, reason_indices, memory_indices)
        if label_subset == 'reasoning':
            eval_data = [entry for entry in sampled_data if is_reasoning(entry.get('label'))]
        elif label_subset == 'memorization':
            eval_data = [entry for entry in sampled_data if is_memorization(entry.get('label'))]
        else:
            eval_data = list(sampled_data)

        prompt_template, prompt_template_no_cot = load_prompt_template(
            ds_name=dataset_name,
            dataset_dir=dataset_dir,
            reasonmem_lang=reasonmem_lang
        )
        if dataset_name != 'ReasonMem':
            ds_data = load_dataset(
                ds_name=dataset_name,
                dataset_dir=dataset_dir,
                split='test',
                reasonmem_lang=reasonmem_lang
            )

        print(f'****Running on {dataset_name} on {model_name} with Features Intervention')

        per_layer_metrics = {}

        if args.layer_start is None and args.layer_end is None:
            layer_range = range(model_layers_num)
        else:
            start = 0 if args.layer_start is None else args.layer_start
            end = (model_layers_num - 1) if args.layer_end is None else args.layer_end
            if start < 0 or end < 0 or start > end or end >= model_layers_num:
                raise ValueError(f"Invalid layer range: start={start} end={end} for {model_layers_num} layers")
            layer_range = range(start, end + 1)

        # Intervention Mode 
        for layer in layer_range:
            
            print(f'Doing Intervention in Layer {layer}')
            ablation_dir = candidate_directions[layer]
            
            if dataset_name != 'ReasonMem':

                evaluation_on_dataset(model = model, tokenizer = tokenizer, val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, ds_name=dataset_name, run_in_fewshot=True, run_in_cot=True, 
                                intervention=True, ablation_dir=ablation_dir, layer_name = layer_name, attn_name = attn_name, mlp_name = mlp_name, model_layers_num = model_layers_num, batch_size=batch_size, scale=scale)

                answers_path = build_answers_path(
                    args,
                    dataset_name,
                    label_subset,
                    target_label=target_label,
                    layer=layer,
                    model_layers_num=model_layers_num
                )
                write_answers(answers_path, ds_data)

                compute_performance_on_reason_subset(val_sampled_data=ds_data, intervention=True, ds_name=dataset_name, intervention_layer=layer)
                overall_metrics = compute_accuracy(ds_data)
                print(
                    f"***Intervention Layer {layer} Accuracy: "
                    f"{overall_metrics['accuracy']:.4f} "
                    f"({overall_metrics['correct']}/{overall_metrics['total']})"
                )
                per_layer_metrics[str(layer)] = {"overall": overall_metrics}
            else:
                
                ds_data = list(eval_data)

                reason_indices = [ix for ix, sample in enumerate(ds_data) if is_reasoning(sample.get('label'))]
                memory_indices = [ix for ix, sample in enumerate(ds_data) if is_memorization(sample.get('label'))]
                
                evaluation_on_dataset(model = model, tokenizer = tokenizer, val_sampled_data=ds_data, prompts_cot=prompt_template, prompts_no_cot=prompt_template_no_cot, ds_name=dataset_name, run_in_fewshot=True, run_in_cot=True, 
                                intervention=True, ablation_dir=ablation_dir, layer_name = layer_name, attn_name = attn_name, mlp_name = mlp_name, model_layers_num = model_layers_num, batch_size=batch_size, scale=scale)

                answers_path = build_answers_path(
                    args,
                    dataset_name,
                    label_subset,
                    target_label=target_label,
                    layer=layer,
                    model_layers_num=model_layers_num
                )
                write_answers(answers_path, ds_data)

                if label_subset == 'all':
                    compute_performance_on_reason_memory_subset(val_sampled_data=ds_data, memory_indices=memory_indices, 
                                                    reason_indices=reason_indices, intervention=True, intervention_layer=layer)
                    overall_metrics = compute_accuracy(ds_data)
                    reason_metrics = compute_accuracy(ds_data, reason_indices)
                    memory_metrics = compute_accuracy(ds_data, memory_indices)
                    print(
                        f"***Intervention Layer {layer} Accuracy: "
                        f"{overall_metrics['accuracy']:.4f} "
                        f"({overall_metrics['correct']}/{overall_metrics['total']})"
                    )
                    print(
                        f"***Intervention Layer {layer} Reason Subset Accuracy: "
                        f"{reason_metrics['accuracy']:.4f} "
                        f"({reason_metrics['correct']}/{reason_metrics['total']})"
                    )
                    print(
                        f"***Intervention Layer {layer} Memory Subset Accuracy: "
                        f"{memory_metrics['accuracy']:.4f} "
                        f"({memory_metrics['correct']}/{memory_metrics['total']})"
                    )
                    per_layer_metrics[str(layer)] = {
                        "overall": overall_metrics,
                        "reason": reason_metrics,
                        "memory": memory_metrics
                    }
                elif label_subset == 'reasoning':
                    reasoning_metrics = compute_accuracy(ds_data)
                    print(
                        f"***Intervention in Layer {layer}, Reason Subset Accuracy: "
                        f"{reasoning_metrics['accuracy']:.4f} "
                        f"({reasoning_metrics['correct']}/{reasoning_metrics['total']})"
                    )
                    per_layer_metrics[str(layer)] = {
                        "reason": reasoning_metrics
                    }
                else:
                    memory_metrics = compute_accuracy(ds_data)
                    print(
                        f"***Intervention in Layer {layer}, Memory Subset Accuracy: "
                        f"{memory_metrics['accuracy']:.4f} "
                        f"({memory_metrics['correct']}/{memory_metrics['total']})"
                    )
                    per_layer_metrics[str(layer)] = {
                        "memory": memory_metrics
                    }

        metrics_payload = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "label_subset": label_subset,
            "intervention": True,
            "scale": scale,
            "metrics_by_layer": per_layer_metrics
        }

        write_metrics(args.metrics_out, metrics_payload)
            



# model_output_dir = os.path.join(output_dir, model_name)  
# os.makedirs(model_output_dir, exist_ok=True)

# with open(os.path.join(model_output_dir, f'{model_name}_model_mute_performance.json'), 'w', encoding='utf-8') as jsonfile:
#     json.dump(results, jsonfile, indent=4, ensure_ascii=False)

# with open(os.path.join(model_output_dir, f'{model_name}_model_mute_performance_per_concept.json'), 'w', encoding='utf-8') as jsonfile:
#     json.dump(ac_per_concept_results, jsonfile, indent=4, ensure_ascii=False)
