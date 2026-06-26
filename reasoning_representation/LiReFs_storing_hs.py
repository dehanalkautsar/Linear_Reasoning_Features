# %%
import torch
import argparse

# 检查CUDA是否可用
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Find {gpu_count} GPU can be used.")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i + 1}: {gpu_name}")
else:
    print("No GPU can be used.")

# %%
import copy
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
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

random.seed(8888)
torch.manual_seed(8888)
random.seed(8888)
np.random.seed(8888)

if torch.cuda.is_available():
    torch.cuda.manual_seed(8888)
    torch.cuda.manual_seed_all(8888)


from tqdm import tqdm

torch.set_grad_enabled(False)
tqdm.pandas()

from transformers import AutoModelForCausalLM, AutoTokenizer
import json

raw_model_name = os.environ.get("MODEL_NAME", "Meta-Llama-3-8B-Instruct")
local_model_root = os.environ.get("MODEL_DIR", "/mnt/workspace/workgroup/yhhong/transformers")
if os.path.isdir(raw_model_name):
    model_source = raw_model_name
    model_tag = os.path.basename(raw_model_name.rstrip('/'))
else:
    model_repo_id = raw_model_name if '/' in raw_model_name else f"meta-llama/{raw_model_name}"
    local_model_candidate = os.path.join(local_model_root, os.path.basename(model_repo_id))
    model_source = local_model_candidate if os.path.isdir(local_model_candidate) else model_repo_id
    model_tag = os.path.basename(model_source.rstrip('/'))
output_dir = '/mnt/workspace/Knowledge_Concentration/outputs'
dataset_dir = '/mnt/workspace/Knowledge_Concentration/dataset'

# determine how many GPUs to use (prefer four for sharding)
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
);

if len(target_gpu_ids) == 1:
    model.to(f'cuda:{target_gpu_ids[0]}')

tokenizer_kwargs = dict(trust_remote_code=True)
if hf_token:
    tokenizer_kwargs["token"] = hf_token
tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"


def infer_primary_device(loaded_model, default_device='cuda:0'):
    if hasattr(loaded_model, 'hf_device_map'):
        embed_key = next((k for k in loaded_model.hf_device_map.keys() if 'embed_tokens' in k or 'wte' in k), None)
        if embed_key:
            return loaded_model.hf_device_map[embed_key]
        first_device = next(iter(loaded_model.hf_device_map.values()))
        return first_device if isinstance(first_device, str) else default_device
    return getattr(loaded_model, 'device', default_device)


primary_device = torch.device(infer_primary_device(model, default_device=f'cuda:{target_gpu_ids[0]}'))

# %%
print('model: ',model)
print('model.config: ',model.config)
print('model.config.model_type.lower(): ',model.config.model_type.lower())  # Often provides a string identifier

model_layers_num = int(model.config.num_hidden_layers)
mlp_vector_num = int(model.config.intermediate_size)
mlp_dim_num = int(model.config.hidden_size)
layer_name = 'model.layers' 
mlp_name = 'mlp'
mlp_last_layer_name = 'down_proj'
attn_name = 'self_attn'
    
    
    

# %%
import datasets
import json
import re
import random
from datasets import load_from_disk
from tqdm import tqdm

n_new_tokens = 100
NUll_num = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Store hidden states for ReasonMem.")
    parser.add_argument("--lang", type=str, default="en", help="ReasonMem language suffix, e.g. en, ar, ch, ind")
    parser.add_argument("--dataset", type=str, default="reason_mem",
                        choices=("reason_mem", "bloom_taxo"),
                        help="Dataset for hidden-state caching: reason_mem or bloom_taxo")
    return parser.parse_args()

args = parse_args()
reasonmem_lang = args.lang
dataset_key = args.dataset
lang_suffix = f"_{reasonmem_lang}" if reasonmem_lang else ""

DATASET_CONFIG = {
    "reason_mem": {
        "file_template": "reason_mem_labels_mmlu_{lang}_validation.json",
        "lang_map": {},
        "output_key": "mmlu",
    },
    "bloom_taxo": {
        "file_template": "bloom_tax_labels_{lang}_validation.json",
        "lang_map": {"ind": "indo"},
        "output_key": "bloom_taxo",
    },
}

dataset_cfg = DATASET_CONFIG[dataset_key]
output_key = dataset_cfg["output_key"]
dataset_lang = dataset_cfg["lang_map"].get(reasonmem_lang, reasonmem_lang)
dataset_path = os.path.join(os.path.dirname(__file__),
                            dataset_cfg["file_template"].format(lang=dataset_lang))
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

def clear_cuda_cache(device_ids):
    if not device_ids:
        return
    for dev_id in device_ids:
        with torch.cuda.device(dev_id):
            torch.cuda.empty_cache()

def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def get_prediction(output):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)
    if match:
        #print('prediction success: ',match.group(1))
        return match.group(1)
    else:
        #print("extraction failed, do a random guess")
        global NUll_num  
        NUll_num += 1
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])


def generate_outputs(questions):
    
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to(primary_device)
    input_length = inputs.input_ids.size(1)
    output = model(**inputs, output_hidden_states = True)
#     Question_input = [[{"role": "user", "content": prompt}] for prompt in questions]
#     texts = tokenizer.apply_chat_template(Question_input ,tokenize=False)

#     inputs = tokenizer(texts, padding="longest", return_tensors="pt")
#     inputs = {key: val.cuda() for key, val in inputs.items()}
#     output = model(**inputs, output_hidden_states = True)
    
    return output

def generate_questions(questions):
    
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to(primary_device)
    input_length = inputs.input_ids.size(1)
    gen_tokens = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=False)

    gen_text = tokenizer.batch_decode(gen_tokens[:, input_length:], skip_special_tokens=True)
    
    return gen_text


with open(dataset_path, 'r', encoding='utf-8') as f:
    full_dataset = json.load(f)

categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
              'health', 'physics', 'business', 'philosophy', 'economics', 'other',
              'psychology', 'history']

per_category_accuracy = {c: [0, 0] for c in categories}
success, fail = 0, 0
answers = []

print('----------------- Start Answering -------------------')
queries_batch = []  # 可以测试一下batch or single哪种方式准确率更高，更合适一些 #发现基本是一样的，padding不会对准确率造成影响
entry_batch = []

batch_size = 30

random.seed(8888)

# run on the entire dataset (shuffle for randomness but keep all samples)
sampled_data = list(full_dataset)
random.shuffle(sampled_data)
total_samples = len(sampled_data)

layers_to_cache = list(range(model_layers_num))
print('layers_to_cache: ',layers_to_cache)
# hs_cache_cot = {}
# hs_cache_no_cot = {}
hs_cache_cot = {layer: [] for layer in layers_to_cache}
hs_cache_no_cot = {layer: [] for layer in layers_to_cache}

print('----------------- Running no cot Inference -------------------')
for ix, entry in tqdm(enumerate(sampled_data), total=total_samples):
        
    question_text = entry['question']
    if entry.get('options'):
        question_text += "\n" + form_options(entry['options'])
    
    query = 'Q: ' + question_text + "\nA: "
    queries_batch.append(query)
    
    if len(queries_batch) == batch_size or ix == total_samples - 1:
        with torch.no_grad():
            output = generate_outputs(queries_batch)
        
        for layer in layers_to_cache:
            # if layer not in hs_cache_no_cot:
            #     hs_cache_no_cot[layer] = output["hidden_states"][layer][: ,-1 , :].cpu() #bs * tok * dims
            # else:
            #     hs_cache_no_cot[layer] = torch.cat((hs_cache_no_cot[layer], output["hidden_states"][layer][: ,-1 , :].cpu()), dim=0)
            hs = output["hidden_states"][layer][:, -1, :]
            hs_cache_no_cot[layer].append(hs.detach().cpu())
        
        del output, hs        
        queries_batch = []
        clear_cuda_cache(target_gpu_ids)
    
# concatenate once
for layer in layers_to_cache:
    hs_cache_no_cot[layer] = torch.cat(hs_cache_no_cot[layer], dim=0)

# %% [markdown]
# # **PCA**

# %%
def normalize_label(label):
    return str(label or "").strip().lower()

if dataset_key == "reason_mem":
    reason_indices = [ix for ix, sample in enumerate(sampled_data)
                      if normalize_label(sample.get('label')) == 'reasoning']
    memory_indices = [ix for ix, sample in enumerate(sampled_data)
                      if normalize_label(sample.get('label')) == 'memory']
else:
    bloom_labels = ("remember", "understand", "apply", "analyze", "evaluate")
    bloom_aliases = {
        "analyse": "analyze",
        "analysis": "analyze",
    }
    bloom_label_indices = {label: [] for label in bloom_labels}
    for ix, sample in enumerate(sampled_data):
        label = normalize_label(sample.get('label'))
        label = bloom_aliases.get(label, label)
        if label in bloom_label_indices:
            bloom_label_indices[label].append(ix)

# %%
# skip other datasets unless explicitly enabled, to avoid missing local paths
RUN_OTHER_DATASETS = False

if RUN_OTHER_DATASETS:
    # loading and running gsm8k or other dataset
    gsm8k_ds_main = load_from_disk('/mnt/workspace/Interp_Reasoning/dataset/gsm8k/main') 
    gsm8k_ds_main_test = list(gsm8k_ds_main['test'])

    # mbpp_ds_full = load_from_disk('/mnt/workspace/Interp_Reasoning/dataset/mbpp/full')
    # mbpp_ds_full_val = list(mbpp_ds_full['validation'])

    # mbpp_ds_full = load_from_disk('/mnt/workspace/Interp_Reasoning/dataset/mbpp/full')
    # mbpp_ds_full_test = list(mbpp_ds_full['test'])

    # example on MGSM, feel free to add other categories in MGSM
    mgsm_zh = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/mgsm/mgsm_zh.tsv', sep='\t')
    mgsm_zh_test = mgsm_zh.values.tolist()
    mgsm_de = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/mgsm/mgsm_de.tsv', sep='\t')
    mgsm_de_test = mgsm_de.values.tolist()
    mgsm_bn = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/mgsm/mgsm_bn.tsv', sep='\t')
    mgsm_bn_test = mgsm_bn.values.tolist()
    mgsm_ja = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/mgsm/mgsm_ja.tsv', sep='\t')
    mgsm_ja_test = mgsm_ja.values.tolist()
    mgsm_te = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/mgsm/mgsm_te.tsv', sep='\t')
    mgsm_te_test = mgsm_te.values.tolist()

    # example on C-Eval, feel free to add other categories in C-Eval
    ceval_chi = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/ceval-exam/test/chinese_language_and_literature_test.csv')['question']
    ceval_chi_test = ceval_chi.tolist()
    ceval_his = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/ceval-exam/test/high_school_history_test.csv')['question']
    ceval_his_test = ceval_his.tolist()
    ceval_pol = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/ceval-exam/test/high_school_politics_test.csv')['question']
    ceval_pol_test = ceval_pol.tolist()
    ceval_mar = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/ceval-exam/test/marxism_test.csv')['question']
    ceval_mar_test = ceval_mar.tolist()
    ceval_bus = pd.read_csv('/mnt/workspace/Interp_Reasoning/dataset/ceval-exam/test/business_administration_test.csv')['question']
    ceval_bus_test = ceval_bus.tolist()

    popqa_test = load_from_disk('/mnt/workspace/Interp_Reasoning/dataset/PopQA/test') 
    popqa_test = list(popqa_test)

    other_running_set_name_list = ['ceval_liberal', 'gsm8k', 'mgsm', 'popqa'] # mbpp,, hoppingtoolate， , 'mbpp', 'popqa',
    # other_running_set_name_list = ['mbpp']
    other_dataset = None

    hs_cache_no_cot_other_all = {}

    for other_running_set_name in other_running_set_name_list:
        
        if other_running_set_name == 'mbpp':
            other_dataset = mbpp_ds_full_test
        elif other_running_set_name == 'gsm8k':
            other_dataset = gsm8k_ds_main_test
        elif other_running_set_name == 'mgsm': #multilingual gsm8k
            other_dataset = mgsm_zh_test + mgsm_de_test + mgsm_bn_test + mgsm_ja_test + mgsm_te_test
        elif other_running_set_name == 'ceval_liberal':
            other_dataset = ceval_chi_test + ceval_his_test + ceval_pol_test + ceval_mar_test # + ceval_bus_test
        elif other_running_set_name == 'popqa': #multilingual gsm8k
            other_dataset = popqa_test

        print(f'#####Running on {other_running_set_name} test set')
        print(f'The size is {len(other_dataset)}')

        layers_to_cache_other = list(range(model_layers_num))
        print('layers_to_cache_other: ',layers_to_cache_other)
        hs_cache_no_cot_other = {}
        queries_batch = []
        batch_size = 30

        for ix, entry in tqdm(enumerate(other_dataset)):

            if other_running_set_name == 'gsm8k':
                query = 'Q: ' + entry['question'] + "\nA: "
            elif other_running_set_name == 'mbpp':
                query = 'Q: ' + entry['text'] + "\nA: "
            elif other_running_set_name == 'mgsm':
                query = 'Q: ' + entry[0] + "\nA: "
            elif other_running_set_name == 'ceval_liberal':
                query = 'Q: ' + entry + "\nA: "
            elif other_running_set_name == 'popqa':
                query = 'Q: ' + entry['question'] + "\nA: "

            queries_batch.append(query)

            if len(queries_batch) == batch_size or ix == len(other_dataset) - 1:
                output = generate_outputs(queries_batch)

                for layer in layers_to_cache_other:
                    if layer not in hs_cache_no_cot_other:
                        hs_cache_no_cot_other[layer] = output["hidden_states"][layer][: ,-1 , :].cpu() #bs * tok * dims
                    else:
                        hs_cache_no_cot_other[layer] = torch.cat((hs_cache_no_cot_other[layer], output["hidden_states"][layer][: ,-1 , :].cpu()), dim=0)

                queries_batch = []
            clear_cuda_cache(target_gpu_ids)

        hs_cache_no_cot_other_all[other_running_set_name] = hs_cache_no_cot_other

# %%
save_path = os.path.join(os.path.dirname(__file__), 'reasoning_representations_outputs')
os.makedirs(save_path, exist_ok=True)

dataset_suffix = "" if dataset_key == "reason_mem" else f"_{dataset_key}"
output_path = os.path.join(
    save_path,
    f'{model_tag}-base_hs_cache_no_cot_all{dataset_suffix}{lang_suffix}.pt'
)
if RUN_OTHER_DATASETS:
    torch.save(hs_cache_no_cot_other_all, output_path)
else:
    if dataset_key == "bloom_taxo":
        def slice_cache(cache, indices):
            if not indices:
                return {layer: cache[layer][:0] for layer in cache}
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            return {layer: cache[layer].index_select(0, idx_tensor) for layer in cache}

        bloom_label_caches = {
            label: slice_cache(hs_cache_no_cot, indices)
            for label, indices in bloom_label_indices.items()
        }
        torch.save(bloom_label_caches, output_path)
    else:
        torch.save({output_key: hs_cache_no_cot}, output_path)
print(f"Saved hidden states to: {output_path}")
