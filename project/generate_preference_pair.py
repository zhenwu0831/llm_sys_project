import os
import numpy as np
import pandas as pd
import json
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse

# Configurations
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
os.environ.setdefault("WANDB__SERVICE_WAIT", "6000")
RANDOM_SEED = 1234
MODEL_NAME = "lvwerra/gpt2-imdb"
BATCH_SIZE = 1
REWARD_MODEL = "siebert/sentiment-roberta-large-english"
DEVICE = "cuda"
set_seed(RANDOM_SEED)

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# config for generation
gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 20}

# Pipeline for reward calculation
reward_pipe = pipeline("text-classification", model=REWARD_MODEL)
reward_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

def length_sampler(min_length, max_length):
    return np.random.choice(range(min_length, max_length + 1))

def tokenize(sample, input_size):
    sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

def build_dataset(dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    ds = load_dataset(dataset_name, split="train").rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200)
    input_size = length_sampler(input_min_text_length, input_max_text_length)
    ds = ds.map(lambda x: tokenize(x, input_size), batched=False)
    ds.set_format(type="torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

def outputs_and_rewards(query, query_tensor, temperature, model=model, tokenizer=tokenizer):
    output = model.generate(query_tensor, temperature=temperature, **gen_kwargs).squeeze()[-gen_kwargs["max_new_tokens"]:]
    response = tokenizer.decode(output)

    texts = [q + r for q, r in zip(query, response)]
    for output in reward_pipe(texts, **reward_kwargs):
        if output[1]["label"] == "POSITIVE":
            rewards = output[1]["score"]
        else:
            assert output[1]["label"] == "NEGATIVE"
            rewards = output[0]["score"]
            
    return response, rewards

def process_data(dataloader):
    results = []
    for batch in dataloader:
        query = batch["query"][0]
        query_tensor = batch["input_ids"].to(DEVICE)

        data_record = {'query': query}
        for i, temp in enumerate([0.2, 0.4, 0.6, 0.8], 1):
            response, rewards = outputs_and_rewards(query, query_tensor, temp)
            data_record[f'response{i}'] = response
            data_record[f'reward{i}'] = rewards

        results.append(data_record)

    return pd.DataFrame(results)

def create_output_json(data, output_file='data/imdb.json'):
    final = []
    for idx, row in data.iterrows():
        responses = [row[f"response{i}"] for i in range(1, 5)]
        rewards = [row[f"reward{i}"] for i in range(1, 5)]
        indices = sorted(range(4), key=lambda k: rewards[k])

        for i in indices:
            for j in indices[i+1:]:
                final.append({"prompt": row["query"], "chosen": responses[i], "rejected": responses[j]})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=4)

def main(args):
    dataloader = build_dataset()
    data = process_data(dataloader)
    data.to_csv(args.output_file, index=False)
    create_output_json(data, args.output_file.replace('.csv', '.json'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="data/imdb.csv")
    args = parser.parse_args()
    main(args)