import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import os


def get_imdb(data_path: str = 'data/imdb.json', split: str = None, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading IMDB RLHF dataset...')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(os.path.dirname(current_dir), data_path)
    # dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    dataset = datasets.load_dataset("json", data_files=json_path)
    print('done')
    # data = defaultdict(lambda: defaultdict(list))
    # for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
    #     # prompt, chosen, rejected = split_prompt_and_responses(row)
    #     prompt = row['Prompt']
    #     chosen = row['positive_response'][len(prompt[:-1]):]
    #     rejected = row['negative_response'][len(prompt[:-1]):]
    #     responses = [chosen, rejected]
    #     n_responses = len(data[prompt]['responses'])
    #     data[prompt]['pairs'].append((n_responses, n_responses + 1))
    #     data[prompt]['responses'].extend(responses)
    #     data[prompt]['sft_target'] = chosen

    return dataset

if __name__ == '__main__':
    dataset = get_imdb()
    print(dataset['train'][0]['prompt'])