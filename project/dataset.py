import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
# from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import os
import transformers

import minitorch
from minitorch import DecoderLM
from minitorch.tensor_functions import *
from minitorch.nn import *
from minitorch.cuda_kernel_ops import CudaKernelOps

import time


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

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    # assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    # assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    if tokenizer.eos_token_id not in chosen_tokens['input_ids']:
        chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
        chosen_tokens['attention_mask'].append(1)
    # chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    # chosen_tokens['attention_mask'].append(1)

    if tokenizer.eos_token_id not in rejected_tokens['input_ids']:
        rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
        rejected_tokens['attention_mask'].append(1)

    # rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    # rejected_tokens['attention_mask'].append(1)

    if len(chosen_tokens['input_ids']) - len(rejected_tokens['input_ids']) > 0:
        longer_response_length = len(chosen_tokens['input_ids'])

    else:
        longer_response_length = len(rejected_tokens['input_ids'])

    # longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [0] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [0] * len(prompt_tokens['input_ids'])

    # print(chosen_sequence_tokens['labels'])
    
    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch

def collate_batch(
        examples, tokenizer, model_max_length, backend=None):
    """
    Prepares a batch of examples for model training or evaluation by tokenizing and padding them.

    Parameters:
    - examples: A list of examples to be processed.
    - src_key: The key for accessing source texts in the examples.
    - tgt_key: The key for accessing target texts in the examples.
    - tokenizer: The tokenizer to be used for encoding the texts.
    - model_max_length: The maximum sequence length the model can handle.
    - backend: The backend of minitorch tensors.

    Returns:
    - A dictionary containing keys: 'input_ids', 'labels', 'label_token_weights',
        each indicates a minitorch tensor with shape (len(examples), model_max_length).

    Notes:
    ["input_ids"] for every example in the DE-EN translation, the "input_ids" will be:
        <de_token_ids> + <de_eos_id> + <en_token_ids> + <en_eos_id> + <pad_ids>
    where the pad_ids makes the length of input_ids to be model_max_length.

    ["labels"]: the next tokens to be predicted, which will be used in the cross-entropy
    loss function, e.g., for an example tokenized as [a, b, c, d], "input_ids" and "labels" 
    can be [a, b, c] and [b, c, d], respectively.

    ["label_token_weights"] The 'label_token_weights' are used to differentiate
    calculation purposes. (the MLE loss is computed on target tokens only.)
    between the source (weight = 0) and target (weight = 1) tokens for loss

    TODO: 
        outputs: [chosen token ids, prompt token ids, rejected token ids]
    
    """
    token_ids_chosen, token_masks_chosen, token_ids_rejected, token_masks_rejected = [], [], [], []
    label_ids_chosen, label_ids_rejected = [], []
    # pad_token_id = tokenizer.vocab['<pad>']
    pad_token_id = tokenizer.encode('<pad>')[1]
    # print(pad_token_id)
    for i in range(len(examples['prompt'])):
        # print('example: ', example)
        batch = tokenize_batch_element(examples['prompt'][i], examples['chosen'][i], examples['rejected'][i], 'keep_start', tokenizer, 512, 256)

        # ------------ToDo------------
        # token_ids_chosen = tokenizer(
        #    f'{example[chosen_key]}<eos_{chosen_key}>')['input_ids']
        # ------------ToDo------------

        # BEGIN ASSIGN2_2
        # TODO
        # create token_ids, labels, and label_token_weights for every example
        # hint: based on token_ids_src, token_ids_tgt, and pad_token_id
        
        # input_ids is <de_token_ids> + <de_eos_id> + <en_token_ids> + <en_eos_id> + <pad_ids> where 
        # the pad_ids makes the length of input_ids to be model_max_length

        # total_len = len(token_ids_src) + len(token_ids_tgt)
        # token_pad_length = model_max_length - total_len

        total_len_chosen = len(batch['chosen_input_ids'])
        token_pad_length_chosen = model_max_length - total_len_chosen
        
        
        
        total_len_rejected = len(batch['rejected_input_ids'])
        token_pad_length_rejected = model_max_length - total_len_rejected
        
        # pad 
        token_id_chosen = batch['chosen_input_ids'] + [pad_token_id] * token_pad_length_chosen
        # print('token_id_chosen: ', len(batch['chosen_input_ids']))
        token_mask_chosen = [1] * len(batch['chosen_input_ids']) + [0] * token_pad_length_chosen
        label_id_chosen = batch['chosen_labels'] + [-0] * token_pad_length_chosen
        token_id_chosen = token_id_chosen[:model_max_length]
        token_mask_chosen = token_mask_chosen[:model_max_length]
        
        token_ids_chosen.append(token_id_chosen)
        token_masks_chosen.append(token_mask_chosen)
        label_ids_chosen.append(label_id_chosen)
        
        token_id_rejected = batch['rejected_input_ids'] + [pad_token_id] * token_pad_length_rejected
        token_mask_rejected = [1] * len(batch['rejected_input_ids']) + [0] * token_pad_length_rejected
        label_id_rejected = batch['rejected_labels'] + [0] * token_pad_length_rejected
        token_id_rejected = token_id_rejected[:model_max_length]
        token_mask_rejected = token_mask_rejected[:model_max_length]
        
        token_ids_rejected.append(token_id_rejected)
        token_masks_rejected.append(token_mask_rejected)
        label_ids_rejected.append(label_id_rejected)
        
        assert len(token_id_chosen) == model_max_length, print(len(token_id_chosen), total_len_chosen, token_pad_length_chosen)
    

    return {
        'chosen_input_ids': minitorch.tensor_functions.tensor_from_numpy(np.array(token_ids_chosen), backend),
        # 'chosen_masks': minitorch.tensor_functions.tensor_from_numpy(np.array(token_masks_chosen), backend),
        # 'chosen_labels': minitorch.tensor_functions.tensor_from_numpy(np.array(label_ids_chosen), backend),
        # 'rejected_input_ids': minitorch.tensor_functions.tensor_from_numpy(np.array(token_ids_rejected), backend),
        # 'rejected_masks': minitorch.tensor_functions.tensor_from_numpy(np.array(token_masks_rejected), backend),
        # 'rejected_labels': minitorch.tensor_functions.tensor_from_numpy(np.array(label_ids_rejected), backend),
        'concatenated_input_ids': minitorch.tensor_functions.tensor_from_numpy(np.concatenate((np.array(token_ids_chosen), np.array(token_ids_rejected))), backend),
        'concatenated_masks': minitorch.tensor_functions.tensor_from_numpy(np.concatenate((np.array(token_masks_chosen), np.array(token_masks_rejected))), backend),
        'concatenated_labels': minitorch.tensor_functions.tensor_from_numpy(np.concatenate((np.array(label_ids_chosen), np.array(label_ids_rejected))), backend)
    }

# def concatenated(batch):
#     concatenated_batch = {}
#     concatenated_batch['input_ids'] = 

# def numpy_gather(arr, dim, index):
#     """
#     Mimic torch.gather using numpy.

#     Parameters:
#         arr (numpy.ndarray): The source array from which to gather values.
#         dim (int): The dimension along which to gather values.
#         index (numpy.ndarray): The indices of elements to gather.

#     Returns:
#         numpy.ndarray: The gathered array.
        
#     TODO: check
#     """
#     # Check if the index array dimensions match the source array along the specified dimension
#     if index.shape != arr.shape:
#         raise ValueError("The shape of the index must match the shape of the source array.")

#     # Transpose the array so that we gather along the first dimension
#     if dim != 0:
#         axes = np.arange(len(arr.shape))
#         axes[0], axes[dim] = axes[dim], axes[0]
#         arr = arr.transpose(axes)
#         index = index.transpose(axes)

#     # Create an array of indices that selects data along the gathered dimension
#     idx = np.ogrid[tuple(slice(i) for i in index.shape)]

#     # Replace the index array for the dimension we are gathering from
#     idx[0] = index

#     # Gather the data
#     result = arr[tuple(idx)]

#     # Transpose back if necessary
#     if dim != 0:
#         result = result.transpose(axes)

#     return result
def numpy_gather(arr, dim, index):
    """
    Mimic torch.gather using numpy, allowing index dimension size of 1 for broadcasting.

    Parameters:
        arr (numpy.ndarray): The source array from which to gather values.
        dim (int): The dimension along which to gather values.
        index (numpy.ndarray): The indices of elements to gather.

    Returns:
        numpy.ndarray: The gathered array.
    """
    # Check if the index array can be broadcasted to the shape of the source array
    if not all((s1 == s2 or s1 == 1 or s2 == 1) for s1, s2 in zip(arr.shape, index.shape)):
        raise ValueError("Index cannot be broadcasted to match the array shape.")

    # Broadcast index if necessary
    if index.shape[dim] == 1:
        index = np.broadcast_to(index, arr.shape)

    # Transpose the array so that we gather along the first dimension
    if dim != 0:
        axes = np.arange(len(arr.shape))
        axes[0], axes[dim] = axes[dim], axes[0]
        arr = arr.transpose(axes)
        index = index.transpose(axes)

    # Create an array of indices that selects data along the gathered dimension
    idx = np.ogrid[tuple(slice(i) for i in index.shape)]

    # Replace the index array for the dimension we are gathering from
    idx[0] = index

    # Gather the data
    result = arr[tuple(idx)]

    # Transpose back if necessary
    if dim != 0:
        result = result.transpose(axes)

    return result

def loss_fn(batch, model):
    """
    The MLE loss for a batch.

    Parameters:
    - batch: The result of collate_fn, a dict with "input_ids", "labels", and "label_token_weights".
    - model: The model to be trained.

    Returns:
    - A scalar loss value for this batch, averaged across all target tokens.

    # ------------ToDo------------
    add preference loss
    add preference model
    # ------------ToDo------------
    """

    idx = batch['concatenated_input_ids']
    idx.requires_grad_(True)
    
    logits = model(idx=idx)
    
    # chosen_logits = logits[:len(batch['chosen_input_ids'])]
    # rejected_logits = logits[len(batch['chosen_input_ids']):]
    
    labels = batch['concatenated_labels']

    loss_mask = labels != 0

  
    batch_size, seq_len, vocab_size = logits.shape
    
    
    print('logits shape: ', logsoftmax(logits, -1).shape)
    print('labels shape: ', labels.shape)

    logits = logits.to_numpy()
    labels = labels.to_numpy()
    logits = torch.tensor(logits)
    labels = torch.tensor(labels, dtype=torch.int64)
    
    # per_token_logps = numpy_gather(logsoftmax(logits, -1).detach().to_numpy(), 2, labels.view(batch_size, seq_len, 1).to_numpy())
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    print(per_token_logps.shape)
    
    batch_logps = (per_token_logps * loss_mask).sum(-1)

    print('batch_logps: ', batch_logps.shape)
    
    # BEGIN ASSIGN2_2
    # TODO
    # compute the MLE loss based on logits obtained by the model.
    # hint: using the function minitorch.nn.softmax_loss

    # labels = batch['labels']
    # label_token_weights = batch['label_token_weights']

    # ------------ToDo------------
    chosen_length = batch['chosen_input_ids'].shape[1]
    loss, chosen_rewards, rejected_rewards = minitorch.nn.preference_loss(batch_logps[:chosen_length], batch_logps[chosen_length:], beta=0.7, reference_free=False)
    # ------------ToDo------------
    # loss = minitorch.nn.softmax_loss(logits.view(batch_size * seq_len, vocab_size), labels.view(batch_size * seq_len,))

    loss = loss.view(batch_size, seq_len)

    # weighted_losses = loss * label_token_weights

    # non_zero_weights = label_token_weights.sum()
    
    # average_loss = weighted_losses.sum() / non_zero_weights if non_zero_weights > 0 else 0

    return loss, chosen_rewards, rejected_rewards


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    """
    Trains the model on the provided examples.

    Parameters:
    - model: The model to be trained.
    - optimizer: The optimizer used for updating the model's parameters.
    - examples: The dataset examples used for training.
    - n_samples: The random samples to train from "examples".
    - collate_fn: The function to collate data examples into batches.
    - batch_size: The number of examples in each batch.
    - desc: Description for the training process (used in progress bars).

    # ------------ToDo------------
    add preference policy model
    # ------------ToDo------------
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        t1 = time.time()

        loss.backward()
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        print(f"Forward: {t1 - t0}")
        print(f"Backward: {t2 - t1}")
        print(f"Opt.step: {t3 - t2}")

        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
            loss=loss.item(),
            lr=optimizer.lr)

if __name__ == '__main__':
    dataset = get_imdb()
    # print(dataset['train'][0]['prompt'])
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-large', special_tokens={'pad_token': '<pad>'})
    # tokenizer.pad_token = '<pad>'
    # print(tokenizer.eos_token_id)
    batch = tokenize_batch_element(dataset['train'][0]['prompt'], dataset['train'][0]['chosen'], dataset['train'][0]['rejected'], 'keep_start', tokenizer, 512, 256)
    print('chosen_input_ids: ', batch['chosen_input_ids'])
    
    print('chosen: ', batch)
    
    # print('dataset: ', dataset['train'][:10])
    backend = minitorch.TensorBackend(CudaKernelOps)
    batch = collate_batch(dataset['train'][:10], tokenizer=tokenizer, model_max_length=512, backend=backend)
    
    config = {
        'n_vocab': 50257,  # vocab_size
        'n_embd': 256,  # n_embed
        'n_head': 8,  # n_head
        'n_positions': 512,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout': 0.1,  # x_pdrop
        'ln_eps': 1e-5,  # layer_norm_epsilon
        'backend': backend
    }
    model = DecoderLM(**config)
    loss, _, _ = loss_fn(batch, model)
    # a = torch.randn(3,5)
    # print('a', a)
    # index = torch.tensor([[0, 0, 0, 1, 1], [1, 0, 0, 1, 1], [2, 0, 0, 1, 1]])
    # out1 = torch.gather(a, 0, index)
    # out2 = numpy_gather(a.numpy(), 0, index.numpy())
    # print('out1', out1)
    # print('out2', out2)
    