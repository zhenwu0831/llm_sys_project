from functools import partial
import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
# import tqdm
from tqdm.notebook import tqdm, trange
import random
# from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from sacrebleu.metrics import BLEU
import os
import transformers
import sys
import json
import wandb

import minitorch
from minitorch import DecoderLM
from minitorch.tensor import *
from minitorch.tensor_functions import *
from minitorch.nn import *
from minitorch.cuda_kernel_ops import CudaKernelOps

import time

def get_imdb(data_path: str = 'data/imdb.json', split: str = None, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading IMDB RLHF dataset...')
    dataset = datasets.load_dataset("json", data_files=f"/home/jiaxins1/11868/llm_sys_project/{data_path}")
    train_testvalid = dataset['train'].train_test_split(test_size=0.2)

    # Split the remaining part into test and validation equally
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    
    # Create a new dataset dictionary with all three splits
    final_dataset = datasets.DatasetDict({
        'train': train_testvalid['train'],  # 80% of the data
        'test': test_valid['test'],         # 10% of the data
        'validation': test_valid['train']   # 10% of the data
    })
    print(f"Size of train split: {len(final_dataset['train'])}")
    print(f"Size of test split: {len(final_dataset['test'])}")
    print(f"Size of validation split: {len(final_dataset['validation'])}")
    print('done loading dataset!')

    return final_dataset

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
        label_id_chosen = batch['chosen_labels'] + [0] * token_pad_length_chosen
        token_id_chosen = token_id_chosen[:model_max_length]
        token_mask_chosen = token_mask_chosen[:model_max_length]

        token_id_chosen = token_id_chosen[:-1]
        token_mask_chosen = token_mask_chosen[1:]
        label_id_chosen = label_id_chosen[1:]
        
        token_ids_chosen.append(token_id_chosen)
        token_masks_chosen.append(token_mask_chosen)
        label_ids_chosen.append(label_id_chosen)
        
        token_id_rejected = batch['rejected_input_ids'] + [pad_token_id] * token_pad_length_rejected
        token_mask_rejected = [1] * len(batch['rejected_input_ids']) + [0] * token_pad_length_rejected
        label_id_rejected = batch['rejected_labels'] + [0] * token_pad_length_rejected
        token_id_rejected = token_id_rejected[:model_max_length]
        token_mask_rejected = token_mask_rejected[:model_max_length]

        token_id_rejected = token_id_rejected[:-1]
        token_mask_rejected = token_mask_rejected[1:]
        label_id_rejected = label_id_rejected[1:]
        
        token_ids_rejected.append(token_id_rejected)
        token_masks_rejected.append(token_mask_rejected)
        label_ids_rejected.append(label_id_rejected)
        
        # assert len(token_id_chosen) == model_max_length, print(len(token_id_chosen), total_len_chosen, token_pad_length_chosen)
    

    return {
        'chosen_input_ids': minitorch.tensor_functions.tensor_from_numpy(np.array(token_ids_chosen), backend),
        'chosen_masks': minitorch.tensor_functions.tensor_from_numpy(np.array(token_masks_chosen), backend),
        'chosen_labels': minitorch.tensor_functions.tensor_from_numpy(np.array(label_ids_chosen), backend),
        'rejected_input_ids': minitorch.tensor_functions.tensor_from_numpy(np.array(token_ids_rejected), backend),
        'rejected_masks': minitorch.tensor_functions.tensor_from_numpy(np.array(token_masks_rejected), backend),
        'rejected_labels': minitorch.tensor_functions.tensor_from_numpy(np.array(label_ids_rejected), backend),
        # 'concatenated_input_ids': minitorch.tensor_functions.tensor_from_numpy(np.concatenate((np.array(token_ids_chosen), np.array(token_ids_rejected))), backend),
        # 'concatenated_masks': minitorch.tensor_functions.tensor_from_numpy(np.concatenate((np.array(token_masks_chosen), np.array(token_masks_rejected))), backend),
        # 'concatenated_labels': minitorch.tensor_functions.tensor_from_numpy(np.concatenate((np.array(label_ids_chosen), np.array(label_ids_rejected))), backend)
    }


def gather(input, dim, index):
    batch_size, seq_len = index.shape
    mask = index.zeros(shape=input.shape)  # Assuming Minitorch has a method to create a zero tensor

    # We use nested loops to iterate over each dimension except the one we are gathering from
    if dim == 2:  # As per your use case, gathering from the last dimension
        for i in range(input.shape[0]):  # Loop over the first dimension (batch size)
            for j in range(input.shape[1]):  # Loop over the second dimension (sequence length)
                k = int(index[i, j])  # This is the index in the vocab size dimension
                # print(mask.shape)
                mask[i, j, k] = 1
    else:
        raise NotImplementedError("Gather function only implemented for dim=2")

    output = input * mask
    output = output.sum(dim=dim).view(batch_size, seq_len)
    # print(output.shape)

    return output

# backend = minitorch.TensorBackend(CudaKernelOps)

# np.random.seed(0)

# logits = tensor_from_numpy(np.random.rand(2, 3, 4), backend=backend)
# labels = tensor_from_numpy(np.ones((2, 3)), backend=backend)

# output = gather(logits, 2, labels)

# logps = tensor_from_numpy(np.random.rand(20, 1), backend=backend)

# print(logps)

# # batch = tensor_from_numpy(np.ones((20, 1)), backend=backend)

# c, r = split_tensor(logps, 10)

# print(c)
# print(r)

def split_tensor(tensor, split_index):
    # Assuming tensor has shape [N] and split_index is the point to split
    part1 = tensor.zeros(tensor.shape)
    part2 = tensor.ones(tensor.shape)

    for i in range(split_index):
        part1[i, 0] = 1
        part2[i, 0] = 0

    chosen = tensor * part1
    rejected = tensor * part2

    return chosen, rejected

# def loss_fn(batch, model, backend):
#     """
#     The MLE loss for a batch.

#     Parameters:
#     - batch: The result of collate_fn, a dict with "input_ids", "labels", and "label_token_weights".
#     - model: The model to be trained.

#     Returns:
#     - A scalar loss value for this batch, averaged across all target tokens.

#     # ------------ToDo------------
#     add preference loss
#     add preference model
#     # ------------ToDo------------
#     """

#     idx = batch['concatenated_input_ids']
#     idx.requires_grad_(True)
    
#     logits = model(idx=idx)
    
#     labels = batch['concatenated_labels']

#     loss_mask = batch['concatenated_masks']
  
#     batch_size, seq_len, vocab_size = logits.shape

#     per_token_logps = gather(logsoftmax(logits, -1), 2, labels)

#     # print('loss_mask', loss_mask)

#     # print('per token logps', per_token_logps)

#     batch_logps = (per_token_logps * loss_mask).sum(1)

#     # print('batch_logps shape: ', batch_logps.shape)

#     chosen_length = batch_logps.shape[0] // 2

#     chosen_logps, rejected_logps = split_tensor(batch_logps, chosen_length)

#     # print('chosen_logps: ', chosen_logps)
#     # print('rejected_logps: ', rejected_logps)
    
#     loss = minitorch.nn.preference_loss(chosen_logps, rejected_logps, beta=0.7)

#     # loss = minitorch.nn.softmax_loss(logits.view(batch_size * seq_len, vocab_size), labels.view(batch_size * seq_len,))
#     # loss = loss.view(batch_size, seq_len)
    
#     batch_loss = loss.mean()

#     print('batch_loss: ', batch_loss)

#     # print("Gradients tracking on logits:", logits.requires_grad())
#     # print("Gradients tracking on chosen_logps:", chosen_logps.requires_grad())
#     # print("Gradients tracking on rejected_logps:", rejected_logps.requires_grad())
#     # print("Gradients tracking on batch loss:", batch_loss.requires_grad())
#     # print("Gradients tracking on batch loss:", batch_loss.requires_grad())

#     # return batch_loss, chosen_rewards, rejected_rewards
#     return batch_loss

def loss_fn(batch, model, backend):
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
    chosen_idx = batch['chosen_input_ids']
    chosen_idx.requires_grad_(True)
    # rejected_idx = batch['rejected_input_ids']
    # rejected_idx.requires_grad_(True)
    
    chosen_logits = model(idx=chosen_idx)
    # rejected_logits = model(idx=rejected_idx)

    chosen_labels = batch['chosen_labels']
    # rejected_labels = batch['rejected_labels']

    chosen_mask = batch['chosen_masks']
    # rejected_mask = batch['rejected_masks']
  
    batch_size, seq_len, vocab_size = chosen_logits.shape

    # chosen_logps = gather(logsoftmax(chosen_logits, -1), 2, chosen_labels)
    # rejected_logps = gather(logsoftmax(rejected_logits, -1), 2, rejected_labels)

    # loss = minitorch.nn.preference_loss(chosen_logps, rejected_logps, beta=0.7)
    loss = minitorch.nn.softmax_loss(chosen_logits.view(batch_size * seq_len, vocab_size), chosen_labels.view(batch_size * seq_len,))

    # loss = minitorch.nn.softmax_loss(logits.view(batch_size * seq_len, vocab_size), labels.view(batch_size * seq_len,))
    # loss = loss.view(batch_size, seq_len)
    
    # batch_loss = loss.mean()

    # # print('train loss: ', batch_loss)
    # return batch_loss

    loss = loss.view(batch_size, seq_len)

    weighted_losses = loss * chosen_mask

    non_zero_weights = chosen_mask.sum()
    
    average_loss = weighted_losses.sum() / non_zero_weights if non_zero_weights > 0 else 0

    return average_loss


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc, backend):
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
    losses = []
    # random.shuffle(examples)
    # examples = examples[:n_samples]
    count = 0
    for i in range(0, len(examples), batch_size):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model, backend=backend)
        print(f'Batch {count}: Train Loss = {loss}')
        losses.append(loss.item())
        t1 = time.time()

        loss.backward()
        # print(loss.grad)
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        # print(f"Forward: {t1 - t0}")
        # print(f"Backward: {t2 - t1}")
        # print(f"Opt.step: {t3 - t2}")

        batch_time = time.time() - t0

        count += 1
    train_loss = np.mean(losses)
    # print('train loss: ', loss)
    return train_loss


def evaluate_loss(model, examples, batch_size, collate_fn, desc, backend):
    """
    Evaluates the model on the provided examples and computes the average loss.

    Parameters:
    - model: The model to be evaluated.
    - examples: The dataset examples used for evaluation.
    - batch_size: The number of examples in each batch.
    - collate_fn: The function to collate data examples into batches.
    - desc: Description for the evaluation process (used in progress bars).

    Returns:
    - The average loss computed over all batches.
    """
    model.eval()
    losses = []

    for i in range(len(examples)):
        batch = collate_fn(examples=examples[i:i + batch_size])
        loss = loss_fn(batch=batch, model=model, backend=backend)

        losses.append(loss.item())
        # prog_bar.set_postfix(loss=loss.item())
    loss = np.mean(losses)
    # print('val loss: ', loss)

    return loss


# TODO: modify generate for our need

def generate(model,
             examples,
             tokenizer,
             max_length,
             backend,
             desc):
    """
    Generates target sequences for the given source sequences using the model, based on argmax decoding.
    Note that it runs generation on examples one-by-one instead of in a batched manner.

    Parameters:
    - model: The model used for generation.
    - examples: The dataset examples containing source sequences.
    - tokenizer: The tokenizer used for encoding texts.
    - model_max_length: The maximum sequence length the model can handle.
    - backend: The backend of minitorch tensors.
    - desc: Description for the generation process (used in progress bars).

    Returns:
    - A list of generated target sequences.
    """

    model.eval()
    gen_sents = []
    # for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
    for i in range(len(examples)):
        # Run generation for every single example
        prompt = examples['prompt'][i]
        token_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']

        while len(token_ids) <= max_length:
            # BEGIN ASSIGN2_2
            # TODO
            # run the model with current token_ids, and predict the next token (gen_id)
            # hint: obtain the logits of next token, and take the argmax.

            token_ids_tensor = tensor_from_numpy(np.array([token_ids]), backend)
            
            # get logits
            logits = model(idx=token_ids_tensor)
            # logits of the last token
            logits_np = logits.to_numpy()[:, -1, :]

            # get the argmax
            gen_id = np.argmax(logits_np, axis=-1).item()

            # print(gen_id)

            # END ASSIGN2_2

            if gen_id == tokenizer.vocab['<|endoftext|>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids))

    return gen_sents



def evaluate_bleu(examples, gen_sents, target='chosen'):
    """
    Evaluates the BLEU score for generated sentences against the target sentences in the examples.

    Parameters:
    - examples: The dataset examples used for evaluation.
    - gen_sents: The generated sentences to be evaluated.
    - tgt_key: The key for accessing target texts in the examples.

    Returns:
    - A dictionary containing the BLEU score.
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[target] for example in examples]]).score
    }


def main(model_max_length=512,
         n_epochs=50,
         batch_size=10,
         learning_rate=0.02,
         samples_per_epoch=2,
         n_vocab=10000,
         n_embd=256,
         seed=11111):
    """
    The main function to train and evaluate the model on a specified dataset.

    Parameters:
    - dataset_name: The name of the dataset to be used.
    - model_max_length: The maximum sequence length the model can handle.
    - n_epochs: The number of training epochs.
    - batch_size: The number of examples in each batch.
    - learning_rate: The learning rate for the optimizer.
    - samples_per_epoch: Samples from the training dataset every epoch.
    - n_vocab: The vocabulary size of the BPE tokenizer.
    - n_embd: The embedding dimension.
    - seed: Random seed.
    """
    wandb.config = {
        'model_max_length': model_max_length,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'samples_per_epoch': samples_per_epoch,
        'n_vocab': n_vocab,
        'n_embd': n_embd,
        'seed': seed
    }

    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir'
    os.makedirs(workdir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)

    config = {
        'n_vocab': 50257,  # vocab_size
        'n_embd': 64,  # n_embed
        'n_head': 2,  # n_head
        'n_positions': 512,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout': 0.1,  # x_pdrop
        'ln_eps': 1e-5,  # layer_norm_epsilon
        'backend': backend
    }

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    dataset = get_imdb(data_path='data/sampled_data.json')
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-large', special_tokens={'pad_token': '<pad>'})

    collate_fn = partial(collate_batch, tokenizer=tokenizer, model_max_length=512, backend=backend)

    sys.stdout.flush()

    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        train_loss = train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc,
            backend=backend)
        
        wandb.log({"train_loss": train_loss, "epoch": epoch_idx})

        validation_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc,
            backend=backend)

        print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

        wandb.log({"validation_loss": validation_loss, "epoch": epoch_idx})

        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            tokenizer=tokenizer,
            max_length=21,
            backend=backend,
            desc=desc)

        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        json.dump(gen_examples, open(
            f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

        eval_scores_chosen = evaluate_bleu(
            examples=dataset['test'], gen_sents=gen_sents, target='chosen')
        # eval_scores_rejected = evaluate_bleu(
        #     examples=dataset['test'], gen_sents=gen_sents, target='rejected')
        
        print(f'Epoch {epoch_idx}: Chosen Eval {eval_scores_chosen}, Rejected Eval {eval_scores_rejected}')

        wandb.log({"chosen_bleu": eval_scores_chosen, "epoch": epoch_idx})  # Separate plot for chosen BLEU
        # wandb.log({"rejected_bleu": eval_scores_rejected, "epoch": epoch_idx})  # Separate plot for rejected BLEU

        eval_scores = {
        'chosen_eval': eval_scores_chosen,
        # 'rejected_eval': eval_scores_rejected
        }
        json.dump(
            {'validation_loss': float(validation_loss), **eval_scores},
            open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))


if __name__ == '__main__':
    wandb.init(project='run_no_dpo_imdb', entity='kellyshiiii')
    main()
    wandb.finish()


