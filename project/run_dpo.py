import transformers
from omegaconf import DictConfig

import contextlib

from preference_datasets import get_batch_iterator
# from utils import (
#     slice_and_move_batch_for_device,
#     formatted_dict,
#     all_gather_if_needed,
#     pad_to_length,
#     get_block_class_from_model,
#     rank0_print,
#     get_local_dir,
# )
import numpy as np
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
from minitorch.tensor import Tensor
from minitorch.tensor_functions import *
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch import DecoderLM

from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

backend = minitorch.TensorBackend(CudaKernelOps)

def pad_to_length(tensor: Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> Tensor:
    if tensor.shape[dim] >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.shape[dim]

        ones_tensor = ones(*pad_size, backend=backend)

        return np.concatenate([tensor, pad_value * ones_tensor], dim=dim)


def _get_batch_logps(logits: Tensor, labels: Tensor, average_log_prob: bool = False) -> Tensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        An array of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = np.copy(labels[:, 1:])
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    # Calculating log softmax along the last dimension of logits
    max_logits = np.max(logits, axis=-1, keepdims=True)
    logits = logits - max_logits  # for numerical stability
    exp_logits = np.exp(logits)
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    log_softmax = np.log(softmax)

    # Gather the log probabilities of the actual labels
    per_token_logps = np.take_along_axis(log_softmax, labels[..., np.newaxis], axis=-1).squeeze(-1)

    if average_log_prob:
        output = np.sum(per_token_logps * loss_mask, axis=-1) / np.sum(loss_mask, axis=-1)
    else:
        output = np.sum(per_token_logps * loss_mask, axis=-1)

    return tensor_from_numpy(output, backend=backend, requires_grad=True)


def concatenated_inputs(batch: Dict[str, Union[List, Tensor]]) -> Dict[str, Tensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            result = np.concatenate((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
            concatenated_batch[concatenated_key] = tensor_from_numpy(result, backend=backend, requires_grad=True)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy, config: DictConfig, seed: int, run_dir: str, reference_model, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        # rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        # rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        # rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, Tensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        ctx = lambda: (contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model, batch: Dict[str, Union[List, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps


    def get_batch_metrics(self, batch: Dict[str, Union[List, Tensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo'}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            # with torch.no_grad():
            model.eval()
            reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            if loss_config.name == 'dpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        metrics[f'loss/{train_test}'] = losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        # rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):

                self.policy.eval()

                all_eval_metrics = defaultdict(list)

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics')):

                    model.eval()
                    _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                # rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

                if self.example_counter > 0:
                    output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                    # rank0_print(f'creating checkpoint to write to {output_dir}...')
                    self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            loss, metrics = self.get_batch_metrics(batch, self.loss_config, train=True)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.loss_config["minimum_log_interval_secs"]:
                # rank0_print(f'train stats after {example_counter} examples: Loss: {loss.item()}, Grad norm: {grad_norm}, Examples/sec: {examples_per_second}')
                last_log = time.time()
            #### END TRAINING ####

    def clip_gradients(self, max_grad_norm):
        # Compute the total norm of all gradients
        total_norm = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        return total_norm


    def write_state_dict(self, step: int, state: Dict[str, Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        # rank0_print(f'writing checkpoint to {output_path}...')

    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

def get_dataset(dataset_name, model_max_length):
    """
    Obtrain IWSLT (de-en) dataset.
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(
                example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]  # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, workdir):
    """
    Trains a tokenizer on the provided dataset examples and saves the tokenizer configuration.

    Parameters:
    - examples: The dataset examples used for training the tokenizer.
    - vocab_size: The desired vocabulary size for the tokenizer.
    - src_key: The key used to access the source text within the dataset examples.
    - tgt_key: The key used to access the target text within the dataset examples.
    - workdir: The directory where the tokenizer should be saved.

    Returns:
    - tokenizer: The trained tokenizer with special tokens,
        e.g., ("<eos_de>", "<eos_en>", "<pad>") if src_key and tgt_key are "de" and "en", respectively.
    """
    tokenizer = ByteLevelBPETokenizer()

    # # Customized training
    # tokenizer.train_from_iterator(
    #     [[example[src_key], example[tgt_key]] for example in examples],
    #     vocab_size=vocab_size,
    #     special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer

def main(dataset_name='bbaaaa/iwslt14-de-en-preprocess',
         model_max_length=40,
         n_epochs=20,
         batch_size=128,
         learning_rate=0.02,
         samples_per_epoch=20000,
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

    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    # backend = minitorch.TensorBackend(CudaKernelOps)

    config = {
        'n_vocab': n_vocab,  # vocab_size
        'n_embd': n_embd,  # n_embed
        'n_head': 8,  # n_head
        'n_positions': model_max_length,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout': 0.1,  # x_pdrop
        'ln_eps': 1e-5,  # layer_norm_epsilon
        'backend': backend
    }

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        workdir=workdir)

    foo

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)

    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        validation_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            backend=backend,
            desc=desc)

        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        json.dump(gen_examples, open(
            f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

        eval_scores = evaluate_bleu(
            examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
        print(f'Epoch {epoch_idx}: {eval_scores}')

        json.dump(
            {'validation_loss': float(validation_loss), **eval_scores},
            open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))


if __name__ == '__main__':
    fire.Fire(main)