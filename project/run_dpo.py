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
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
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
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, Tensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
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
        # policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            # reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model, batch: Dict[str, Union[List, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        # all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
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

            # chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            # rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            # reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            # policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        # policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        # all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        # self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        # torch.manual_seed(self.seed)
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
                # rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                # if self.config.sample_during_eval:
                #     all_policy_samples, all_reference_samples = [], []
                #     policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                #     if self.config.loss.name in {'dpo', 'ipo'}:
                #         reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics')):
                    # local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    # with torch.no_grad():
                    model.eval()
                    _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                # if self.config.sample_during_eval:
                #     if self.config.n_eval_model_samples < self.config.eval_batch_size:
                #         # rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                #         sample_batches = self.eval_batches[:1]
                #     else:
                #         n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                #         sample_batches = self.eval_batches[:n_sample_batches]
                #     for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...')):
                #         # local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                #         policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                #         all_policy_samples.extend(policy_samples)
                #         all_reference_samples.extend(reference_samples)

                #         for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                #             policy_text_table.add_data(self.example_counter, prompt, sample)
                #         if self.config.loss.name in {'dpo', 'ipo'}:
                #             for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                #                 reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                # if self.config.sample_during_eval:                    
                #     rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                #     if self.config.loss.name in {'dpo', 'ipo'}:
                #         rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                # if self.config.wandb.enabled and self.rank == 0:
                #     wandb.log(mean_eval_metrics, step=self.example_counter)

                    # if self.config.sample_during_eval:
                    #     wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                    #     if self.config.loss.name in {'dpo', 'ipo'}:
                    #         wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                if self.example_counter > 0:
                    output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                    rank0_print(f'creating checkpoint to write to {output_dir}...')
                    self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            # batch_metrics = defaultdict(list)
            # for microbatch_idx in range(self.config.gradient_accumulation_steps):
            #     # global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
            #     # local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
            #     loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
            #     (loss / self.config.gradient_accumulation_steps).backward()

            #     for k, v in metrics.items():
            #         batch_metrics[k].extend(v)
            loss, metrics = self.get_batch_metrics(batch, self.loss_config, train=True)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            # self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            # batch_metrics['examples_per_second'].append(examples_per_second)
            # batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            # if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
            #     mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
            #     mean_train_metrics['counters/examples'] = self.example_counter
            #     mean_train_metrics['counters/updates'] = self.batch_counter
            #     rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

            #     if self.config.wandb.enabled and self.rank == 0:
            #         wandb.log(mean_train_metrics, step=self.example_counter)

            #     last_log = time.time()
            # else:
            #     rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            if last_log is None or time.time() - last_log > self.loss_config["minimum_log_interval_secs"]:
                rank0_print(f'train stats after {example_counter} examples: Loss: {loss.item()}, Grad norm: {grad_norm}, Examples/sec: {examples_per_second}')
                last_log = time.time()
            #### END TRAINING ####


    # def clip_gradient(self):
    #     """Clip the gradient norm of the parameters of a non-FSDP policy."""
    #     return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()
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
        rank0_print(f'writing checkpoint to {output_path}...')
        # torch.save({
        #     'step_idx': step,
        #     'state': state,
        #     'metrics': metrics if metrics is not None else {},
        # }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        # scheduler_state_dict = self.scheduler.state_dict()
        # self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        

# class TensorParallelTrainer(BasicTrainer):
#     def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
#         """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

#            Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
#               see https://github.com/BlackSamorez/tensor_parallel/issues/66.
#         """
#         super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        
#         rank0_print('Sharding policy...')
#         self.policy = tp.tensor_parallel(policy, sharded=True)
#         if config.loss.name in {'dpo', 'ipo'}:
#             rank0_print('Sharding reference model...')
#             self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

#     def save(self, output_dir=None, metrics=None):
#         """Save (unsharded) policy state to disk."""
#         with tp.save_tensor_parallel(self.policy):
#             policy_state_dict = self.policy.state_dict()
    
#         self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
#         del policy_state_dict
        