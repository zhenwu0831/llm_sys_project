import warnings
from typing import Dict
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
# from trl import DPOTrainer
from dpo_trainer import DPOTrainer
from dataclasses import dataclass, field
@dataclass
class ScriptArguments:
    beta: float = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )

if __name__ == "__main__":
    # parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    # args, training_args, model_config = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # model_name_or_path = "test" + str(args.beta)
    model_name_or_path = "gpt2"
    per_device_train_batch_size = 1
    max_steps = 10000
    gradient_accumulation_steps = 2
    gradient_checkpointing = False
    learning_rate = 1e-4
    report_to = None
    max_length = 1024
    max_prompt_length = 128
    max_target_length = 128
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        max_steps=max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        eval_steps=500,
        output_dir="./test" + str(args.beta),
        optim="rmsprop",
        warmup_steps=150,
        report_to=report_to,
        fp16=True,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset("json", data_files="data/train.json", split="train")
    eval_dataset = load_dataset("json", data_files="data/test.json", split="train")
    def split_prompt_ands(sample) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
    # train_dataset = train_dataset.map(split_prompt_ands)
    # eval_dataset = eval_dataset.map(split_prompt_ands)

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_target_length=max_target_length,
        max_prompt_length=max_prompt_length,
        generate_during_eval=False,
    )
    
    wandb.init(project="reproduce-dpo", name="modelbeta=" + str(args.beta))
    trainer.train()
    trainer.save_model("./test" + str(args.beta))
    output_dir = os.path.join("./test" + str(args.beta), "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.custom_eval(str(args.beta))