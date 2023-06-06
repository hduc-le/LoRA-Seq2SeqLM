import os
import argparse
import numpy as np
import torch
import transformers
import datasets
from torch.optim import AdamW
from sophia import SophiaG # Sophia optimizer
from utils.data_utils import DataProcessor
from datasets import load_dataset # huggingface dataset
from tqdm import tqdm # progress bar
from torchmetrics import MeanMetric # gather and compute losses
from accelerate import Accelerator # distributed training with multi-gpus
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import get_scheduler, set_seed, DataCollatorForSeq2Seq
from utils.utils import read_config, print_config
from utils.consts import logger
from peft.utils.other import fsdp_auto_wrap_policy
from utils.model_utils import load_model_tokenizer, print_trainable_parameters

def train(config):
    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(config["training"]["seed"])
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"]
    )
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    # Show training config only in main process
    if accelerator.is_local_main_process:
        print("Training config:")
        print_config(config)

    # Load T5 model and tokenizer
    model, tokenizer = load_model_tokenizer(config["model"]["name"])

    # LoRA fine tune
    if config["lora"]["active"]:
        if config["lora"]["checkpoint"]:
            accelerator.print(
                f"Loading pretrained peft model from {config['lora']['checkpoint']}"
            )
            model = PeftModel.from_pretrained(
                model, config["lora"]["checkpoint"], is_trainable=True
            )
        else:
            peft_config = LoraConfig(
                r=config["lora"]["r"],
                inference_mode=False,
                lora_alpha=config["lora"]["alpha"],
                lora_dropout=config["lora"]["dropout"],
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            model = get_peft_model(model, peft_config)

        # FSDP plugin with accelerate
        if getattr(accelerator.state, "fsdp_plugin", None) is not None:
            accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
        
    logger.info(
        f"Mem needed: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB"
    )

    if accelerator.is_local_main_process:
        print_trainable_parameters(model)
        
    model = accelerator.prepare(model)

    # Initialize the data processor
    processor = DataProcessor(
        tokenizer, src_len=config["data"]["src_len"], tgt_len=config["data"]["tgt_len"]
    )

    # Load your custom dataset
    data = load_dataset("json", data_files=config["data"]["path"], split="train")

    # Preprocess the data
    encoded_dataset = data.map(
        processor.preprocess_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=data.column_names,
        keep_in_memory=False,
        load_from_cache_file=True,
    )
    # 90% train, 10% test + validation
    encoded_dataset = encoded_dataset.train_test_split(
        test_size=config["data"]["test_size"], seed=config["training"]["seed"]
    )
    # We want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    train_dataloader = DataLoader(
        encoded_dataset["train"],
        batch_size=config["data"]["train_batch_size"],
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        encoded_dataset["test"],
        batch_size=config["data"]["eval_batch_size"],
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=True,
    )

    optimizer_cls = SophiaG if config["optimizer"]["name"] == "sophia" else AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]

    # Decay to min_lr instead of 0
    lr_ratio = config["optimizer"]["min_lr"] / config["optimizer"]["lr"]
    accelerator.print(f"Len of train_dataloader: {len(train_dataloader)}")
    total_num_steps = (len(train_dataloader) / gradient_accumulation_steps) * config["training"]["num_epochs"]
    # Instead of decaying to zero, decay to ratio of min_lr / lr
    total_num_steps += (
        int(total_num_steps * lr_ratio) + config["optimizer"]["warmup_steps"]
    )
    accelerator.print(f"Total training steps: {total_num_steps}")

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config["optimizer"]["warmup_steps"]
        * accelerator.num_processes,
        num_training_steps=total_num_steps,
    )

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    (
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(optimizer, scheduler, train_dataloader, eval_dataloader)

    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = np.inf

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        train_loss = MeanMetric(nan_strategy="error").to(model.device)
        for step, batch in enumerate(
            (
                pbar := tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch} - Training",
                    disable=not accelerator.is_main_process,
                )
            )
        ):
            # Forward & get loss
            outputs = model(**batch)
            loss = outputs.loss

            # Progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Gather loss before backprop in case of gradient accumulation
            loss_values = accelerator.gather_for_metrics(
                {"loss": loss.detach().float()}
            )
            train_loss.update(loss_values["loss"])

            # Gradient accumulation and backprop
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Evaluate at the end of the epoch (distributed evaluation as we have all GPU cores)
        model.eval()
        val_loss = MeanMetric(nan_strategy="error").to(model.device)
        for batch in (
            pbar := tqdm(
                eval_dataloader,
                desc=f"Epoch {epoch} - Validation",
                disable=not accelerator.is_main_process,
            )
        ):
            with torch.no_grad():
                loss = model(**batch).loss

                pbar.set_postfix({"loss": loss.item()})

                loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})

                val_loss.update(loss_values["loss"])

        # Compute average train and validation loss
        log_items = {"train_loss": train_loss.compute(), "val_loss": val_loss.compute()}
        # Use accelerator.print to print only on the main process.
        accelerator.print(
            f"Summary epoch {epoch}: train loss: {log_items['train_loss'].item()} || validation loss: {log_items['val_loss'].item()}"
        )

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss

            # Pushing model to HuggingFace hub
            accelerator.print(f"Epoch {epoch} finished")
            accelerator.print(f"Pushing to HF hub")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            try:
                if accelerator.is_main_process:
                    unwrapped_model.push_to_hub(
                        config["model"]["save_name"] + f"-epoch-{epoch}", private=True
                    )

            except Exception as e:
                accelerator.print(e)
                accelerator.print(f"Failed to push to hub")

            # Local saving
            unwrapped_model.save_pretrained(
                f"{config['model']['output_dir']}/{config['model']['save_name']}-epoch-{epoch}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
        else:
            # Check early stopping condition
            epochs_no_improve += 1
            if epochs_no_improve == config["training"]["patience"]:
                accelerator.print("Early stopping!")
                break

    # Local saving trained model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # Use accelerator.save to save
    unwrapped_model.save_pretrained(
        f"{config['model']['output_dir']}/{config['model']['save_name']}-final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    logger.info("Done. ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    args = parser.parse_args()

    # Load the training config file
    config = read_config(args.config)
    train(config)
