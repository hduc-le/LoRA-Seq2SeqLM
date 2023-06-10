import os
import argparse
import numpy as np
import torch
import transformers
import datasets
from torch.optim import AdamW
from sophia import SophiaG  # Sophia optimizer
from utils.data_utils import DataProcessor
from datasets import load_dataset  # huggingface dataset
from tqdm import tqdm  # progress bar
from torchmetrics import MeanMetric  # gather and compute losses

################ DDP ###################
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
########################################

from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import get_scheduler, set_seed, DataCollatorForSeq2Seq
from utils.utils import read_config, print_config
from utils.utils import get_logger
from utils.model_utils import load_model_tokenizer, print_trainable_parameters


def ddp_setup(rank: int, world_size: int):
    """Set up for distributed training

    Args:
        rank (int): Unique identifier of each process
        world_size (int): Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, init_method="env://", world_size=world_size)
    torch.cuda.set_device(rank)


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    args = parser.parse_args()

    # Load the training config file
    config = read_config(args.config)

    n_gpus = torch.cuda.device_count()

    mp.spawn(train, nprocs=n_gpus, args=(n_gpus, config,))


def train(rank, n_gpus, config):
    logger = get_logger()

    # DDP setup
    ddp_setup(rank=rank, world_size=n_gpus)
    logger.info(f"Using {n_gpus} GPUs")

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(config["train"]["seed"])
    if rank == 0:
        print("Training config:")
        print_config(config)

    # Load T5 model and tokenizer
    model, tokenizer = load_model_tokenizer(config["model"]["name"])

    # LoRA fine tune
    if config["lora"]["active"]:
        if config["lora"]["checkpoint"]:
            logger.info(
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

    if rank == 0:
        print_trainable_parameters(model)
        logger.info(f"Mem needed per device: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB")

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
        remove_columns=data.column_names,
        keep_in_memory=False,
        load_from_cache_file=True,
    )

    # 90% train, 10% test + validation
    encoded_dataset = encoded_dataset.train_test_split(
        test_size=config["data"]["test_size"], seed=config["train"]["seed"]
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
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(encoded_dataset["train"]),
    )

    eval_dataloader = DataLoader(
        encoded_dataset["test"],
        batch_size=config["data"]["eval_batch_size"],
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(encoded_dataset["test"]),
    )

    if rank == 0:
        logger.info(f"Len of train_dataloader: {len(train_dataloader)}")
        logger.info(f"Len of eval_dataloader: {len(eval_dataloader)}")

    optimizer_cls = SophiaG if config["optimizer"]["name"] == "sophia" else AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    gradient_accumulation_steps = config["train"]["gradient_accumulation_steps"]

    # Decay to min_lr instead of 0
    lr_ratio = config["optimizer"]["min_lr"] / config["optimizer"]["lr"]

    total_num_steps = (len(train_dataloader) / gradient_accumulation_steps) * config["train"]["num_epochs"]
    # Instead of decaying to zero, decay to ratio of min_lr / lr
    total_num_steps += int(total_num_steps * lr_ratio) + config["optimizer"]["warmup_steps"]
    
    logger.info(f"Total training steps: {total_num_steps}")

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config["optimizer"]["warmup_steps"] * n_gpus,
        num_training_steps=total_num_steps,
    )

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = np.inf

    for epoch in range(config["train"]["num_epochs"]):
        model.train()
        train_loss = MeanMetric(nan_strategy="error").cuda(rank)
        train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(
            pbar := tqdm(
                train_dataloader,
                desc=f"Epoch {epoch} - Training",
                disable=not rank == 0,
            )
        ):
            # Forward & get loss
            outputs = model(**batch)
            loss = outputs.loss

            # Progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Gather loss before backprop in case of gradient accumulation
            train_loss.update(loss.detach().float())

            # Gradient accumulation and backprop
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
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
                disable=not rank == 0,
            )
        ):
            with torch.no_grad():
                loss = model(**batch).loss

                pbar.set_postfix({"loss": loss.item()})

                val_loss.update(loss.detach().float())

        # Compute average train and validation loss
        log_items = {"train_loss": train_loss.compute(), "val_loss": val_loss.compute()}
        # Use logger.info to print only on the main process.
        if rank == 0:
            logger.info(
                f"Summary epoch {epoch}: train loss: {log_items['train_loss'].item()} || validation loss: {log_items['val_loss'].item()}"
            )

        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            
            if rank == 0:
                # Pushing model to HuggingFace hub
                logger.info(f"Epoch {epoch} finished")
                logger.info(f"Pushing to HF hub")
                try:
                    unwrapped_model = model.module
                    unwrapped_model.push_to_hub(config["model"]["save_name"] + f"-epoch-{epoch}", private=True)
                except Exception as e:
                    logger.info(e)
                    logger.info(f"Failed to push to hub")
                # Local saving
                unwrapped_model = model.module
                unwrapped_model.save_pretrained(
                    f"{config['model']['output_dir']}/{config['model']['save_name']}-epoch-{epoch}",
                )
        else:
            # Check early stopping condition
            epochs_no_improve += 1
            if epochs_no_improve == config["train"]["patience"]:
                logger.info("Early stopping!")
                break

    # Local saving trained model
    if rank == 0:
        unwrapped_model = model.module
        # Use accelerator.save to save
        unwrapped_model.save_pretrained(
            f"{config['model']['output_dir']}/{config['model']['save_name']}-final",
        )

    logger.info("Done. ")
    destroy_process_group()


if __name__ == "__main__":
    main()
