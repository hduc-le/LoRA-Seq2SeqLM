# LoRA-Seq2SeqLM
## Overview
A simple and friendly code for fine-tuning Seq2SeqLM models like T5 with `LoRA` (Low-Rank Adaptation) method that can be applied to the custom datasets and provide distributed training mode on multi-GPUs using `Accelerate`.

## Installation
To install the necessary software, follow the following command:
```bash
pip install -r requirements.txt
```

Recommend to install the `git-lfs` if pushing the model to HuggingFace hub while training:
```bash
sudo apt-get install git-lfs
```

Verify the HuggingFace access token to be allowed to create a repository for saving the model in the hub. 
```bash
huggingface-cli login
```

## Usage
Before training, you need to specify some configuration in `configs/finetune.yaml`, it will be loaded internally when executing `train.py`. 

### Training
To fine-tune with `Accelerate` framework, follow the steps:

1. Generate config and follow the instruction (to specify number gpus, machines, mixed-precision, etc).
```bash
accelerate config
```

2. Perform fine-tuning.
```bash
accelerate launch train.py --config configs/finetune.yaml
```
### Generation
Update later

## Todos (future works):
1. DeepSpeed Integration
2. FSDP Integration (but can basically run by configuring `accelerate config`)