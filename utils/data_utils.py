# from torch.utils.data import Dataset
# from transformers import T5Tokenizer
# from typing import List, Dict


# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         data: List[Dict],
#         tokenizer: T5Tokenizer,
#         source_max_token_len: int = 512,
#         target_max_token_len: int = 512,
#     ):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.source_max_token_len = source_max_token_len
#         self.target_max_token_len = target_max_token_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index: int):
#         data_row = self.data[index]

#         source_encoding = self.tokenizer(
#             data_row["question_text"],
#             data_row["question_context"],
#             max_length=self.source_max_token_len,
#             padding="max_length",
#             truncation="only_second",
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt",
#         )

#         target_encoding = self.tokenizer(
#             data_row["answer_text"],
#             max_length=self.target_max_token_len,
#             padding="max_length",
#             truncation=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt",
#         )

#         labels = target_encoding["input_ids"]
#         labels[labels == 0] = -100
#         return dict(
#             input_ids=source_encoding["input_ids"].flatten(),
#             attention_mask=source_encoding["attention_mask"].flatten(),
#             labels=labels.flatten(),
#         )

from .consts import SOURCE_FORMAT, TARGET_FORMAT

class DataProcessor:
    def __init__(self, tokenizer, src_len=512, tgt_len=512):
        self.tokenizer = tokenizer
        self.src_len = src_len
        self.tgt_len = tgt_len

    def preprocess_function(self, examples):
        inputs = [SOURCE_FORMAT.format(source=q) for q in examples["input"]]

        outputs = [TARGET_FORMAT.format(target=a) for a in examples["output"]]

        tokenized_inputs = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.src_len,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_outputs = self.tokenizer(
            outputs,
            truncation=True,
            max_length=self.tgt_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Ignore padding tokens in the labels
        labels = tokenized_outputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
        }