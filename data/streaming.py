import time

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from configs.model_args.args_5M import ModelArgs

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        model_args: ModelArgs,
        dataset_name: str,
        split: str,
        max_samples: int | None = None
    ):
        self.tokenizer = tokenizer
        self.model_args = model_args

        # Load dataset with retry logic
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.ds = load_dataset(
                    dataset_name, 
                    split=split, 
                    streaming=True
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

        self.dataset = []
        if max_samples is not None:
            for i, example in enumerate(self.ds):
                if i >= max_samples:
                    break
                
                max_example_retries = 3
                for attempt in range(max_example_retries):
                    try:
                        self.dataset.append(example)
                        break
                    except Exception as e:
                        if attempt < max_example_retries - 1:
                            time.sleep(2)
        else:
            for i, example in enumerate(self.ds):
                self.dataset.append(example)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['content']

        # tokenize with padding to max_length
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model_args.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )

        # squeeze to remove batch dimension
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        # create labels, mask padding tokens
        labels = input_ids.clone()
        assert self.tokenizer.pad_token_id is not None
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # shift input_ids and labels for causal LM
        shifted_input_ids = input_ids[:-1]
        shifted_labels = labels[1:]

        shifted_attention_mask = attention_mask[:-1]

        return {
            'input_ids': shifted_input_ids,
            'attention_mask': shifted_attention_mask,
            'labels': shifted_labels
        }
    