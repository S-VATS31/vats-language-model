import time
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, interleave_datasets

from configs.model_args.args_5M import ModelArgs
from utils.logger import setup_logger

logger = setup_logger(__name__, "training.log")

class TextDataset(Dataset):
    """Text dataset class.
    
    Args:
        tokenizer: HuggingFace tokenizer.
        model_args (ModelArgs): Model hyperparameters.
        dataset_name (str): Name of dataset to be downloaded.
        split (str): Split of the dataset to train on.
        max_samples (int, optional): Maximum samples to train on.
            `None` trains on the entire dataset.
    """
    def __init__(
        self,
        tokenizer,
        model_args: ModelArgs,
        dataset_names: list[str],
        split: str,
        max_samples: int | None = None,
        interleave: bool = True
    ):
        self.tokenizer = tokenizer
        self.model_args = model_args
        
        # initialize list of loaded datasets
        datasets = []

        if not dataset_names:
            raise ValueError("Got 0 dataset names.")

        for i, dataset_name in enumerate(dataset_names):
            dataset = load_dataset(
                dataset_name, 
                split=split, 
                streaming=True
            )
            logger.info(f"Dataset {i+1}: {dataset_name}")
            datasets.append(dataset)

        if interleave:
            self.ds = interleave_datasets(datasets)
        else:
            self.ds = concatenate_datasets(datasets)

        self.dataset = []
        if max_samples is not None:
            for i, example in enumerate(self.ds):
                if i >= max_samples:
                    break
                self.dataset.append(example)
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
    