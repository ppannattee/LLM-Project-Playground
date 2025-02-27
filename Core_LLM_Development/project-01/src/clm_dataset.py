from datasets import load_dataset, concatenate_datasets
from itertools import chain
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch

class CLMDataset(Dataset):
    def __init__(self, block_size):

        self.block_size = block_size

        # Load the dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

        # Merge train, validation, and test splits into a single dataset
        dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Tokenize entired dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"])
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Group tokens into chunks for training causal language model 
        self.concated_tokens = list(chain.from_iterable(example['input_ids'] for example in tokenized_dataset))

    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())

    def __len__(self):
        # Return number of batches (total tokens divided by block size)
        return len(self.concated_tokens) // self.block_size

    def __getitem__(self, idx):
        # Get a chunk of tokens
        start = idx * self.block_size
        end = start + self.block_size
        input_ids = self.concated_tokens[start:end]
        labels = self.concated_tokens[start+1:end+1]  # Shift by one for the labels (next token)

        # Convert the tokens to a tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}