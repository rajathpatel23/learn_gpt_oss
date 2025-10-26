import torch, gc
from torch.utils.data import Dataset, DataLoader
import tiktoken
import logging
logger = logging.getLogger(__name__)
PATH = "/Users/rajatpatel/research/learn_gpt_oss/src/data/tiny_shakespeare.txt"

batch_size = 2
context_len = 128

def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer
    
# Load a small local dataset subset for quick runs
with open(PATH, "r") as f:
    full_text = f.read()
max_train_chars = 50_000
max_val_chars = 10_000
train_text = full_text[:max_train_chars]
val_text = full_text[max_train_chars:max_train_chars + max_val_chars]
tokenizer = get_tokenizer()
train_tokens = tokenizer.encode(train_text)
val_tokens = tokenizer.encode(val_text)

class TextDataset(Dataset):
    def __init__(self, tokens, max_length=context_len, stride=context_len):
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i:i + max_length]
            target_chunk = tokens[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
val_dataset = TextDataset(val_tokens, max_length=context_len, stride=context_len)

# Light-weight DataLoaders for local runs
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)