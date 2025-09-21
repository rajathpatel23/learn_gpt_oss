"""
Training a Custom Tokenizer using Byte-Pair Encoding (BPE) on a small dataset.
"""

import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import ByteLevel as ByteLevelPostProcessor
import tokenizers.decoders as decoder
from tokenizers.trainers import BpeTrainer
import torch
import os



torch.manual_seed(123)


def train_tokenizer(data_path: str = "data/train.txt", tokenizer_path: str = "tokenizer_data/tokenizer.json"):
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(data_path, "r") as f:
        data = f.read()
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.post_processor = ByteLevelPostProcessor(trim_offsets=False)
    tokenizer.decoder = decoder.ByteLevel()
    trainer = BpeTrainer(special_tokens=["<|endoftext|>"], min_frequency=2)
    tokenizer.train([data_path], trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer

def encode_text(text: str, tokenizer: Tokenizer):
    return tokenizer.encode(text).ids

def decode_text(ids: list[int], tokenizer: Tokenizer):
    return tokenizer.decode(ids)

if __name__ == "__main__":
    data_path = "/Users/rajatpatel/research/learn_gpt_oss/src/data/tiny_shakespeare.txt"
    tokenizer_path = "/Users/rajatpatel/research/learn_gpt_oss/src/tokenizer_data/tokenizer.json"
    tokenizer = train_tokenizer(data_path, tokenizer_path)
    
    with open(data_path, "r") as f:
        text = f.read()
    text = text[:10000]
    ids = encode_text(text, tokenizer)
    print(ids)
    decoded_text = decode_text(ids, tokenizer)
    print(decoded_text)

