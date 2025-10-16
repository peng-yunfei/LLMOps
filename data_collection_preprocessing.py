import pandas as pd
import re
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('data.csv')
df = df['text'][:250000]

# ==================================
# CLEANING
# ==================================
# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with less than 50 words
df = df[df.str.split().str.len() >= 50]

# Normalize text: lowercase, remove HTML tags, reference markers, and extra whitespace
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)        # remove HTML tags
    text = re.sub(r"\[.*?\]", " ", text)      # remove reference markers
    text = re.sub(r"\s+", " ", text).strip()
    return text
df = df.apply(normalize_text)
# print( f"✅ Cleaned dataset." )
# print("Sample cleaned text:", df.iloc[0][:10])

# ==================================
# TOKENIZATION + CHUNKING
# ==================================

# Load tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize and chunk text
max_len = tokenizer.model_max_length
def tokenize_and_chunk(text, tokenizer, max_len):
    tokens = tokenizer.encode(text, add_special_tokens=True)

    if len(tokens) > max_len:
        return [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]
    else:
        return [tokens]
df = df.apply(
    lambda x: tokenize_and_chunk(x, tokenizer, max_len)
)

# Flatten list of lists
tokenized_sequences = [chunk for chunks in df for chunk in chunks]
# print(f"✅ Tokenized and chunked dataset into {len(tokenized_sequences)} sequences.")
# print("Sample tokenized sequence (first 20 tokens):", tokenized_sequences[0][:20])

# ==================================
# CUSTOM DATA LOADER (PyTorch)
# ==================================

# Define a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",  
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),   
            "attention_mask": encoding["attention_mask"].squeeze(0)  
        }
    
# Create DataLoader
dataset = TextDataset(tokenized_sequences, max_len=128, tokenizer=tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ==================================
# SAMPLE PROCESSED DATA
# ==================================
sample_batches = []
for i, batch in enumerate(loader):
    sample_batches.append(batch)
    print(f"\n--- Batch {i+1} ---")
    print("input_ids shape:", batch["input_ids"].shape)        
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("First 10 tokens of first sample:", batch["input_ids"][0][:10].tolist())
    if i == 4:  
        break

# Save sample batches
torch.save(sample_batches, "sample_dataset.pt")
print("\n✅ Saved 5 sample batches to sample_dataset.pt")