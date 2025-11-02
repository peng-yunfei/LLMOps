import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

# 1. Configuration
"""
    Global hyperparameter configuration.

    Tuned based on HW2 experiments:
        - Learning rate: 1e-3 (stable convergence, PPL ≈ 42)
        - Batch size: 32 (balanced speed vs. GPU memory)
        - Sequence length: 64 tokens (short-sequence adaptability)
        - Embedding dim: 128
        - Attention heads: 4
        - Transformer layers: 2
"""
tokenizer = AutoTokenizer.from_pretrained("gpt2")

class Config:
    seq_len = 64
    vocab_size = len(tokenizer)
    embed_dim = 128
    n_heads = 4
    ff_dim = 512
    n_layers = 2
    batch_size = 32
    lr = 1e-3
    n_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "mini_gpt_checkpoint.pt"


# 2. Dataset
class TokenDataset(Dataset):
    """
    Dataset wrapper for next-token prediction.

    Each sample is a sequence of `seq_len` tokens (x)
    and its corresponding next-token target (y).
    """
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# 3. Positional Encoding
class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    These embeddings are more adaptive for short sequence lengths
    compared to fixed sinusoidal encodings.
    """
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        b, t = x.size()
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        return self.pos_emb(positions)


# 4. Transformer Language Model
class MiniGPT(nn.Module):
    """
    Mini-GPT model: a simplified GPT-style transformer encoder.

    Architecture:
        Token Embedding + Positional Embedding
        → Transformer Encoder (2 layers, 4 heads)
        → LayerNorm + Linear Projection to vocab logits
    """
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = PositionalEmbedding(cfg.seq_len, cfg.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.embed_dim)
        self.fc_out = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def forward(self, x):
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb
        x = self.encoder(x)
        x = self.ln(x)
        return self.fc_out(x)


# 5. Training and Evaluation
def train_model(model, dataloader, cfg):
    """
    Main training loop.

    Uses:
        - CrossEntropyLoss for next-token prediction
        - AdamW optimizer (stable and generalizes well)
        - Gradient clipping to prevent exploding gradients
        - Perplexity computed from average loss
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    loss_history = []

    for epoch in range(cfg.n_epochs):
        epoch_loss = 0.0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}"):
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Perplexity={ppl:.2f}")

    # Save model checkpoint
    torch.save(model.state_dict(), cfg.checkpoint_path)
    print(f"Checkpoint saved to {cfg.checkpoint_path}")
    return loss_history


def plot_loss(loss_history):
    plt.figure(figsize=(6,4))
    plt.plot(loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Mini-GPT)")
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    plt.show()

# 6. Load Toekenized Data
def load_tokenized_data(path="sample_dataset.pt"):
    """
    Loads and flattens token IDs from sample_batches.pt or sample_dataset.pt
    produced by data_collection_preprocessing.py.

    Each saved object is a list of 5 mini-batches (dicts with 'input_ids').
    This function concatenates all tokens into one flat list suitable
    for the Mini-GPT training pipeline.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Please run data_collection_preprocessing.py first."
        )

    data = torch.load(path)

    if isinstance(data, list) and isinstance(data[0], dict):
        print(f"Detected {len(data)} batches in {path}")
        all_ids = []
        for batch in data:
            ids = batch["input_ids"].view(-1).tolist()
            all_ids.extend(ids)
        print(f"Flattened total tokens: {len(all_ids):,}")
        return all_ids

    if isinstance(data, torch.Tensor):
        return data.view(-1).tolist()
    if isinstance(data, list):
        if isinstance(data[0], int):
            return data
        else:
            raise ValueError("Unsupported nested list structure in token file.")

    raise ValueError("Unsupported format for sample dataset file.")

# 6. Execution
def main():
    cfg = Config()
    tokens = load_tokenized_data('sample_batches.pt') 
    dataset = TokenDataset(tokens, cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = MiniGPT(cfg).to(cfg.device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    loss_history = train_model(model, dataloader, cfg)
    plot_loss(loss_history)


if __name__ == "__main__":
    main()