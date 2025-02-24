import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
from transformers import GPT2Tokenizer

from clm_dataset import CLMDataset
from model import MiniGPT

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

def save_checkpoint(model, optimizer, iteration, checkpoint_dir, device):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)

    tqdm.write(f"\nCheckpoint saved at iteration {iteration}: {checkpoint_path}")

    # Generate sample text to verify training progress
    sample_prompt = "The dog ran across the yard"
    generated_text = generate_text(model, sample_prompt, max_new_tokens=20, device=device)
    tqdm.write(f"Sample Inference: {generated_text}")

def write_log(log_file, log_list):
    with open(log_file, "a") as log_f:
        log_f.write("\n".join(log_list) + "\n")
    log_list.clear()  # Clear the log list after writing to the file

# Function to generate text using the trained model
def generate_text(model, prompt, max_new_tokens=20, device="cuda"):
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :]  # Get logits for last token
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)  # Greedy decoding
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    return tokenizer.decode(input_ids[0])

# Training function
def train(model, dataloader, optimizer, scheduler, loss_fn, device, num_epochs, total_iterations, log_file, checkpoint_interval, checkpoint_dir):
    model.to(device)
    model.train()
    iteration = 0
    log_list = []

    progress_bar = tqdm(total=total_iterations, desc="Training", dynamic_ncols=True)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            if iteration >= total_iterations:
                break

            input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            iteration += 1

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            progress_bar.update(1)

            log_list.append(f"{iteration},{loss.item():.6f}")

            # Save model checkpoint and write to log file at specified intervals
            if iteration % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, iteration, checkpoint_dir, device)
                write_log(log_file, log_list)

    save_checkpoint(model, optimizer, iteration, checkpoint_dir)
    write_log(log_file, log_list)

    progress_bar.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a MiniGPT model on CLM dataset")

    parser.add_argument("--block_size", type=int, default=128, help="Size of token chunks")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension of the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--loss_log_file", type=str, default="training_loss.log", help="File to store loss logs")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save model checkpoint every X iterations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")

    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    # Set seed
    set_seed()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    dataset = CLMDataset(block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize MiniGPT
    model = MiniGPT(vocab_size=dataset.get_vocab_size(),
                    seq_len=args.block_size,
                    embed_dim=args.embed_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_iterations = len(dataloader) * args.num_epochs
    warmup_steps = int(0.1 * total_iterations)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_iterations)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Start training
    train(model, dataloader, optimizer, scheduler, loss_fn, device, args.num_epochs, total_iterations, 
          args.loss_log_file, args.checkpoint_interval, args.checkpoint_dir)

# Run script
if __name__ == "__main__":
    main()
