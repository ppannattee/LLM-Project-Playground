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

def save_checkpoint(model, optimizer, iteration, checkpoint_dir, device):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)

    tqdm.write(f"\n### Checkpoint saved at iteration {iteration} ###")

    # Generate sample text to verify training progress
    sample_prompt = "The dog ran across the yard"
    generated_text = generate_text(model, sample_prompt, max_new_tokens=20, device=device)
    tqdm.write(f"Sample Inference: {generated_text}")

def write_log(log_file, log_list):
    with open(log_file, "a") as log_f:
        log_f.write("\n".join(log_list) + "\n")
    log_list.clear()  # Clear the log list after writing to the file

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

def train(model, dataloader, optimizer, scheduler, loss_fn, device, num_epochs, total_iterations, 
          log_file, checkpoint_interval, checkpoint_dir, grad_accum_steps):
    model.to(device)
    model.train()
    iteration = 0
    log_list = []

    progress_bar = tqdm(total=total_iterations, desc="Training", dynamic_ncols=True)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            if iteration >= total_iterations:
                break

            input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:  
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iteration += 1
            progress_bar.set_postfix(loss=f"{(loss.item() * grad_accum_steps):.4f}")
            progress_bar.update(1)

            log_list.append(f"{iteration},{(loss.item() * grad_accum_steps):.6f}")

            if iteration % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, iteration, checkpoint_dir, device)
                write_log(log_file, log_list)

    save_checkpoint(model, optimizer, iteration, checkpoint_dir, device)
    write_log(log_file, log_list)
    progress_bar.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a MiniGPT model on CLM dataset")

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Number of steps for gradient accumulation")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--loss_log_file", type=str, default="training_loss.log")
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CLMDataset(block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MiniGPT(vocab_size=dataset.get_vocab_size(),
                    seq_len=args.block_size,
                    embed_dim=args.embed_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_iterations = len(dataloader) * args.num_epochs
    warmup_steps = int(0.02 * total_iterations)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_iterations)

    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, scheduler, loss_fn, device, args.num_epochs, total_iterations,
          args.loss_log_file, args.checkpoint_interval, args.checkpoint_dir, args.grad_accum_steps)

if __name__ == "__main__":
    main()