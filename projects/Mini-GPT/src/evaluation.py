import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import argparse
from model import MiniGPT
from clm_dataset import CLMDataset

def plot_loss_log(log_file):

    # Read the log file
    loss_log = []
    with open(log_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(",")
            loss = float(parts[1])
            loss_log.append(loss)

    # Plot training loss
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6), dpi=150)

    # Plot training loss
    plt.plot(loss_log, label="Training Loss", color="royalblue", linewidth=2, alpha=0.8)

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training Loss Over Iterations", fontsize=16, fontweight='bold')

    plt.legend(fontsize=12, loc="upper right", frameon=True, fancybox=True, shadow=True)

    # Save the plot
    plot_filename = "training_loss_plot.png"
    plt.savefig(plot_filename)

    plt.show()



def run_demo_inference(model, tokenizer, prompt, device, max_length, temperature=0.5):
    
    model.eval()
    
    # Encode the input prompt using the tokenizer
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded]).to(device) 

    generated = input_ids
    for _ in range(max_length-input_ids.shape[1]):
        with torch.no_grad():
            logits = model(generated)
            logits = logits[:, -1, :]  # Get the logits for the last token
            
            # Apply temperature (higher values = more randomness)
            logits = logits / temperature
            
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Use argmax to pick the token with the highest probability (greedy search)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
                break

    generated_text = tokenizer.decode(generated[0].cpu().numpy().tolist())

    return generated_text


def plot_loss(log_file):
    plot_loss_log(log_file)


def demo_inference(model_checkpoint, prompt, max_length=100):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load the model from checkpoint
    model = MiniGPT(vocab_size=tokenizer.vocab_size,
                    seq_len=128,
                    embed_dim=768,
                    num_heads=12,
                    num_layers=12).to(device)

    # Load model state_dict
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run demo inference
    generated_text = run_demo_inference(model, tokenizer, prompt, device, max_length=max_length)
    print("\nGenerated Text:")
    print(generated_text)


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--mode', type=str, required=True, choices=['plot', 'inference'],
                        help="Choose between 'plot' for loss plot or 'inference' for running a demo inference")
    parser.add_argument('--log_file', type=str, help="Path to the loss log file", default="training_loss.log")
    parser.add_argument('--model_checkpoint', type=str, help="Path to the model checkpoint", default="checkpoints/checkpoint.pt")
    parser.add_argument('--prompt', type=str, help="Prompt to generate text from", default="The theory of evolution by natural selection was proposed by")
    parser.add_argument('--max_length', type=int, help="Max length of generated text", default=128)

    args = parser.parse_args()

    if args.mode == 'plot':
        plot_loss(args.log_file)
    elif args.mode == 'inference':
        demo_inference(args.model_checkpoint, args.prompt, args.max_length)


if __name__ == "__main__":
    main()
