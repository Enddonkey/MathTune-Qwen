import os
import pickle
import argparse
import json
import time
from tqdm import tqdm

import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def get_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")

    # Model and Data paths
    parser.add_argument('--model_name', type=str, default='C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\Qwen\\Qwen3-0___6B-Base', help='The name of the pretrained model to use.')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory where the data is stored.')
    parser.add_argument('--output_dir', type=str, default='../train_log/out-instruction-tuning', help='Directory to save the fine-tuned model.')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW optimizer beta1.')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW optimizer beta2.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation.')
    parser.add_argument('--grad_accumulation_steps', type=int, default=16, help='Number of steps to accumulate gradients.')

    # Logging and Evaluation
    parser.add_argument('--log_interval', type=int, default=10, help='Log training loss every N steps.')
    parser.add_argument('--eval_interval', type=int, default=50, help='Run validation every N steps.')

    # Optimization method
    parser.add_argument('--optimization_method', type=str, default='adam', choices=['adam', 'sgd', 'lora'], help='Optimization method to use.')
    parser.add_argument('--lora_rank', type=int, default=8, help='The rank of the LoRA matrices.')

    # Add plotting-related command-line arguments
    parser.add_argument('--plot', action='store_true', help='Whether to plot loss curve after training (default: False).')
    parser.add_argument('--log_path_plot', type=str, default=None, help='Path to training_logs.json (if not using default output_dir).')
    parser.add_argument('--save_plot_path', type=str, default='../result/loss_curve.png', help='Path to save the loss plot (default: loss_curve.png).')

    return parser.parse_args()

class TokenizedDataset(Dataset):
    """A simple dataset class to load tokenized IDs from a pickle file."""
    def __init__(self, pickle_file_path):
        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError(
                f"Pickle file not found at {pickle_file_path}. "
                "Please run the data preparation script first."
            )
        with open(pickle_file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} examples from {pickle_file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SmartDataCollator:
    """
    Pads sequences to the max length in a batch and creates labels.
    Labels are -100 for pad tokens.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': padded_labels
        }

def init_json_log(log_path, args):
    """Initializes a JSON log file."""
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    log_data = {
        'args': vars(args),
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'val_step': [],
        'training_time': None,
        'max_memory_allocated_gb': None
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

def update_json_log(log_path, step=None, train_loss=None, val_loss=None, val_step=None, training_time=None, max_memory_gb=None):
    """Updates the JSON log file."""
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    if step is not None:
        log_data['step'].append(step)
    if train_loss is not None:
        log_data['train_loss'].append(round(train_loss, 4))
    if val_loss is not None:
        log_data['val_loss'].append(round(val_loss, 4))
    if val_step is not None:
        log_data['val_step'].append(val_step)
    if training_time is not None:
        log_data['training_time'] = f"{training_time:.2f}s"
    if max_memory_gb is not None:
        log_data['max_memory_allocated_gb'] = round(max_memory_gb, 2)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

def plot_loss(log_path, save_plot_path, optimization_method, num_epochs):
    """Loads JSON logs and plots training/validation loss curve."""
    if not os.path.exists(log_path):
        print(f"Warning: Log file {log_path} not found. Skipping plotting.")
        return
    
    with open(log_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    steps = logs.get('step', [])
    train_losses = logs.get('train_loss', [])
    val_losses = logs.get('val_loss', [])
    val_steps = logs.get('val_step', [])
    
    if not steps or not train_losses:
        print("Warning: No training logs found. Skipping plotting.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label='Training Loss', color='#2E86AB', linewidth=1.8)
    if val_steps and val_losses:
        plt.plot(val_steps, val_losses, label='Validation Loss', color='#A23B72', linewidth=2.2, marker='o', markersize=4.5)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training/Validation Loss (Optimizer: {optimization_method}, Epochs: {num_epochs})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_plot_path, dpi=300)
    print(f"\nLoss curve saved to: {save_plot_path}")
    # plt.show()

def main():
    start_time = time.time()
    args = get_args()

    # Derived paths
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    data_dir = os.path.join(script_dir, args.data_dir)
    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = os.path.join(data_dir, 'val.pkl')
    output_dir = os.path.join(script_dir, args.output_dir)
    log_path = os.path.join(output_dir, 'training_logs.json')
    init_json_log(log_path, args)

    print(f"Loading model and tokenizer from {args.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("Set pad_token to eos_token")

    collate_fn = SmartDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = TokenizedDataset(train_data_path)
    val_dataset = TokenizedDataset(val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Setting up optimizer: {args.optimization_method}")

    # TODO: Apply different optimizer
    if args.optimization_method == "adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimization_method == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimization_method == "lora":
        print(f"Setting up LoRA with rank={args.lora_rank}")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ], # Apply Lora to all possible modules
        )
        model = get_peft_model(model, lora_config)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    else:
        raise ValueError(f"Unknown optimization_method: {args.optimization_method}")

    print("Starting training...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val_loss = float('inf')
    global_step = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / args.grad_accumulation_steps
            loss.backward()
            if (step + 1) % args.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                train_loss = loss.item() * args.grad_accumulation_steps
                if global_step % args.log_interval == 0:
                    print(f"Step {global_step}: Train Loss = {train_loss:.4f}")
                    update_json_log(log_path, step=global_step, train_loss=train_loss)
                if global_step % args.eval_interval == 0:
                    model.eval()
                    print("\nRunning validation...")
                    total_val_loss = 0
                    with torch.no_grad():
                        for val_batch in tqdm(val_loader, desc="Validating"):
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            with torch.autocast(device_type=device, dtype=dtype):
                                val_outputs = model(**val_batch)
                                val_loss = val_outputs.loss
                            total_val_loss += val_loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                    print(f"Step {global_step}: Validation Loss = {avg_val_loss:.4f}")
                    update_json_log(log_path, val_loss=avg_val_loss, val_step=global_step)
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        print(f"  -> New best validation loss! Saving model to {output_dir}")
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                    model.train()

    print("\nTraining finished. Running one final evaluation...")
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Final Validation"):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Final Validation Loss = {avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss:
        print(f"  -> Final model was the best! Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        print(f"  -> An earlier checkpoint was better (Val Loss: {best_val_loss:.4f}). Final model not saved.")

    print(f"\nProcess complete. Best model is saved in {output_dir}")

    # Log total training time and memory usage
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    max_memory_gb = None
    if device == "cuda":
        max_memory_bytes = torch.cuda.max_memory_allocated()
        max_memory_gb = max_memory_bytes / (1024 ** 3)
        print(f"Peak GPU memory allocated: {max_memory_gb:.2f} GB")

    update_json_log(log_path, training_time=total_training_time, max_memory_gb=max_memory_gb)

    if args.plot:
        plot_log_path = args.log_path_plot if args.log_path_plot else log_path
        plot_loss(plot_log_path, args.save_plot_path, args.optimization_method, args.num_epochs)

if __name__ == '__main__':
    main()
