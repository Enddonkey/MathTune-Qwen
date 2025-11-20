import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel # Used for loading LoRA adapters
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def main(model_name, output_filename, lora_path=None):
    """
    Runs the MATH-500 test set evaluation with a specified model and an optional local LoRA adapter
    using the Hugging Face Transformers library, and saves the results to a specified JSONL file.

    Args:
        model_name (str): The name of the Hugging Face model to use (e.g., "Qwen/Qwen3-0.6B-Base").
        output_filename (str): The name of the output JSONL file.
        lora_path (str, optional): The path to the local directory containing the LoRA adapter files. 
                                   Defaults to None.
    """
    # 1. Load MATH-500 test set
    ds = load_dataset("ricdomolm/MATH-500", split="test")
    prompts = ds["problem"]  # question text
    gold_answers = ds["answer"]  # gold answers aligned by index

    # 2. Initialize Tokenizer and Model
    # padding_side='left' to resolve the warning and ensure correct generation
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    
    # Set pad token if it's not set, which is required for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO:
    # Prepare chat-style prompts
    math_instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompt_chats = [
        [
            {"role": "user", "content": p + math_instruction}
        ]
        for p in prompts
    ]

    # Apply chat template to each
    prompt_strs = [
        tokenizer.apply_chat_template(
            conversation=chat,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        for chat in prompt_chats
    ]

    # 3. Create Transformers Model with conditional LoRA support
    print(f"Loading base model: {model_name}")
    
    # The inference speed will be significantly slower if we use float32 here.
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"  # Automatically maps the model to available GPUs
    )

    if lora_path:
        print(f"LoRA adapter specified. Loading from local path: {lora_path}")
        
        # Validate that the path is a valid directory
        if not os.path.isdir(lora_path):
            raise ValueError(f"LoRA path '{lora_path}' is not a valid directory.")
            
        # Load the LoRA adapter and merge it into the base model
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("Successfully merged LoRA adapter into the base model.")
    else:
        print("No LoRA adapter specified, running the base model.")
    
    model.eval() # Set the model to evaluation mode

    # Generation parameters (equivalent to vLLM's SamplingParams)
    generation_kwargs = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_new_tokens": 512,  # Note: transformers uses max_new_tokens
        "repetition_penalty": 1.0,
        "do_sample": True, # Required for temperature and top_p to have an effect
    }

    # 4. Generate in batches
    batch_size = 32 # A smaller batch size is often safer for Transformers to avoid OOM issues
    results = []
    for i in range(0, len(prompt_strs), batch_size):
        batch = prompt_strs[i : i + batch_size]
        
        # Tokenize the batch of prompts
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        # Generate text using the model
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode only the newly generated tokens, skipping the prompt
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Process and store results
        for idx, gen_text in enumerate(generated_texts):
            orig_idx = i + idx
            results.append({
                "id": orig_idx,
                "prompt": prompts[orig_idx],
                "answer": gen_text,
                "gold": gold_answers[orig_idx]
            })
        print(f"Processed batch {i//batch_size + 1}/{(len(prompt_strs) + batch_size - 1)//batch_size}")

    # 5. Save to JSONL
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_path.resolve()}")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"Saved generations to {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MATH-500 evaluation with HF Transformers and an optional local LoRA adapter.")
    parser.add_argument("--model", type=str, required=False,default="C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\Qwen\\Qwen3-0___6B-Base", help="The Hugging Face model to use for generation.")
    parser.add_argument("--output_file", type=str, required=False,default="../result/answer.jsonl", help="The path to the output JSONL file.")
    parser.add_argument("--lora_path", type=str, default="../train_log/out-instruction-tuning", help="Optional: Path to the local directory containing the LoRA adapter files.")
    args = parser.parse_args()
    main(args.model, args.output_file, args.lora_path)