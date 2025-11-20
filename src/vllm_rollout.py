import argparse
import csv
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def main(model_name, output_filename, lora_path=None):
    """
    Runs the MATH-500 test set evaluation with a specified model and an optional local LoRA adapter,
    and saves the results to a specified CSV file.

    Args:
        model_name (str): The name of the Hugging Face model to use (e.g., "Qwen/Qwen3-0.6B-Base").
        output_filename (str): The name of the output CSV file.
        lora_path (str, optional): The path to the local directory containing the LoRA adapter files
                                   (adapter_config.json and adapter_model.safetensors). Defaults to None.
    """
    # 1. Load MATH-500 test set
    ds = load_dataset("ricdomolm/MATH-500", split="test")
    prompts = ds["problem"]  # question text
    gold_answers = ds["answer"]  # gold answers aligned by index

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # TODO:
    # Prepare chat-style prompts
    prompt_chats = [
        [
            {"role": "user", "content": p + XXX}
        ]
        for p in prompts
    ]

    # Apply chat template to each
    prompt_strs = [
        tokenizer.apply_chat_template(
            conversation=XXX,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        for XXX in XXX
    ]

    # 3. Create vLLM LLM with conditional LoRA support
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
    }
    lora_request = None

    if lora_path:
        print(f"LoRA adapter specified. Loading from local path: {lora_path}")
        
        # --- LoRA Specific Setup ---
        # Validate that the path is a valid directory
        if not os.path.isdir(lora_path):
            raise ValueError(f"LoRA path '{lora_path}' is not a valid directory.")

        # Read the LoRA rank 'r' from the adapter_config.json
        try:
            with open(os.path.join(lora_path, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            lora_rank = adapter_config.get("r")
            if lora_rank is None:
                raise ValueError("'r' (lora_rank) not found in adapter_config.json")
            print(f"Successfully read LoRA rank r={lora_rank} from adapter_config.json")
        except FileNotFoundError:
            raise FileNotFoundError(f"adapter_config.json not found in the directory: {lora_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from adapter_config.json in {lora_path}")

        # Prepare LLM engine arguments for LoRA
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1  # We are using one adapter at a time
        llm_kwargs["max_lora_rank"] = lora_rank # Set max rank based on your adapter

        # Create the LoRARequest object to be used during generation
        # The first two arguments are a unique name and a unique integer ID for the adapter.
        lora_request = LoRARequest("my_finetuned_lora", 1, lora_path)
    else:
        print("No LoRA adapter specified, running the base model.")

    # Initialize the LLM with the prepared arguments
    llm = LLM(**llm_kwargs)

    # Sampling parameters (tweak as needed)
    sampling_params = SamplingParams(
        temperature=1,
        top_p=0.95,
        max_tokens=512,
        repetition_penalty=1.0
    )

    # 4. Generate in batches
    batch_size = 64
    results = []
    for i in range(0, len(prompt_strs), batch_size):
        batch = prompt_strs[i : i + batch_size]
        
        # Pass the lora_request to the generate method.
        # If lora_path was not provided, lora_request is None, and vLLM runs the base model.
        outputs = llm.generate(batch, sampling_params, lora_request=lora_request)

        # Process and store results
        for out, orig_idx in zip(outputs, range(i, i + len(batch))):
            gen_text = out.outputs[0].text
            results.append({
                "id": orig_idx,
                "prompt": prompts[orig_idx],
                "answer": gen_text,
                "gold": gold_answers[orig_idx]
            })
        print(f"Processed batch {i//batch_size + 1}/{(len(prompt_strs) + batch_size - 1)//batch_size}")

    # 5. Save to JSONL with columns id, prompt, answer, gold
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_path.resolve()}")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            # Convert each dictionary to a JSON string and write it to a new line
            f.write(json.dumps(row) + "\n")

    print(f"Saved generations to {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MATH-500 evaluation with a specified model and optional local LoRA adapter.")
    parser.add_argument("--model", type=str, required=True, help="The Hugging Face model to use for generation.")
    parser.add_argument("--output_file", type=str, required=True, help="The path to the output CSV file.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional: Path to the local directory containing the LoRA adapter files.")
    args = parser.parse_args()
    main(args.model, args.output_file, args.lora_path)