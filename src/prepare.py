# =====================================================================================
# A. SCRIPT SETUP & IMPORTS
# =====================================================================================
import argparse
import os
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# =====================================================================================
# B. CORE DATA PREPARATION FUNCTION
# =====================================================================================


def prepare_tokenize_and_save(
    dataset_name: str,
    tokenizer_name: str,
    prompt_column: str,
    response_column: str,
    max_length: int,
    num_proc: int,
    debug_mode: bool,
    output_dir: str,
) -> None:
    """
    Handles the entire data pipeline: loading, tokenizing, splitting,
    and saving the data for training.
    """
    # -----------------------------------------------------------------
    # 1. Load Tokenizer and Dataset from Hugging Face Hub
    # -----------------------------------------------------------------
    print(f"Loading tokenizer: '{tokenizer_name}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Loading dataset: '{dataset_name}'")
    original_dataset = load_dataset(dataset_name, split="train")

    if debug_mode:
        num_samples = len(original_dataset) // 20
        original_dataset = original_dataset.shuffle(seed=42).select(range(num_samples))
        print(
            f"--- DEBUG MODE: Using {len(original_dataset)} samples (5% of original) ---"
        )

    # -----------------------------------------------------------------
    # 2. Split the Dataset (90% train, 10% validation)
    # -----------------------------------------------------------------
    split_dataset = original_dataset.train_test_split(
        test_size=0.1, shuffle=True, seed=42
    )
    print(
        f"Dataset split into {len(split_dataset['train'])} training and {len(split_dataset['test'])} validation examples."
    )

    # -----------------------------------------------------------------
    # 3. Define the Tokenization Logic
    # -----------------------------------------------------------------
    def tokenize_and_format(examples):
        """
        Processes a batch of examples to prepare them for supervised fine-tuning.
        This now correctly includes the attention_mask.
        """
        prompts = examples[prompt_column]
        responses = examples[response_column]

        # TODO:
        # We need to convert the chat templates to strings first to use the main tokenizer call
        math_instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        full_chats = [
            [
                {"role": "user", "content": p + math_instruction},
                {"role": "assistant", "content": r},
            ]
            for p, r in zip(prompts, responses)
        ]
        full_texts = [
            tokenizer.apply_chat_template(conversation=chat,
                                          tokenize=False,
                                          add_generation_prompt=False,
                                          enable_thinking=False)
            for chat in full_chats
        ]


        # Tokenize prompts separately to calculate their length for masking
        prompt_only_chats = [
            [{"role": "user", "content": p + math_instruction}] for p in prompts
        ]
        prompt_texts = [
            tokenizer.apply_chat_template(conversation=chat,
                                          tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
            for chat in prompt_only_chats 
        ]

        # Use the tokenizer's main `__call__` method to get input_ids AND attention_mask
        tokenized_outputs = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding=False,  # Collator will handle padding
        )
        tokenized_prompts = tokenizer(
            prompt_texts, max_length=max_length, truncation=True
        )

        # Create labels by masking prompt tokens
        labels_list = []
        for i, full_ids in enumerate(tokenized_outputs["input_ids"]):
            prompt_len = len(tokenized_prompts["input_ids"][i])
            label = list(full_ids)  # Copy input_ids
            label[:prompt_len] = [-100] * prompt_len  # Mask prompt when calculating loss
            labels_list.append(label)

        # Add labels to our dictionary
        tokenized_outputs["labels"] = labels_list
        return tokenized_outputs

    # -----------------------------------------------------------------
    # 4. Apply Tokenization and Filter Long Sequences
    # -----------------------------------------------------------------
    print("\nTokenizing and formatting datasets...")
    # The `map` function will now create 'input_ids', 'attention_mask', and 'labels' columns
    remove_cols = split_dataset["train"].column_names
    train_dataset = split_dataset["train"].map(
        tokenize_and_format, batched=True, num_proc=num_proc, remove_columns=remove_cols
    )
    val_dataset = split_dataset["test"].map(
        tokenize_and_format, batched=True, num_proc=num_proc, remove_columns=remove_cols
    )

    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) < max_length, num_proc=num_proc
    )
    val_dataset = val_dataset.filter(
        lambda x: len(x["input_ids"]) < max_length, num_proc=num_proc
    )

    print(f"Final training samples: {len(train_dataset)}")
    print(f"Final validation samples: {len(val_dataset)}")

    # -----------------------------------------------------------------
    # 5. Extract FULL RECORDS and Save to Pickle Files
    # -----------------------------------------------------------------
    # Instead of just getting input_ids, convert the entire dataset to a list of dictionaries.
    # Each dictionary will be a complete example: {'input_ids': [...], 'attention_mask': [...], 'labels': [...]}
    train_data = train_dataset.to_list()
    val_data = val_dataset.to_list()

    # --- Print Final Statistics ---
    print("\n--- DATASET STATS ---")
    if train_data:
        # We now access the input_ids from the first dictionary in the list
        text_sample = tokenizer.decode(
            train_data[0]["input_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print("Sample of first training example:\n", text_sample)

        all_lengths = [len(x["input_ids"]) for x in train_data] + [
            len(x["input_ids"]) for x in val_data
        ]
        print(f"\nMax sequence length: {max(all_lengths)}")
        print(f"Average sequence length: {sum(all_lengths) / len(all_lengths):.2f}")

    # --- Save Files ---
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the 'data' directory exists

    train_output_path = os.path.join(output_dir, "train.pkl")
    val_output_path = os.path.join(output_dir, "val.pkl")

    print(f"\nSaving training data to '{train_output_path}'...")
    with open(train_output_path, "wb") as f:
        pickle.dump(train_data, f)

    print(f"Saving validation data to '{val_output_path}'...")
    with open(val_output_path, "wb") as f:
        pickle.dump(val_data, f)

    print("--- PREPARATION COMPLETE ---")


# =====================================================================================
# C. SCRIPT EXECUTION
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Flexible SFT Data Preparation Script")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Directory to save the tokenized pickle files.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\Qwen\\Qwen3-0___6B-Base",
        help="Hugging Face model for the tokenizer.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ricdomolm/MATH-500",
        help="Dataset name on Hugging Face Hub.",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="problem",
        help="The name of the column for prompts/questions.",
    )
    parser.add_argument(
        "--response_column",
        type=str,
        default="solution",
        help="The name of the column for responses/answers.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for truncation.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=os.cpu_count(),
        help="Number of CPU cores for parallel processing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode on a small subset of the data.",
    )

    args = parser.parse_args()

    prepare_tokenize_and_save(
        dataset_name=args.dataset_name,
        tokenizer_name=args.model_name_or_path,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
        max_length=args.max_length,
        num_proc=args.num_proc,
        debug_mode=args.debug,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
