import os
import subprocess
import numpy as np

def run_experiment():
    """
    Runs a series of fine-tuning and rollout experiments with different learning rates
    for the AdamW optimizer.
    """
    # --- Configuration ---
    # Define the learning rates to test
    learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    num_epochs = 5 # Number of training epochs for each experiment

    # Define base directories for outputs, ensuring they are relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_train_log_dir = os.path.join(script_dir, "../train_log/AdamW_lr_exp")
    base_result_dir = os.path.join(script_dir, "../result/AdamW_lr_exp")
    
    # Path to the scripts to be called
    finetune_script_path = os.path.join(script_dir, "finetune.py")
    rollout_script_path = os.path.join(script_dir, "rollout.py")

    print("Starting batch experiment for AdamW with different learning rates...")
    print(f"Learning rates to be tested: {learning_rates}")
    print("-" * 50)

    # --- Experiment Loop ---
    for lr in learning_rates:
        lr_str = f"{lr:.0e}" # Format learning rate for directory name (e.g., 1e-05)
        print(f"\n[INFO] Running experiment for Learning Rate: {lr}")

        # 1. Define output directories for the current experiment
        model_output_dir = os.path.join(base_train_log_dir, f"lr_{lr_str}")
        result_output_dir = os.path.join(base_result_dir, f"lr_{lr_str}")

        # Create directories if they don't exist
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(result_output_dir, exist_ok=True)
        print(f"  - Model/Logs will be saved to: {model_output_dir}")
        print(f"  - Results will be saved to: {result_output_dir}")

        # 2. Run Fine-tuning
        print("\n  -> Step 1: Starting fine-tuning...")
        finetune_command = [
            "python", finetune_script_path,
            "--optimization_method", "adam",
            "--learning_rate", str(lr),
            "--num_epochs", str(num_epochs),
            "--output_dir", model_output_dir,
            "--save_plot_path", os.path.join(result_output_dir, "loss_curve.png"),
            "--plot"
        ]
        
        try:
            # Run the subprocess and allow its output to stream to the console in real-time.
            # Removed capture_output=True to see live progress bars.
            subprocess.run(finetune_command, check=True)
            print("  -> Fine-tuning completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Fine-tuning failed for learning rate {lr}.")
            print(f"  - Return Code: {e.returncode}")
            print("  - Check the output above for more details.")
            print("  - Skipping rollout for this learning rate.")
            print("-" * 50)
            continue # Move to the next learning rate

        # 3. Run Rollout (Inference)
        print("\n  -> Step 2: Starting rollout (inference)...")
        rollout_command = [
            "python", rollout_script_path,
            "--lora_path", model_output_dir,
            "--output_file", os.path.join(result_output_dir, "answers.jsonl")
        ]

        try:
            # Run the subprocess and allow its output to stream to the console.
            subprocess.run(rollout_command, check=True)
            print("  -> Rollout completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Rollout failed for learning rate {lr}.")
            print(f"  - Return Code: {e.returncode}")
            print("  - Check the output above for more details.")
        
        print(f"[INFO] Finished experiment for Learning Rate: {lr}")
        print("-" * 50)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    run_experiment()
