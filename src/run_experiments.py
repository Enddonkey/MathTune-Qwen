import os
import subprocess
import sys
from datetime import datetime

def run_experiment():
    """
    Runs a series of fine-tuning, rollout, and evaluation experiments.
    Allows switching between logging to a file or printing to the console.
    """
    # --- Configuration ---
    # Set to True to log all subprocess output to a file in the result directory.
    # Set to False to see live progress bars and output in the console.
    LOG_TO_FILE = True

    learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    num_epochs = 1

    # --- Path Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_train_log_dir = os.path.join(script_dir, "../train_log/AdamW_lr_exp")
    base_result_dir = os.path.join(script_dir, "../result/AdamW_lr_exp")
    
    finetune_script_path = os.path.join(script_dir, "finetune.py")
    rollout_script_path = os.path.join(script_dir, "rollout.py")
    evaluate_script_path = os.path.join(script_dir, "evaluate.py")

    print("Starting batch experiment for AdamW with different learning rates...")
    print(f"Logging mode: {'File' if LOG_TO_FILE else 'Console'}")
    print(f"Learning rates to be tested: {learning_rates}")
    print("-" * 50)

    # --- Experiment Loop ---
    for lr in learning_rates:
        lr_str = f"{lr:.0e}"
        
        # --- Directory and Log File Setup ---
        model_output_dir = os.path.join(base_train_log_dir, f"lr_{lr_str}")
        result_output_dir = os.path.join(base_result_dir, f"lr_{lr_str}")
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(result_output_dir, exist_ok=True)
        
        log_file_path = os.path.join(result_output_dir, "execution.log")
        
        # --- Main Logic ---
        print(f"\n--- Starting Experiment for LR={lr} at {datetime.now()} ---")

        try:
            # Open log file only if in file logging mode
            log_file = open(log_file_path, 'w', encoding='utf-8') if LOG_TO_FILE else None
            
            # Determine output stream
            output_stream = log_file if LOG_TO_FILE else None

            # --- Step 1: Fine-tuning ---
            print("[STEP 1/3] Starting fine-tuning...")
            finetune_command = [
                sys.executable, finetune_script_path,
                "--optimization_method", "adam",
                "--learning_rate", str(lr),
                "--num_epochs", str(num_epochs),
                "--output_dir", model_output_dir,
                "--save_plot_path", os.path.join(result_output_dir, "loss_curve.png"),
                "--log_interval", "1",
                "--eval_interval", "1",
                "--plot"
            ]
            subprocess.run(finetune_command, check=True, stdout=output_stream, stderr=output_stream)
            print("  -> Fine-tuning completed successfully.")

            # --- Step 2: Rollout (Inference) ---
            print("[STEP 2/3] Starting rollout (inference)...")
            absolute_model_path = os.path.abspath(model_output_dir)
            answers_file_path = os.path.join(result_output_dir, "answers.jsonl")
            
            rollout_command = [
                sys.executable, rollout_script_path,
                "--lora_path", absolute_model_path,
                "--output_file", answers_file_path
            ]
            subprocess.run(rollout_command, check=True, stdout=output_stream, stderr=output_stream)
            print("  -> Rollout completed successfully.")

            # --- Step 3: Evaluation ---
            print("[STEP 3/3] Starting evaluation...")
            scored_answers_file_path = os.path.join(result_output_dir, "scored_answers.jsonl")
            
            evaluate_command = [
                sys.executable, evaluate_script_path,
                "--input_file", answers_file_path,
                "--output_file", scored_answers_file_path
            ]
            subprocess.run(evaluate_command, check=True, stdout=output_stream, stderr=output_stream)
            print("  -> Evaluation completed successfully.")

        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] A step failed with exit code {e.returncode} for LR={lr}.")
            print("  -> Check the console output or the log file for details.")
            print("  -> Skipping subsequent steps for this learning rate.")
        
        finally:
            if log_file:
                log_file.close()
            print(f"--- Finished Experiment for LR={lr} at {datetime.now()} ---")
            print("-" * 50)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    run_experiment()
