import json
import argparse
from pathlib import Path
from verifier import compute_score


def evaluate_results(input_file, output_file):
    """
    Reads a JSONL file of model generations, computes scores, and saves the enhanced results to a new JSONL file.
    """
    total_score = 0
    total_count = 0

    input_path = Path(input_file)
    if not input_path.is_file():
        print(f"Error: Input file not found at: {input_file}")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            
            model_answer = data["answer"]
            gold_answer = data["gold"]
            
            try:
                score_dict = compute_score(model_answer, gold_answer)
                data.update(score_dict) # Merge score results (including extracted_pred)
                
                if score_dict:
                    total_score += score_dict.get("score", 0.0)
                    total_count += 1
                    
            except Exception as e:
                print(f"Error processing line {total_count + 1}: {e}")
                data["error"] = str(e)

            # Write the combined data (original + scores) to the new file
            f_out.write(json.dumps(data) + '\n')

    if total_count > 0:
        accuracy = (total_score / total_count) * 100
        print(f"\nEvaluation complete.")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Scored results saved to {output_file}")
    else:
        print("No lines were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model generations from a JSONL file.")
    parser.add_argument("--input_file", type=str, required=False, default="../result/answer.jsonl", help="The path to the input JSONL file (generations).")
    parser.add_argument("--output_file", type=str, required=False, default="../result/scored_answer.jsonl", help="The path to the output JSONL file (scored results).")
    args = parser.parse_args()
    evaluate_results(args.input_file, args.output_file)