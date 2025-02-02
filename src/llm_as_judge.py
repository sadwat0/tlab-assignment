import sys
import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import MODEL_NAME, REINFORCE_MAX_NEW_TOKENS
from src.utils import get_tokenizer, format_reinforce_dataset, load_and_split_dataset
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Comparative evaluation of two models using Gemini."
    )
    parser.add_argument(
        "--model1_path",
        type=str,
        help="Path to the first model.",
    )
    parser.add_argument(
        "--model2_path",
        type=str,
        help="Path to the second model.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./evaluation_results.csv",
        help="CSV file to save the evaluation results.",
    )
    args = parser.parse_args()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing in the environment variables.")
    genai.configure(api_key=GEMINI_API_KEY)
    evaluation_model = genai.GenerativeModel("gemini-2.0-flash-exp")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_tokenizer(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading models for comparison...")
    model1 = AutoModelForCausalLM.from_pretrained(args.model1_path).to(device)
    model1.eval()
    model2 = AutoModelForCausalLM.from_pretrained(args.model2_path).to(device)
    model2.eval()

    dataset = load_and_split_dataset(test_size=0.2)
    val_dataset = dataset["test"].map(
        lambda ex: format_reinforce_dataset(ex, tokenizer)
    )
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_examples = []
    for ex in val_dataset:
        prompt_text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        val_examples.append(
            {
                "prompt": prompt_text,
                "input_ids": ex["input_ids"],
                "attention_mask": ex["attention_mask"],
            }
        )

        if len(val_examples) >= 100:
            break

    results = []
    random.seed(42)
    for example in tqdm(val_examples, desc="Evaluating examples"):
        prompt_text = example["prompt"]
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            output1 = model1.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=REINFORCE_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )
            answer1 = tokenizer.decode(output1[0], skip_special_tokens=True)
            output2 = model2.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=REINFORCE_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )
            answer2 = tokenizer.decode(output2[0], skip_special_tokens=True)
        candidates = [("model1", answer1), ("model2", answer2)]
        random.shuffle(candidates)
        displayed_answer1 = candidates[0]
        displayed_answer2 = candidates[1]
        eval_prompt = (
            "Evaluate the two answers for the given prompt. First, provide a detailed explanation for your choice, "
            "then on a new line output only an integer (with no additional text) from the following options:\n\n"
            "0: [Answer 1] is significantly better,\n"
            "1: [Answer 1] is somewhat better,\n"
            "2: Tie,\n"
            "3: [Answer 2] is somewhat better,\n"
            "4: [Answer 2] is significantly better.\n\n"
            f"Prompt: {prompt_text}\n\n"
            f"[Answer 1]: \n```{displayed_answer1[1]}```\n\n"
            f"[Answer 2]: \n```{displayed_answer2[1]}```\n\n"
            "Please first provide your explanation, then on a new line output the number (0-4). "
            "Note: Try to avoid choosing 'Tie' (option 2) unless the two answers are virtually identical."
        )

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        try:
            response = evaluation_model.generate_content(
                eval_prompt,
                generation_config=generation_config,
            )
            response_text = response.text
            lines = response_text.strip().splitlines()
            if not lines:
                raise ValueError("Empty response from evaluation model.")
            decision_line = ""
            for line in reversed(lines):
                if line.strip():
                    decision_line = line.strip()
                    break
            try:
                decision = int(decision_line)
            except ValueError:
                decision = 2  # default = tie

            explanation_lines = lines[:-1] if len(lines) > 1 else []
            eval_explanation = "\n".join(explanation_lines).strip()
        except Exception as e:
            print(f"Exception: {e}")
            eval_explanation = f"Evaluation error: {e}"
            decision = 2

        if decision in [0, 1]:
            winner = displayed_answer1[0]
        elif decision in [3, 4]:
            winner = displayed_answer2[0]
        else:
            winner = "draw"

        if decision == 0:
            win_type = "strong win " + displayed_answer1[0]
        elif decision == 1:
            win_type = "slight win " + displayed_answer1[0]
        elif decision == 3:
            win_type = "slight win " + displayed_answer2[0]
        elif decision == 4:
            win_type = "strong win " + displayed_answer2[0]
        else:
            win_type = "draw"

        results.append(
            {
                "prompt": prompt_text,
                "model1_answer": answer1,
                "model2_answer": answer2,
                "display_order": f"First: {displayed_answer1[0]}, Second: {displayed_answer2[0]}",
                "eval_explanation": eval_explanation,
                "decision": decision,
                "winner": winner,
                "win_type": win_type,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Evaluation results saved to {args.output_csv}")
    win_counts = df["winner"].value_counts()
    win_type_counts = df["win_type"].value_counts()
    print("Win statistics by winner:")
    print(win_counts)
    print("Win statistics by win type:")
    print(win_type_counts)


if __name__ == "__main__":
    main()
