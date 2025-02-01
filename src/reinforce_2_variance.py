import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from src.config import (
    WANDB_LOGGING,
    IS_ON_KAGGLE,
    MODEL_NAME,
    SECOND_REWARD_MODEL_OUTPUT_DIR,
    REINFORCE_BATCH_SIZE,
    REINFORCE_LR,
    REINFORCE_MAX_NEW_TOKENS,
    MAX_LENGTH,
    REINFORCE_NUM_EPOCHS,
)
from src.utils import get_tokenizer, format_reinforce_dataset, load_and_split_dataset

if WANDB_LOGGING:
    import wandb

# pylint: disable=import-error
if IS_ON_KAGGLE and WANDB_LOGGING:
    from kaggle_secrets import UserSecretsClient  # type: ignore


def train_reinforce(reward_model_path, output_model_dir):
    if WANDB_LOGGING:
        # pylint: disable=used-before-assignment
        if IS_ON_KAGGLE:
            user_secrets = UserSecretsClient()
            my_secret = user_secrets.get_secret("wandb_api_key")
            wandb.login(key=my_secret)

        wandb.init(
            project="tlab-reinforce-with-baseline",
            config={
                "batch_size": REINFORCE_BATCH_SIZE,
                "learning_rate": REINFORCE_LR,
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    reward_tokenizer = get_tokenizer(MODEL_NAME)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path
    ).to(device)
    reward_model.eval()

    dataset = load_and_split_dataset(
        test_size=0.2, train_subset_sizes=1000, val_subset_size=300
    )
    train_dataset = dataset["train"].map(
        lambda ex: format_reinforce_dataset(ex, tokenizer)
    )
    val_dataset = dataset["test"].map(
        lambda ex: format_reinforce_dataset(ex, tokenizer)
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(
        train_dataset, batch_size=REINFORCE_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=REINFORCE_BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)

    def compute_rewards_and_variance(generated_sequences, input_ids):
        texts = reward_tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )

        inputs = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = reward_model(**inputs).logits  # shape: [batch_size, num_labels=10]
            probabilities = torch.softmax(logits, dim=-1)
            ratings = torch.arange(
                1, 11, dtype=probabilities.dtype, device=probabilities.device
            )

            # E[r]
            expected_rewards = (probabilities * ratings).sum(dim=-1)
            # E[r^2]
            expected_square = (probabilities * (ratings**2)).sum(dim=-1)
            # D[r] = E[r^2] - (E[r])^2
            variance = expected_square - expected_rewards.pow(2)

        return expected_rewards, variance

    def evaluate(model, dataloader):
        model.eval()
        all_rewards = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=REINFORCE_MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
                rewards, _ = compute_rewards_and_variance(generated, input_ids)
                all_rewards.extend(rewards.cpu().numpy())
        return np.mean(all_rewards), all_rewards

    mean_reward_before, all_rewards_before = evaluate(model, val_loader)
    print(f"Initial mean reward: {mean_reward_before:.4f}")
    if WANDB_LOGGING:
        wandb.log(
            {
                "eval/mean_reward_before": mean_reward_before,
                "eval/all_rewards_before": all_rewards_before,
            }
        )

    # main algorithm

    total_rewards = 0.0
    step_count = 0
    baseline = 0.0

    for epoch in range(REINFORCE_NUM_EPOCHS):
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=REINFORCE_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            full_sequences = outputs.sequences  # input + generated
            generated_tokens = full_sequences[:, input_ids.shape[1] :]

            rewards, rewards_variance = compute_rewards_and_variance(
                full_sequences, input_ids
            )

            # calculating log of probabilities
            gen_attention_mask = torch.ones(
                generated_tokens.shape, dtype=torch.long, device=device
            )
            full_attention_mask = torch.cat([attention_mask, gen_attention_mask], dim=1)

            # [batch_size, sequence_length, vocab_size]
            logits = model(full_sequences, attention_mask=full_attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)

            # [batch_size, generated_length]
            selected_log_probs = log_probs.gather(
                -1, generated_tokens.unsqueeze(-1)
            ).squeeze(-1)
            aggregated_log_probs = selected_log_probs.sum(dim=1)

            # calculating adantage
            total_rewards += rewards.sum().item()
            step_count += rewards.shape[0]
            baseline = total_rewards / step_count
            baseline_tensor = torch.tensor(baseline, device=device)
            advantages = rewards - baseline_tensor

            # added D[r]
            loss = -(
                advantages * aggregated_log_probs / torch.sqrt(rewards_variance + 1e-8)
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if WANDB_LOGGING:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/avg_reward": rewards.mean().item(),
                        "train/avg_advantage": advantages.mean().item(),
                        "train/baseline": baseline,
                        "train/step": batch_idx,
                    }
                )

            if batch_idx % 10 == 0:
                print(
                    f"Batch {batch_idx}: loss: {loss.item():.4f}, avg_reward: {rewards.mean().item():.4f}"
                )

    mean_reward_after, all_rewards_after = evaluate(model, val_loader)
    print(f"Final mean reward: {mean_reward_after:.4f}")
    if WANDB_LOGGING:
        wandb.log(
            {
                "eval/mean_reward_after": mean_reward_after,
                "eval/all_rewards_after": all_rewards_after,
            }
        )
        wandb.finish()

    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    print("Model saved to:", output_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using reinforce w/ baseline")
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=SECOND_REWARD_MODEL_OUTPUT_DIR,
        help="Path to the pretrained reward model",
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        default="./reinforce_with_alignment_v1",
        help="Directory to save trained model",
    )
    args = parser.parse_args()
    train_reinforce(
        reward_model_path=args.reward_model_path, output_model_dir=args.output_model_dir
    )
