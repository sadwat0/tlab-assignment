import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from trl import RewardTrainer, RewardConfig
from src.utils import get_tokenizer, format_reward_dataset, load_and_split_dataset
from src.config import (
    WANDB_LOGGING,
    IS_ON_KAGGLE,
    MODEL_NAME,
    REWARD_MODEL_OUTPUT_DIR,
    REWARD_TRAIN_EPOCHS,
    REWARD_LR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    MAX_LENGTH,
    SECOND_REWARD_MODEL_OUTPUT_DIR,
)

if WANDB_LOGGING:
    import wandb

# pylint: disable=import-error
if IS_ON_KAGGLE and WANDB_LOGGING:
    from kaggle_secrets import UserSecretsClient  # type: ignore


class CustomRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Updated compute_loss:
            L_RM = - log (sum_{i=2}^{10} sum_{j=1}^{i-1} w_i * l_j)
        """
        logits_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        logits_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]

        winner_probabilities = F.softmax(logits_chosen, dim=-1)
        loser_probabilities = F.softmax(logits_rejected, dim=-1)

        # mask[i, j] = float(reward(i) > reward(j))
        device = logits_chosen.device
        rating_range = torch.arange(10, device=device)
        mask = (
            rating_range.view(-1, 1) > rating_range.view(1, -1)
        ).float()  # shape: [10, 10]

        # [batch_size, 10, 10]
        prod = winner_probabilities.unsqueeze(2) * loser_probabilities.unsqueeze(1)
        prob = (prod * mask).sum(dim=(1, 2))  # shape: [batch_size]

        loss = -torch.log(prob + 1e-8).mean()

        if return_outputs:
            return loss, {
                "logits_chosen": logits_chosen,
                "logits_rejected": logits_rejected,
            }
        return loss


def compute_metrics(eval_pred):
    logits_chosen, logits_rejected = eval_pred

    logits_chosen = torch.tensor(logits_chosen)
    logits_rejected = torch.tensor(logits_rejected)

    ratings = torch.arange(1, 11, dtype=torch.float32, device=logits_chosen.device)

    probs_chosen = torch.softmax(logits_chosen, dim=-1)
    probs_rejected = torch.softmax(logits_rejected, dim=-1)

    chosen = (probs_chosen * ratings).sum(dim=-1)
    rejected = (probs_rejected * ratings).sum(dim=-1)

    accuracy = (chosen > rejected).float().mean().item()

    return {"accuracy": accuracy}


def train_reward_model(output_model_dir):
    if WANDB_LOGGING:
        # pylint: disable=used-before-assignment
        if IS_ON_KAGGLE:
            user_secrets = UserSecretsClient()
            my_secret = user_secrets.get_secret("wandb_api_key")
            wandb.login(key=my_secret)

        wandb.init(project="tlab-reward-model-training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=10
    ).to(device)

    for param in model.parameters():
        param.requires_grad = True

    dataset = load_and_split_dataset(test_size=0.2)
    dataset = dataset.map(lambda ex: format_reward_dataset(ex, tokenizer))

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    training_args = RewardConfig(
        output_dir=output_model_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=REWARD_TRAIN_EPOCHS,
        learning_rate=REWARD_LR,
        gradient_accumulation_steps=1,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=25,
        max_length=MAX_LENGTH,
        remove_unused_columns=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = CustomRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print("Initial eval metrics:", metrics)
    if WANDB_LOGGING:
        wandb.log(metrics)

    trainer.train()

    trainer.save_model(output_model_dir)
    print("Reward model saved to:", output_model_dir)

    metrics = trainer.evaluate()
    print("Post-training eval metrics:", metrics)
    if WANDB_LOGGING:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--output_model_dir",
        type=str,
        default=SECOND_REWARD_MODEL_OUTPUT_DIR,
        help="Directory to save trained model",
    )
    args = parser.parse_args()
    train_reward_model(output_model_dir=args.output_model_dir)
