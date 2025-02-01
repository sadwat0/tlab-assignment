import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from src.custom_reward_trainer import (
    CustomRewardTrainer,
)
from trl import RewardConfig
from src.utils import get_tokenizer, format_reward_dataset, load_and_split_dataset
from src.config import (
    WANDB_LOGGING,
    IS_ON_KAGGLE,
    MODEL_NAME,
    SECOND_REWARD_MODEL_OUTPUT_DIR,
    REWARD_TRAIN_EPOCHS,
    REWARD_LR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    MAX_LENGTH,
)

if WANDB_LOGGING:
    import wandb

# pylint: disable=import-error
if IS_ON_KAGGLE and WANDB_LOGGING:
    from kaggle_secrets import UserSecretsClient  # type: ignore


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    preds = (predictions[:, 0] > predictions[:, 1]).astype(np.int32)
    accuracy = (preds == labels).mean() if labels.size > 0 else 0.0
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

    init_metrics = trainer.evaluate()
    print("Initial eval metrics:", init_metrics)
    if WANDB_LOGGING:
        wandb.log(init_metrics)

    trainer.train()

    trainer.save_model(output_model_dir)
    print("Reward model saved to:", output_model_dir)

    post_metrics = trainer.evaluate()
    print("Post-training eval metrics:", post_metrics)
    if WANDB_LOGGING:
        wandb.log(post_metrics)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-class reward model")
    parser.add_argument(
        "--output_model_dir",
        type=str,
        default=SECOND_REWARD_MODEL_OUTPUT_DIR,
        help="Directory to save trained model",
    )
    args = parser.parse_args()
    train_reward_model(output_model_dir=args.output_model_dir)
