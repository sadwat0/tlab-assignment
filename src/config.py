import os

WANDB_LOGGING: bool = True
IS_ON_KAGGLE: bool = False  # need to login manually on False

# === Level 1 ===

# Reward model
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
REWARD_MODEL_OUTPUT_DIR = os.path.join(os.getcwd(), "trained_reward_model")
REWARD_TRAIN_EPOCHS = 1
REWARD_LR = 5e-5
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

# REINFORCE with baseline
REINFORCE_BATCH_SIZE = 4
REINFORCE_LR = 5e-5
MAX_NEW_TOKENS = 512

# === Level 2 ===
# TODO
