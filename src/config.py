import os

WANDB_LOGGING: bool = False
IS_ON_KAGGLE: bool = False  # need to login manually on False

# === Level 1 ===

# Reward model
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
REWARD_TRAIN_EPOCHS = 1
REWARD_LR = 5e-5
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
REWARD_MODEL_OUTPUT_DIR = os.path.join(os.getcwd(), "trained_reward_model")

# REINFORCE with baseline
REINFORCE_BATCH_SIZE = 4
REINFORCE_LR = 5e-5
REINFORCE_MAX_NEW_TOKENS = 512
REINFORCE_NUM_EPOCHS = 1
REINFORCE_OUTPUT_DIR = os.path.join(os.getcwd(), "reinforce_w_baseline_v2")

# === Level 2 ===
# TODO
