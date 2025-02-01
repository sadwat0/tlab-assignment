from transformers import AutoTokenizer
from datasets import load_dataset
from config import MAX_LENGTH


def get_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_reward_dataset(example, tokenizer):
    "Samples for reward model"

    def extract_text(conversation):
        return "\n".join(
            [f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation]
        )

    prompt_chosen = extract_text(example["chosen"])
    prompt_rejected = extract_text(example["rejected"])

    kwargs = {
        "padding": "max_length",
        "truncation": True,
        "max_length": MAX_LENGTH,
        "return_tensors": "pt",
    }
    tokens_chosen = tokenizer.encode_plus(prompt_chosen, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_rejected, **kwargs)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0],
        "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0],
        "attention_mask_rejected": tokens_rejected["attention_mask"][0],
    }


def format_reinforce_dataset(example, tokenizer):
    "Sample for REINFORCE w/ baseline."

    def extract_prompt(conversation):
        return (
            "\n".join(
                [
                    f"{turn['role'].capitalize()}: {turn['content']}"
                    for turn in conversation[:-1]
                ]
            )
            + "\nAssistant: "
        )

    tokens = tokenizer(
        extract_prompt(example["chosen"]),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {k: v[0] for k, v in tokens.items()}


def load_and_split_dataset(
    test_size=0.2, train_subset_sizes=None, val_subset_size=None
):
    dataset = load_dataset("esfrankel17/HelpSteer2_binarized")["average_rating_split"]
    dataset = dataset.train_test_split(test_size=test_size, seed=42)

    if train_subset_sizes is not None:
        dataset["train"] = dataset["train"].select(range(train_subset_sizes))
    if val_subset_size is not None:
        dataset["test"] = dataset["test"].select(range(val_subset_size))

    return dataset
