"""Dataset split helpers used consistently by all ABSA models."""


def train_eval_test_split(dataset_dict, eval_size=0.2, seed=42):
    """Return train/eval/test with no test leakage.

    - train/eval are created only from the original ``train`` split.
    - test is the original dataset ``validation`` split.
    """
    train_eval = dataset_dict["train"].train_test_split(
        test_size=eval_size,
        seed=seed,
    )
    return {
        "train": train_eval["train"],
        "eval": train_eval["test"],
        "test": dataset_dict["validation"],
    }
