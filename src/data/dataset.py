from datasets import load_dataset

from src.utils.config import DATASET_CONFIG, DATASET_NAME


def load_restaurant_dataset():
    """Load the same SemEval ABSA restaurant dataset used in all notebooks."""
    return load_dataset(DATASET_NAME, DATASET_CONFIG)
