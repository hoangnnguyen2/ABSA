from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = PROJECT_ROOT / CONFIG["project"]["data_dir"]
OUTPUT_DIR = PROJECT_ROOT / CONFIG["project"]["output_dir"]
SAVED_MODELS_DIR = PROJECT_ROOT / CONFIG["project"]["saved_models_dir"]

DATASET_NAME = CONFIG["dataset"]["name"]
DATASET_CONFIG = CONFIG["dataset"]["config"]
EVAL_SIZE = float(CONFIG["dataset"].get("eval_size", 0.2))
SPLIT_SEED = int(CONFIG["dataset"].get("split_seed", 42))
TEST_SPLIT = CONFIG["dataset"].get("test_split", "validation")

BERT_MODEL = CONFIG["models"]["bert"]
TRANSFORMER_MODEL = CONFIG["models"]["transformer"]
GENERATIVE_TOKENIZER_MODEL = CONFIG["models"]["generative_tokenizer"]
GENERATIVE_MODEL = CONFIG["models"]["generative"]

LSTM_EXTRACTOR_DIR = PROJECT_ROOT / CONFIG["paths"]["lstm_extractor"]
LSTM_SENTIMENT_DIR = PROJECT_ROOT / CONFIG["paths"]["lstm_sentiment"]
TRANSFORMER_EXTRACTOR_DIR = PROJECT_ROOT / CONFIG["paths"]["transformer_extractor"]
TRANSFORMER_SENTIMENT_DIR = PROJECT_ROOT / CONFIG["paths"]["transformer_sentiment"]
GENERATIVE_DIR = PROJECT_ROOT / CONFIG["paths"]["generative"]
