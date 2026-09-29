from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.utils.config import TRANSFORMER_EXTRACTOR_DIR, TRANSFORMER_SENTIMENT_DIR


class TransformerABSAPipeline:
    def __init__(
        self,
        extractor_dir=TRANSFORMER_EXTRACTOR_DIR,
        sentiment_dir=TRANSFORMER_SENTIMENT_DIR,
        device=None,
    ):
        extractor_dir = Path(extractor_dir)
        sentiment_dir = Path(sentiment_dir)
        if not extractor_dir.exists() or not sentiment_dir.exists():
            raise FileNotFoundError(
                "Transformer checkpoints not found. Train both stages first: "
                "python scripts/train_transformer.py --stage all"
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.extractor_tokenizer = AutoTokenizer.from_pretrained(extractor_dir)
        self.extractor_model = AutoModelForTokenClassification.from_pretrained(extractor_dir).to(self.device)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_dir)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_dir).to(self.device)
        self.extractor_model.eval()
        self.sentiment_model.eval()

    @torch.no_grad()
    def extract_aspects(self, text):
        inputs = self.extractor_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        pred_ids = self.extractor_model(**model_inputs).logits.argmax(dim=-1)[0].cpu().tolist()

        aspects = []
        current_start = None
        current_end = None
        for pred_id, (start, end) in zip(pred_ids, offsets):
            if start == end:
                continue
            label = self.extractor_model.config.id2label[int(pred_id)]
            if label == "B-ASP":
                if current_start is not None:
                    aspects.append(text[current_start:current_end].strip())
                current_start, current_end = start, end
            elif label == "I-ASP" and current_start is not None:
                current_end = end
            else:
                if current_start is not None:
                    aspects.append(text[current_start:current_end].strip())
                    current_start = current_end = None
        if current_start is not None:
            aspects.append(text[current_start:current_end].strip())
        return [a for a in aspects if a]

    @torch.no_grad()
    def predict(self, text):
        aspects = self.extract_aspects(text)
        results = []
        for aspect in aspects:
            inputs = self.sentiment_tokenizer(
                text, aspect, return_tensors="pt", truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pred_id = self.sentiment_model(**inputs).logits.argmax(dim=-1).item()
            polarity = self.sentiment_model.config.id2label[int(pred_id)]
            results.append({"aspect": aspect, "polarity": polarity})
        return results
