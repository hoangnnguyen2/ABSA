from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.utils.config import BERT_MODEL, LSTM_EXTRACTOR_DIR, LSTM_SENTIMENT_DIR
from src.models.lstm import AspectExtractor, SentinentLSTM
from src.data.lstm_preprocessing import ID2TAG, IDX2LABEL, TAG2ID, SENTIMENT_LABELS


class LSTMABSAPipeline:
    def __init__(
        self,
        extractor_dir=LSTM_EXTRACTOR_DIR,
        sentiment_dir=LSTM_SENTIMENT_DIR,
        device=None,
    ):
        extractor_dir = Path(extractor_dir)
        sentiment_dir = Path(sentiment_dir)
        extractor_state = extractor_dir / "model_state.pt"
        sentiment_state = sentiment_dir / "model_state.pt"
        if not extractor_state.exists() or not sentiment_state.exists():
            raise FileNotFoundError(
                "LSTM checkpoints not found. Train both stages first: "
                "python scripts/train_lstm.py --stage all"
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer_source = extractor_dir if (extractor_dir / "tokenizer_config.json").exists() else BERT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        self.extractor = AspectExtractor(
            vocab_size=self.tokenizer.vocab_size,
            embedding_size=128,
            hidden_dim=256,
            output_dim=3,
            dropout=0.2,
            tag2idx=TAG2ID,
        )
        self.extractor.load_state_dict(torch.load(extractor_state, map_location=self.device))
        self.extractor.to(self.device).eval()

        self.sentiment = SentinentLSTM(
            vocab_size=len(self.tokenizer),
            embedding_size=128,
            hidden_dim=256,
            output_dim=len(SENTIMENT_LABELS),
            dropout=0.2,
            num_heads=8,
        )
        self.sentiment.load_state_dict(torch.load(sentiment_state, map_location=self.device))
        self.sentiment.to(self.device).eval()

    @torch.no_grad()
    def extract_aspects(self, text):
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        emissions = self.extractor(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        paths = self.extractor.crf.decode(emissions, mask=attention_mask.bool())
        pred_ids = paths[0]

        aspects = []
        current_start = None
        current_end = None
        for pred_id, (start, end) in zip(pred_ids, offsets):
            tag = ID2TAG[int(pred_id)]
            if tag == "B-ASP":
                if current_start is not None:
                    aspects.append(text[current_start:current_end].strip())
                current_start, current_end = start, end
            elif tag == "I-ASP" and current_start is not None:
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
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for aspect in aspects:
            text_ids = self.tokenizer.encode(
                text, truncation=True, max_length=64, add_special_tokens=False
            )
            aspect_ids = self.tokenizer.encode(
                aspect, truncation=True, max_length=16, add_special_tokens=False
            )
            text_tensor = torch.tensor([text_ids], dtype=torch.long, device=self.device)
            aspect_tensor = torch.tensor([aspect_ids], dtype=torch.long, device=self.device)
            text_mask = text_tensor != pad_id
            logits = self.sentiment(
                text_ids=text_tensor,
                aspect_ids=aspect_tensor,
                text_mask=text_mask,
            )["logits"]
            pred_id = logits.argmax(dim=-1).item()
            results.append({"aspect": aspect, "polarity": IDX2LABEL[int(pred_id)]})
        return results
