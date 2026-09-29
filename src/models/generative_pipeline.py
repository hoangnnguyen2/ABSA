from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.config import GENERATIVE_DIR, GENERATIVE_MODEL, GENERATIVE_TOKENIZER_MODEL


class GenerativeABSAPipeline:
    def __init__(self, model_dir=None, device=None):
        local_dir = Path(model_dir or GENERATIVE_DIR)
        model_source = str(local_dir) if local_dir.exists() and any(local_dir.iterdir()) else GENERATIVE_MODEL
        tokenizer_source = str(local_dir) if local_dir.exists() and (local_dir / "tokenizer_config.json").exists() else GENERATIVE_TOKENIZER_MODEL
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_source).to(self.device)
        self.model.eval()

    @staticmethod
    def parse_output(output_text):
        if output_text.strip().upper() == "NONE":
            return []
        results = []
        for pair in output_text.split("|"):
            if ":" not in pair:
                continue
            aspect, polarity = pair.split(":", 1)
            aspect, polarity = aspect.strip(), polarity.strip()
            if aspect:
                results.append({"aspect": aspect, "polarity": polarity})
        return results

    @torch.no_grad()
    def predict(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=128, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, max_length=64, num_beams=2)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"raw_output": output_text, "aspects": self.parse_output(output_text)}
