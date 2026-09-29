import re
import torch
from torch.nn.utils.rnn import pad_sequence

ASPECT_LABELS = ["O", "B-ASP", "I-ASP"]
TAG2ID = {label: i for i, label in enumerate(ASPECT_LABELS)}
ID2TAG = {i: label for i, label in enumerate(ASPECT_LABELS)}

SENTIMENT_LABELS = sorted(["negative", "neutral", "positive", "conflict"])
IDX2LABEL = {idx: label for idx, label in enumerate(SENTIMENT_LABELS)}
LABEL2IDX = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}


def cleaned(batch):
    text_list = []
    for text in batch["text"]:
        cleaned_text = re.sub(r"/s+", " ", text).strip()
        text_list.append(cleaned_text)
    return {"text": text_list}


def tokenize_and_align_labels(batch, tokenizer):
    tokenized_inputs = tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
        padding=False,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    all_labels = []
    for i in range(len(batch["text"])):
        raw_text = batch["text"][i]
        offsets = tokenized_inputs["offset_mapping"][i]
        aspect_dict = batch["aspects"][i]
        start_spans = aspect_dict.get("from", [])
        end_spans = aspect_dict.get("to", [])
        valid_spans = sorted(
            [(s, e) for s, e in zip(start_spans, end_spans) if s < e],
            key=lambda x: x[0],
        )

        label_ids = []
        current_active_span = None
        prev_token_end = 0
        seq_ids = tokenized_inputs.sequence_ids(batch_index=i)

        for idx, (off_start, off_end) in enumerate(offsets):
            if seq_ids[idx] is None or off_start == off_end:
                label_ids.append(-100)
                continue

            matched_span = None
            for asp_start, asp_end in valid_spans:
                if max(off_start, asp_start) < min(off_end, asp_end):
                    matched_span = (asp_start, asp_end)
                    break

            if matched_span is None:
                label_ids.append(TAG2ID["O"])
                current_active_span = None
            elif current_active_span != matched_span:
                label_ids.append(TAG2ID["B-ASP"])
                current_active_span = matched_span
            else:
                is_new_word = (off_start > prev_token_end) or (
                    off_start < len(raw_text) and raw_text[off_start].isspace()
                )
                if is_new_word:
                    label_ids.append(TAG2ID["I-ASP"])
                else:
                    label_ids.append(-100)
            prev_token_end = off_end

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs


def prepare_for_sentiment(batch, tokenizer):
    batch_texts, batch_terms, batch_polarities = [], [], []
    texts = batch.get("text")
    aspects = batch.get("aspects")

    for i in range(len(texts)):
        aspect = aspects[i]
        text_ids = tokenizer.encode(
            texts[i], truncation=True, max_length=64, add_special_tokens=False
        )
        terms = aspect.get("term", [])
        polarities = aspect.get("polarity", [])

        for term, polarity in zip(terms, polarities):
            if term == "" or polarity not in LABEL2IDX:
                continue
            term_ids = tokenizer.encode(
                term, truncation=True, max_length=16, add_special_tokens=False
            )
            batch_texts.append(text_ids)
            batch_terms.append(term_ids)
            batch_polarities.append(LABEL2IDX[polarity])

    return {
        "text_ids": batch_texts,
        "aspect_ids": batch_terms,
        "label": batch_polarities,
    }


class ABSADataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        text_ids = [torch.tensor(item["text_ids"], dtype=torch.long) for item in batch]
        aspect_ids = [torch.tensor(item["aspect_ids"], dtype=torch.long) for item in batch]
        text_padded = pad_sequence(text_ids, batch_first=True, padding_value=self.pad_token_id)
        aspect_padded = pad_sequence(aspect_ids, batch_first=True, padding_value=self.pad_token_id)
        text_mask = text_padded != self.pad_token_id

        batch_dict = {
            "text_ids": text_padded,
            "aspect_ids": aspect_padded,
            "text_mask": text_mask,
        }
        if "label" in batch[0]:
            batch_dict["labels"] = torch.tensor(
                [item["label"] for item in batch], dtype=torch.long
            )
        return batch_dict
