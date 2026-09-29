ASPECT_LABELS = ["O", "B-ASP", "I-ASP"]
TAG2ID = {label: i for i, label in enumerate(ASPECT_LABELS)}
ID2TAG = {i: label for i, label in enumerate(ASPECT_LABELS)}

SENTIMENT_LABELS = ["conflict", "negative", "neutral", "positive"]
ID2LABEL = {i: label for i, label in enumerate(SENTIMENT_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}


def tokenize_and_align_labels(batch, tokenizer):
    tokenized_inputs = tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
        padding=False,
        return_offsets_mapping=True,
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


def prepare_for_sentiment(batch):
    batch_text, batch_term, batch_sentiment = [], [], []
    texts = batch["text"]
    aspects = batch["aspects"]

    for i in range(len(texts)):
        text = texts[i]
        aspect = aspects[i]
        polarities = aspect["polarity"]
        terms = aspect["term"]

        # Source notebook used: `if terms == [""] or term = "":`, which is invalid Python.
        # Equivalent intended check is implemented here without changing the loop logic.
        if terms == [""]:
            continue

        for term, polarity in zip(terms, polarities):
            if term == "":
                continue
            batch_text.append(text)
            batch_term.append(term)
            batch_sentiment.append(polarity)

    return {"text": batch_text, "term": batch_term, "polarity": batch_sentiment}


def tokenize_sentiment(batch, tokenizer):
    tokenized_inputs = tokenizer(
        batch["text"], batch["term"], truncation=True, padding=False
    )
    tokenized_inputs["labels"] = [LABEL2ID[label] for label in batch["polarity"]]
    return tokenized_inputs
