def format_input(batch):
    batch_labels = []
    texts = batch["text"]
    aspects = batch["aspects"]

    for i in range(len(texts)):
        terms = aspects[i]["term"]
        polarities = aspects[i]["polarity"]
        pairs = []
        for term, polarity in zip(terms, polarities):
            if term != "" and polarity != "":
                pairs.append(f"{term.strip()}: {polarity.strip()}")
        target = "NONE" if not pairs else " | ".join(pairs)
        batch_labels.append(target)
    return {"input_text": texts, "labels": batch_labels}


def tokenize_batch(batch, tokenizer):
    tokenized_inputs = tokenizer(
        batch["input_text"], max_length=128, truncation=True
    )
    tokenized_labels = tokenizer(
        batch["labels"], max_length=64, truncation=True
    )
    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs
