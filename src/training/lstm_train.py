import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import load_restaurant_dataset
from src.data.splits import train_eval_test_split
from src.evaluation.io import save_metrics
from src.evaluation.metrics import aspect_extraction_metrics, sentiment_classification_metrics
from src.utils.config import BERT_MODEL, EVAL_SIZE, SPLIT_SEED, LSTM_EXTRACTOR_DIR, LSTM_SENTIMENT_DIR
from src.models.lstm import AspectExtractor, SentinentLSTM
from src.data.lstm_preprocessing import (
    ABSADataCollator,
    ID2TAG,
    TAG2ID,
    SENTIMENT_LABELS,
    cleaned,
    prepare_for_sentiment,
    tokenize_and_align_labels,
)


class CRFTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.get("loss")
            if prediction_loss_only:
                return (loss, None, None)

            emissions = outputs.get("logits")
            labels = inputs.get("labels")
            if labels is not None:
                mask = (labels != -100).bool()
            elif inputs.get("attention_mask") is not None:
                mask = inputs.get("attention_mask").bool()
            else:
                mask = None

            crf_layer = model.module.crf if hasattr(model, "module") else model.crf
            best_paths = crf_layer.decode(emissions, mask=mask)
            ref_tensor = labels if labels is not None else inputs.get("input_ids")
            padded_predictions = torch.full_like(
                ref_tensor, fill_value=-100, dtype=torch.long
            )

            for i, path in enumerate(best_paths):
                if mask is not None:
                    true_indices = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
                    padded_predictions[i, true_indices] = torch.tensor(
                        path, device=ref_tensor.device, dtype=torch.long
                    )
                else:
                    padded_predictions[i, : len(path)] = torch.tensor(
                        path, device=ref_tensor.device, dtype=torch.long
                    )
        return (loss, padded_predictions, labels)


def extractor_metrics(eval_preds):
    predictions, labels = eval_preds
    true_predictions = [
        [ID2TAG[int(p)] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2TAG[int(l)] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return aspect_extraction_metrics(true_labels, true_predictions)


def sentiment_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return sentiment_classification_metrics(labels, predictions)


def train_extractor():
    dataset = load_restaurant_dataset()
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    tokenized_dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    split = train_eval_test_split(tokenized_dataset, eval_size=EVAL_SIZE, seed=SPLIT_SEED)

    model = AspectExtractor(
        vocab_size=tokenizer.vocab_size,
        embedding_size=128,
        hidden_dim=256,
        output_dim=3,
        dropout=0.2,
        tag2idx=TAG2ID,
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )
    args = TrainingArguments(
        output_dir=str(LSTM_EXTRACTOR_DIR),
        num_train_epochs=50,
        learning_rate=5e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        logging_strategy="epoch",
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )
    trainer = CRFTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["eval"],
        data_collator=data_collator,
        compute_metrics=extractor_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    LSTM_EXTRACTOR_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LSTM_EXTRACTOR_DIR / "model_state.pt")
    tokenizer.save_pretrained(LSTM_EXTRACTOR_DIR)
    test_metrics = trainer.predict(split["test"], metric_key_prefix="test").metrics
    save_metrics(test_metrics, "lstm_extractor_test.json")


def train_sentiment():
    dataset = load_restaurant_dataset()
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    cleaned_dataset = dataset.map(cleaned, batched=True)
    tokenized_dataset = cleaned_dataset.map(
        lambda batch: prepare_for_sentiment(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    split = train_eval_test_split(tokenized_dataset, eval_size=EVAL_SIZE, seed=SPLIT_SEED)

    model = SentinentLSTM(
        vocab_size=len(tokenizer),
        embedding_size=128,
        hidden_dim=256,
        output_dim=len(SENTIMENT_LABELS),
        dropout=0.2,
        num_heads=8,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    args = TrainingArguments(
        output_dir=str(LSTM_SENTIMENT_DIR),
        num_train_epochs=50,
        learning_rate=5e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        logging_strategy="epoch",
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["eval"],
        data_collator=ABSADataCollator(pad_token_id=pad_token_id),
        compute_metrics=sentiment_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    LSTM_SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LSTM_SENTIMENT_DIR / "model_state.pt")
    tokenizer.save_pretrained(LSTM_SENTIMENT_DIR)
    test_metrics = trainer.predict(split["test"], metric_key_prefix="test").metrics
    save_metrics(test_metrics, "lstm_sentiment_test.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM ABSA models")
    parser.add_argument("--stage", choices=["extractor", "sentiment", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ("extractor", "all"):
        train_extractor()
    if args.stage in ("sentiment", "all"):
        train_sentiment()
    if args.stage == "all":
        from src.evaluation.evaluate import evaluate_pipeline
        evaluate_pipeline("lstm")
