import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import load_restaurant_dataset
from src.data.splits import train_eval_test_split
from src.evaluation.io import save_metrics
from src.evaluation.metrics import aspect_extraction_metrics, sentiment_classification_metrics
from src.utils.config import EVAL_SIZE, SPLIT_SEED, TRANSFORMER_EXTRACTOR_DIR, TRANSFORMER_MODEL, TRANSFORMER_SENTIMENT_DIR
from src.data.transformer_preprocessing import (
    ID2LABEL,
    ID2TAG,
    LABEL2ID,
    TAG2ID,
    prepare_for_sentiment,
    tokenize_and_align_labels,
    tokenize_sentiment,
)


def label_from_predictions(predictions, labels):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [ID2TAG[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2TAG[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_predictions, true_labels


def extractor_metrics(eval_preds):
    predictions, labels = eval_preds
    true_predictions, true_labels = label_from_predictions(predictions, labels)
    return aspect_extraction_metrics(true_labels, true_predictions)


def sentiment_metrics(eval_preds):
    logits, labels = eval_preds
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return sentiment_classification_metrics(labels, predictions)


def train_extractor():
    dataset = load_restaurant_dataset()
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        TRANSFORMER_MODEL,
        num_labels=len(TAG2ID),
        id2label=ID2TAG,
        label2id=TAG2ID,
    )
    tokenized_dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    split = train_eval_test_split(tokenized_dataset, eval_size=EVAL_SIZE, seed=SPLIT_SEED)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )
    args = TrainingArguments(
        output_dir=str(TRANSFORMER_EXTRACTOR_DIR),
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        lr_scheduler_type="linear",
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_strategy="epoch",
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
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
        data_collator=data_collator,
        compute_metrics=extractor_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    trainer.save_model(str(TRANSFORMER_EXTRACTOR_DIR))
    tokenizer.save_pretrained(TRANSFORMER_EXTRACTOR_DIR)
    test_metrics = trainer.predict(split["test"], metric_key_prefix="test").metrics
    save_metrics(test_metrics, "transformer_extractor_test.json")


def train_sentiment():
    dataset = load_restaurant_dataset()
    sentiment_dataset = dataset.map(
        prepare_for_sentiment,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    tokenized_dataset = sentiment_dataset.map(
        lambda batch: tokenize_sentiment(batch, tokenizer),
        batched=True,
        remove_columns=sentiment_dataset["train"].column_names,
    )
    split = train_eval_test_split(tokenized_dataset, eval_size=EVAL_SIZE, seed=SPLIT_SEED)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    args = TrainingArguments(
        output_dir=str(TRANSFORMER_SENTIMENT_DIR),
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=30,
        lr_scheduler_type="linear",
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_strategy="epoch",
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
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
        data_collator=data_collator,
        compute_metrics=sentiment_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(str(TRANSFORMER_SENTIMENT_DIR))
    tokenizer.save_pretrained(TRANSFORMER_SENTIMENT_DIR)
    test_metrics = trainer.predict(split["test"], metric_key_prefix="test").metrics
    save_metrics(test_metrics, "transformer_sentiment_test.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Transformer ABSA models")
    parser.add_argument("--stage", choices=["extractor", "sentiment", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ("extractor", "all"):
        train_extractor()
    if args.stage in ("sentiment", "all"):
        train_sentiment()
    if args.stage == "all":
        from src.evaluation.evaluate import evaluate_pipeline
        evaluate_pipeline("transformer")
