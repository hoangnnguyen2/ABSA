import numpy as np
from sklearn.metrics import accuracy_score as sklearn_accuracy
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import (
    accuracy_score as seqeval_accuracy,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)


def aspect_extraction_metrics(true_labels, true_predictions):
    """Entity-level ATE metrics plus token accuracy for BIO predictions."""
    return {
        "precision": float(seqeval_precision(true_labels, true_predictions)),
        "recall": float(seqeval_recall(true_labels, true_predictions)),
        "f1": float(seqeval_f1(true_labels, true_predictions)),
        "accuracy": float(seqeval_accuracy(true_labels, true_predictions)),
    }


def sentiment_classification_metrics(labels, predictions):
    """Macro metrics are used because ABSA sentiment classes are imbalanced."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(sklearn_accuracy(labels, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        # Alias kept so existing Trainer selection on eval_f1 still works.
        "f1": float(f1),
    }


def normalize_pair(aspect, polarity):
    return (" ".join(str(aspect).lower().strip().split()), str(polarity).lower().strip())


def parse_generated_pairs(text):
    text = str(text).strip()
    if not text or text.upper() == "NONE":
        return set()
    pairs = set()
    for part in text.split("|"):
        if ":" not in part:
            continue
        aspect, polarity = part.split(":", 1)
        aspect = aspect.strip()
        polarity = polarity.strip()
        if aspect and polarity:
            pairs.add(normalize_pair(aspect, polarity))
    return pairs


def pair_level_metrics(gold_sets, pred_sets):
    """Micro P/R/F1 over exact (aspect, polarity) pairs + sentence exact match."""
    tp = sum(len(gold & pred) for gold, pred in zip(gold_sets, pred_sets))
    total_pred = sum(len(pred) for pred in pred_sets)
    total_gold = sum(len(gold) for gold in gold_sets)

    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = (
        sum(gold == pred for gold, pred in zip(gold_sets, pred_sets)) / len(gold_sets)
        if gold_sets
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_match": float(exact_match),
        "true_positives": int(tp),
        "total_predicted_pairs": int(total_pred),
        "total_gold_pairs": int(total_gold),
        "num_samples": int(len(gold_sets)),
    }
