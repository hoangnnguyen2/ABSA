import argparse

from src.data.dataset import load_restaurant_dataset
from src.evaluation.io import save_metrics
from src.evaluation.metrics import normalize_pair, pair_level_metrics


def gold_pairs(sample):
    aspects = sample["aspects"]
    pairs = set()
    for term, polarity in zip(aspects.get("term", []), aspects.get("polarity", [])):
        if str(term).strip() and str(polarity).strip():
            pairs.add(normalize_pair(term, polarity))
    return pairs


def evaluate_pipeline(model_name):
    if model_name == "lstm":
        from src.models.lstm_pipeline import LSTMABSAPipeline
        pipeline = LSTMABSAPipeline()
    elif model_name == "transformer":
        from src.models.transformer_pipeline import TransformerABSAPipeline
        pipeline = TransformerABSAPipeline()
    elif model_name == "generative":
        from src.models.generative_pipeline import GenerativeABSAPipeline
        pipeline = GenerativeABSAPipeline()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # IMPORTANT: original validation split is used only here as the held-out test set.
    test_dataset = load_restaurant_dataset()["validation"]
    gold_sets, pred_sets = [], []

    for sample in test_dataset:
        text = sample["text"]
        gold_sets.append(gold_pairs(sample))

        prediction = pipeline.predict(text)
        if model_name == "generative":
            items = prediction["aspects"]
        else:
            items = prediction

        pred_sets.append({
            normalize_pair(item["aspect"], item["polarity"])
            for item in items
            if str(item.get("aspect", "")).strip()
            and str(item.get("polarity", "")).strip()
        })

    metrics = pair_level_metrics(gold_sets, pred_sets)
    metrics["model"] = model_name
    metrics["test_split"] = "original validation"
    save_metrics(metrics, f"{model_name}_end_to_end_test.json")
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ABSA model on held-out test set")
    parser.add_argument("--model", choices=["lstm", "transformer", "generative"], required=True)
    args = parser.parse_args()
    evaluate_pipeline(args.model)
