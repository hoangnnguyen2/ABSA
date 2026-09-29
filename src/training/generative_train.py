import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data.dataset import load_restaurant_dataset
from src.data.splits import train_eval_test_split
from src.data.generative_preprocessing import format_input, tokenize_batch
from src.evaluation.io import save_metrics
from src.evaluation.metrics import pair_level_metrics, parse_generated_pairs
from src.utils.config import EVAL_SIZE, SPLIT_SEED, GENERATIVE_DIR, GENERATIVE_MODEL, GENERATIVE_TOKENIZER_MODEL


def build_generation_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # 1. Chuyển đổi thành Numpy Array an toàn
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        # 2. Xử lý PREDICTIONS: Nếu là Logits (3D), lấy index có xác suất cao nhất
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # 3. Ép kiểu về số nguyên chuẩn và dọn dẹp các giá trị âm (như -100) trên cả 2 biến
        predictions = predictions.astype(np.int64)
        labels = labels.astype(np.int64)

        # Thay thế tất cả giá trị nhỏ hơn 0 hoặc bằng -100 bằng pad_token_id
        predictions = np.where((predictions >= 0) & (predictions != -100), predictions, tokenizer.pad_token_id)
        labels = np.where((labels >= 0) & (labels != -100), labels, tokenizer.pad_token_id)

        # 4. Giải mã chuỗi (Giờ đây không còn lo lỗi số âm / lỗi tràn bộ nhớ)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 5. Đưa vào hàm tính toán metric ABSA của bạn
        pred_sets = [parse_generated_pairs(text) for text in pred_texts]
        gold_sets = [parse_generated_pairs(text) for text in gold_texts]
        return pair_level_metrics(gold_sets, pred_sets)

    return compute_metrics


def train():
    dataset = load_restaurant_dataset()
    format_dataset = dataset.map(
        format_input,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_TOKENIZER_MODEL)
    tokenized_datasets = format_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=format_dataset["train"].column_names,
    )

    split = train_eval_test_split(tokenized_datasets, eval_size=EVAL_SIZE, seed=SPLIT_SEED)

    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATIVE_MODEL)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(GENERATIVE_DIR),
        predict_with_generate=True,
        generation_max_length=64,
        generation_num_beams=2,
        fp16=True,
        num_train_epochs=5,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
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
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["eval"],
        data_collator=data_collator,
        compute_metrics=build_generation_metrics(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(GENERATIVE_DIR))
    tokenizer.save_pretrained(GENERATIVE_DIR)

    # The original dataset validation split is evaluated only after training.
    test_metrics = trainer.predict(split["test"], metric_key_prefix="test").metrics
    save_metrics(test_metrics, "generative_test.json")

    # Also run raw end-to-end evaluation using the same held-out validation split.
    from src.evaluation.evaluate import evaluate_pipeline
    evaluate_pipeline("generative")


if __name__ == "__main__":
    train()
