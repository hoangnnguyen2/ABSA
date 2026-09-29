# ABSA Project

Project được tái cấu trúc từ 3 notebook gốc: LSTM, Transformer và Generative ABSA. Notebook gốc được giữ trong `notebooks/` để đối chiếu.

## Cấu trúc

```text
my_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   ├── api/
│   └── ui/
├── configs/
├── outputs/
├── saved_models/
├── requirements.txt
├── CHANGELOG.md
└── README.md
```


## Quy tắc chia Train / Eval / Test

Để tránh dùng tập test trong quá trình chọn mô hình, cả 3 pipeline dùng cùng một quy tắc:

```text
Dataset gốc
├── train
│   ├── 80% -> TRAIN
│   └── 20% -> EVAL (seed = 42)
└── validation -> TEST
```

- **TRAIN**: cập nhật trọng số model.
- **EVAL**: early stopping, chọn best checkpoint và theo dõi metric trong quá trình train.
- **TEST**: chính là split `validation` gốc của `jakartaresearch/semeval-absa`; chỉ dùng sau khi training để báo cáo kết quả cuối cùng.

Cấu hình nằm trong `configs/default.yaml`.

## Evaluation metrics

### Aspect Term Extraction (ATE)
- Precision
- Recall
- F1
- Accuracy

### Aspect Sentiment Classification (ASC)
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

### End-to-end ABSA
Một prediction chỉ được tính đúng khi **cả aspect term và polarity đều khớp**. Metric gồm:
- Precision
- Recall
- Micro F1
- Sentence exact-match

Kết quả test được ghi vào `outputs/evaluation/`.

Có thể đánh giá lại checkpoint đã train:

```python
!python -m src.evaluation.evaluate --model lstm
!python -m src.evaluation.evaluate --model transformer
!python -m src.evaluation.evaluate --model generative
```

## Chạy trên Google Colab

Sau khi clone repository:

```python
%cd /content/my_project
!pip install -q -r requirements.txt
```

### Train LSTM

```python
!python -m src.training.lstm_train --stage all
```

Có thể dùng `--stage extractor` hoặc `--stage sentiment` để train từng phần.

### Train Transformer

```python
!python -m src.training.transformer_train --stage all
```

### Train Generative

```python
!python -m src.training.generative_train
```

### Chạy UI Gradio

```python
!python -m src.ui.app
```

### Chạy API FastAPI

```python
!uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Dữ liệu

Dataset trong notebook gốc được tải từ Hugging Face (`jakartaresearch/semeval-absa`, config `restaurant`). Các thư mục `data/raw`, `data/processed`, `data/external` được chuẩn bị cho trường hợp cần lưu dataset cục bộ sau này.

## Model output

Checkpoint được lưu trong `saved_models/`:

- `saved_models/lstm_extractor/`
- `saved_models/lstm_sentiment/`
- `saved_models/transformer_extractor/`
- `saved_models/transformer_sentiment/`
- `saved_models/generative/`

Các đường dẫn và tên model được cấu hình ở `configs/default.yaml`.

## Lưu ý

- Không thay đổi notebook gốc.
- Logic model/hyperparameter được giữ từ source đã tách.
- Các thay đổi bắt buộc để module hóa và chạy project được ghi trong `CHANGELOG.md`.
