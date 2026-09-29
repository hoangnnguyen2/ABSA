import json
from functools import lru_cache

import gradio as gr

from src.models.generative_pipeline import GenerativeABSAPipeline
from src.models.lstm_pipeline import LSTMABSAPipeline
from src.models.transformer_pipeline import TransformerABSAPipeline


@lru_cache(maxsize=1)
def generative_pipeline():
    return GenerativeABSAPipeline()


@lru_cache(maxsize=1)
def transformer_pipeline():
    return TransformerABSAPipeline()


@lru_cache(maxsize=1)
def lstm_pipeline():
    return LSTMABSAPipeline()


def analyze(text, model_name):
    if not text or not text.strip():
        return {"error": "Please enter text."}
    try:
        if model_name == "Generative T5":
            return generative_pipeline().predict(text)
        if model_name == "Transformer XLM-R":
            return transformer_pipeline().predict(text)
        return lstm_pipeline().predict(text)
    except Exception as exc:
        return {"error": str(exc)}


def build_demo():
    with gr.Blocks(title="Aspect-Based Sentiment Analysis") as demo:
        gr.Markdown("# Aspect-Based Sentiment Analysis")
        gr.Markdown(
            "Choose one of the three model families extracted from the original notebooks. "
            "Transformer/LSTM require local checkpoints after training; Generative can fall back "
            "to the pretrained NUSTM restaurant model."
        )
        model = gr.Dropdown(
            ["Generative T5", "Transformer XLM-R", "LSTM + CRF"],
            value="Generative T5",
            label="Model",
        )
        text = gr.Textbox(
            label="Restaurant review",
            lines=5,
            placeholder="The food was excellent but the service was slow.",
        )
        run = gr.Button("Analyze")
        output = gr.JSON(label="Result")
        run.click(analyze, inputs=[text, model], outputs=output)
    return demo


if __name__ == "__main__":
    build_demo().launch(share=True)
