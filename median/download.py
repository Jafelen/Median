import spacy.cli

try:
    from mlx_lm import load

    model_name = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
    load(model_name, lazy=False)
except ImportError:  # pragma: no cover - fallback for non-macOS platforms
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "distilgpt2"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)

spacy.cli.download("en_core_web_sm")
