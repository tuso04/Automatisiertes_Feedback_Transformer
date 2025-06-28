"""
tokenizer_utils.py

Kapselt das Laden des deutschen GPT-2-Kleinmodells und seines Tokenizers für Inference und Adapter.

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer


def load_gpt2_tokenizer_and_model(model_name: str):
    """
    Lädt den GPT-2-Tokenizer und das GPT-2-Model (Language Modeling Head) vom HuggingFace-Name.
    Gibt (tokenizer, model) zurück. Model wird direkt auf GPU (.cuda()) gesetzt.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda().eval()
    return tokenizer, model
