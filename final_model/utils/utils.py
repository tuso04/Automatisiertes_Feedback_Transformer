"""
utils.py

Allgemeine Hilfsfunktionen:
- load_config: Lädt config.json
- setup_logging: Konfiguriert Logging (Konsole + Datei)
- read_jsonl / write_jsonl: JSONL-Datei-Lese- bzw. Schreibfunktionen

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import json
import logging

def load_config(config_path: str = "config.json"):
    """Lädt die Konfigurationsdatei (JSON) und gibt ein Dictionary zurück."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def setup_logging(log_file: str, level: str = "INFO"):
    """
    Konfiguriert das Python-Logging.
    Logs gehen gleichzeitig in die Konsole und in die angegebene Datei.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )

def read_jsonl(filepath: str):
    """Liest eine JSONL-Datei und yieldet Zeile für Zeile als Python-Dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(data_iterable, filepath: str):
    """
    Schreibt ein Iterable von Python-Dicts als JSONL.
    Überschreibt ggf. vorhandene Datei.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data_iterable:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
