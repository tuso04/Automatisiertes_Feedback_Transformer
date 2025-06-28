"""
chapter_coherence_eval.py

Aufgabe: Bewertet Gesamtkohärenz einzelner Textdokumente mittels SkipFlow-Modell für Kapitel-Level-Analyse

- Lädt SkipFlow-Kohärenz-Modell und GBERT-Tokenizer für Textbewertung
- Preprocessiert Texte durch Unicode-Normalisierung und Whitespace-Bereinigung
- Tokenisiert komplette Textdokumente mit Padding auf MAX_LEN
- Berechnet Gesamtkohärenz-Score via Sigmoid-Aktivierung des Modells
- Sammelt Kohärenz-Scores für alle Eingabetexte aus Verzeichnis
- Exportiert Ergebnisse mit Zusammenfassungsstatistiken in CSV-Format

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""


import torch
import unicodedata
import re
import csv
from pathlib import Path
from transformers import AutoTokenizer
from coherence_model.model.edit_skipflow import SkipFlowCoherenceOnly


def preprocess_text(text: str) -> str:
    """Bereinigt und normalisiert Text."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class CoherenceAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer und Modell laden
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
        self.model = SkipFlowCoherenceOnly()

        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_coherence(self, text: str, max_len: int = 500) -> float:
        """Berechnet Kohärenz-Score für einen Text."""
        clean_text = preprocess_text(text)

        # Text tokenisieren
        encoding = self.tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        # Kohärenz-Score berechnen
        with torch.no_grad():
            logit = self.model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
                return_matrix=True
            )
            coherence_score = torch.sigmoid(logit).item()

        return coherence_score


def load_texts_from_directory(directory: str) -> dict:
    """Lädt alle .txt Dateien aus einem Verzeichnis."""
    texts = {}
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts[file_path.stem] = f.read()
    return texts


def main():
    # Konfiguration
    MODEL_PATH = "/home/pthn17/bachelor_tim/coherence_model/model/checkpoints/a_model_epoch30.pt"
    INPUT_DIR = "input_texts"
    OUTPUT_CSV = "chapter_coherence_results.csv"

    # Analyzer initialisieren
    analyzer = CoherenceAnalyzer(MODEL_PATH)

    # Texte laden
    print("Lade Texte...")
    texts = load_texts_from_directory(INPUT_DIR)
    print(f"Gefunden: {len(texts)} Texte")

    if not texts:
        print(f"Keine .txt Dateien in {INPUT_DIR} gefunden!")
        return

    # Texte analysieren
    print("Analysiere Texte...")
    results = []

    for filename, text in texts.items():
        coherence_score = analyzer.predict_coherence(text)

        results.append({
            "filename": filename,
            "coherence_score": coherence_score
        })

        print(f"  {filename}: {coherence_score:.4f}")

    # Ergebnisse in CSV speichern
    print(f"\nSpeichere Ergebnisse in {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'coherence_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'filename': result['filename'],
                'coherence_score': f"{result['coherence_score']:.4f}"
            })

    # Zusammenfassung
    print("\n=== ZUSAMMENFASSUNG ===")
    scores = [r["coherence_score"] for r in results]
    avg_score = sum(scores) / len(scores)
    print(f"Durchschnittliche Kohärenz: {avg_score:.4f}")
    print(f"Höchste Kohärenz: {max(scores):.4f}")
    print(f"Niedrigste Kohärenz: {min(scores):.4f}")


if __name__ == "__main__":
    main()