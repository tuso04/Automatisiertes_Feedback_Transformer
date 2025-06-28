"""
local_coherence_model_eval.py

Aufgabe: Analysiert Textkohärenz durch Vergleich von Originaltexten mit synthetischen Mischtexten mittels SkipFlow-Modell

- Lädt SkipFlow-Kohärenz-Modell und GBERT-Tokenizer für Satzpaar-Bewertung
- Tokenisiert Texte in Sätze und filtert nach Mindestlänge
- Berechnet Kohärenz-Scores für aufeinanderfolgende Satzpaare mit Sigmoid-Aktivierung
- Erstellt Mischtexte durch zufällige Rekombination aller verfügbaren Sätze
- Identifiziert höchste und niedrigste Kohärenz-Satzpaare pro Text
- Exportiert Vergleichsstatistiken zwischen Original- und Mischtexten in CSV

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import torch
import unicodedata
import re
import os
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from coherence_model.model.edit_skipflow import SkipFlowCoherenceOnly

# NLTK Daten herunterladen falls nötig
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ==== 1. Text-Vorverarbeitung ====
def preprocess_text(text: str) -> str:
    """Bereinigt und normalisiert Text."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==== 2. Modell Setup ====
class CoherenceAnalyzer:
    def __init__(self, model_path: str, min_sentence_length: int = 30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_sentence_length = min_sentence_length

        # Tokenizer und Modell laden
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
        self.model = SkipFlowCoherenceOnly()

        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_coherence(self, text: str, max_len: int = 500) -> float:
        """Berechnet Kohärenz-Score für einen Text."""
        clean = preprocess_text(text)
        enc = self.tokenizer(
            clean,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        with torch.no_grad():
            logit = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
                return_matrix=True
            )
            prob = torch.sigmoid(logit).item()

        return prob

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenisiert Text in Sätze und filtert kurze Sätze."""
        sentences = sent_tokenize(text, language='german')
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]

    def analyze_sentence_pairs(self, sentences: List[str]) -> Dict:
        """Analysiert aufeinanderfolgende Satzpaare."""
        if len(sentences) < 2:
            return {"avg_score": 0.0, "scores": [], "pairs": []}

        scores = []
        pairs = []

        for i in range(len(sentences) - 1):
            pair_text = f"{sentences[i]} {sentences[i + 1]}"
            score = self.predict_coherence(pair_text)
            scores.append(score)
            pairs.append((sentences[i], sentences[i + 1]))

        avg_score = np.mean(scores) if scores else 0.0
        variance = np.var(scores) if scores else 0.0

        return {
            "avg_score": avg_score,
            "variance": variance,
            "scores": scores,
            "pairs": pairs
        }

    def get_extreme_pairs(self, analysis: Dict, top_k: int = 5) -> Tuple[List, List]:
        """Gibt die k höchsten und niedrigsten Satzpaare zurück."""
        if not analysis["scores"]:
            return [], []

        indexed_scores = list(enumerate(analysis["scores"]))
        indexed_scores.sort(key=lambda x: x[1])

        lowest = indexed_scores[:top_k]
        highest = indexed_scores[-top_k:][::-1]

        lowest_pairs = [(analysis["pairs"][i], score) for i, score in lowest]
        highest_pairs = [(analysis["pairs"][i], score) for i, score in highest]

        return highest_pairs, lowest_pairs


# ==== 3. Text-Verarbeitung ====
def load_texts_from_directory(directory: str) -> Dict[str, str]:
    """Lädt alle .txt Dateien aus einem Verzeichnis."""
    texts = {}
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts[file_path.stem] = f.read()
    return texts


def create_mixed_texts(all_sentences: List[Tuple[str, str]], num_texts: int, sentences_per_text: int = 20) -> Dict[
    str, str]:
    """Erstellt Mischtexte aus allen verfügbaren Sätzen."""
    mixed_texts = {}
    all_sent_list = [sent for sentences in all_sentences for sent in sentences[1]]

    for i in range(num_texts):
        if len(all_sent_list) < sentences_per_text:
            selected_sentences = all_sent_list
        else:
            selected_sentences = random.sample(all_sent_list, sentences_per_text)

        mixed_text = " ".join(selected_sentences)
        mixed_texts[f"mixed_text_{i + 1}"] = mixed_text

    return mixed_texts


# ==== 4. Hauptfunktion ====
def main():
    # Konfiguration
    MODEL_PATH = "/home/pthn17/bachelor_tim/coherence_model/model/checkpoints/a_model_epoch30.pt"
    INPUT_DIR = "input_texts"  # Verzeichnis mit .txt Dateien
    OUTPUT_CSV = "coherence_results_v2.csv"
    MIN_SENTENCE_LENGTH = 30
    TOP_K = 5

    # Analyzer initialisieren
    analyzer = CoherenceAnalyzer(MODEL_PATH, MIN_SENTENCE_LENGTH)

    # Texte laden
    print("Lade Texte...")
    texts = load_texts_from_directory(INPUT_DIR)
    print(f"Gefunden: {len(texts)} Texte")

    if not texts:
        print(f"Keine .txt Dateien in {INPUT_DIR} gefunden!")
        return

    # Originaltexte analysieren
    print("Analysiere Originaltexte...")
    results = []
    all_sentences = []

    for filename, text in texts.items():
        sentences = analyzer.tokenize_sentences(text)
        all_sentences.append((filename, sentences))

        analysis = analyzer.analyze_sentence_pairs(sentences)
        highest_pairs, lowest_pairs = analyzer.get_extreme_pairs(analysis, TOP_K)

        result = {
            "filename": filename,
            "text_type": "original",
            "avg_coherence": analysis["avg_score"],
            "variance": analysis["variance"],
            "num_sentence_pairs": len(analysis["scores"]),
            "highest_scores": highest_pairs,
            "lowest_scores": lowest_pairs
        }
        results.append(result)

        print(
            f"  {filename}: Ø {analysis['avg_score']:.3f} (Var: {analysis['variance']:.3f}, {len(analysis['scores'])} Satzpaare)")

    # Mischtexte erstellen und analysieren
    print("\nErstelle und analysiere Mischtexte...")
    avg_sentences_per_text = int(np.mean([len(sentences) for _, sentences in all_sentences]))
    mixed_texts = create_mixed_texts(all_sentences, len(texts), avg_sentences_per_text)

    for filename, text in mixed_texts.items():
        sentences = analyzer.tokenize_sentences(text)
        analysis = analyzer.analyze_sentence_pairs(sentences)
        highest_pairs, lowest_pairs = analyzer.get_extreme_pairs(analysis, TOP_K)

        result = {
            "filename": filename,
            "text_type": "mixed",
            "avg_coherence": analysis["avg_score"],
            "variance": analysis["variance"],
            "num_sentence_pairs": len(analysis["scores"]),
            "highest_scores": highest_pairs,
            "lowest_scores": lowest_pairs
        }
        results.append(result)

        print(
            f"  {filename}: Ø {analysis['avg_score']:.3f} (Var: {analysis['variance']:.3f}, {len(analysis['scores'])} Satzpaare)")

    # Ergebnisse in CSV speichern
    print(f"\nSpeichere Ergebnisse in {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'text_type', 'avg_coherence', 'variance', 'num_sentence_pairs',
                      'top_5_highest', 'top_5_lowest']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Formatiere extreme Paare für CSV
            highest_str = "; ".join([f"({pair[0][:50]}... | {pair[1][:50]}...) = {score:.3f}"
                                     for (pair, score) in result["highest_scores"]])
            lowest_str = "; ".join([f"({pair[0][:50]}... | {pair[1][:50]}...) = {score:.3f}"
                                    for (pair, score) in result["lowest_scores"]])

            writer.writerow({
                'filename': result['filename'],
                'text_type': result['text_type'],
                'avg_coherence': f"{result['avg_coherence']:.4f}",
                'variance': f"{result['variance']:.4f}",
                'num_sentence_pairs': result['num_sentence_pairs'],
                'top_5_highest': highest_str,
                'top_5_lowest': lowest_str
            })

    # Zusammenfassung
    print("\n=== ZUSAMMENFASSUNG ===")
    original_scores = [r["avg_coherence"] for r in results if r["text_type"] == "original"]
    mixed_scores = [r["avg_coherence"] for r in results if r["text_type"] == "mixed"]

    print(f"Originaltexte - Ø Kohärenz: {np.mean(original_scores):.3f} (±{np.std(original_scores):.3f})")
    print(f"Mischtexte    - Ø Kohärenz: {np.mean(mixed_scores):.3f} (±{np.std(mixed_scores):.3f})")
    print(f"Differenz: {np.mean(original_scores) - np.mean(mixed_scores):.3f}")


if __name__ == "__main__":
    main()