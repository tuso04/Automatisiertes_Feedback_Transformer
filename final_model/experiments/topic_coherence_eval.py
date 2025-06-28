"""
coherence_analysis.py

Thematische Kohärenz-Analyse für Exposé-Korpus:
- Extrahiert Kapitel aus DOCX-Dateien mittels vereinfachter Heading-Erkennung
- Berechnet Sentence-BERT Embeddings für Titel und Kapitel-Inhalte
- Misst thematische Kohärenz via Kosinus-Ähnlichkeit zwischen Titel und Kapiteln
- Generiert Kohärenz-Scores pro Kapitel und Exposé
- Speichert Ergebnisse in coherence_analysis/thematic_coherence_scores.csv
- Erstellt Korpus-Statistiken (Mittelwert, Standardabweichung, Min/Max)
- Nutzt SBERT-Modelle für Satz- und Langtext-Embeddings

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from docx import Document
from sentence_transformers import SentenceTransformer
from utils.utils import load_config, read_jsonl
import nltk

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Berechnet Kosinus-Ähnlichkeit zwischen zwei Tensoren."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def extract_chapters_efficient(docx_path: Path, heading_map: dict) -> dict:
    """Extrahiert Kapitel aus DOCX-Datei - vereinfachte Version."""
    if not docx_path.exists():
        return {}

    doc = Document(str(docx_path))
    flow = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Vereinfachte Kapitel-Extraktion (angepasst an Ihre preprocessing-Logik)
    chapters = {}
    current_chapter = "Introduction"  # Default
    current_text = []

    for paragraph in flow:
        # Prüfe ob Paragraph eine Überschrift ist
        is_heading = any(term.lower() in paragraph.lower() for term in heading_map.keys())

        if is_heading and len(current_text) > 0:
            chapters[current_chapter] = " ".join(current_text)
            current_chapter = paragraph
            current_text = []
        else:
            current_text.append(paragraph)

    # Letztes Kapitel hinzufügen
    if current_text:
        chapters[current_chapter] = " ".join(current_text)

    return chapters


def main():
    # Konfiguration laden
    config = load_config("config.json")

    raw_dir = Path(config["paths"]["raw_exposes_dir"])
    jsonl_path = config["paths"]["preprocessed_jsonl"]
    output_dir = Path("coherence_analysis")
    output_dir.mkdir(exist_ok=True)

    # Modelle laden
    sbert_sent = SentenceTransformer(config["models"]["sbert_sentence"])
    sbert_long = SentenceTransformer(config["models"]["sbert_long"])

    # Heading-Map laden
    with open("../data/heading_terms.json", encoding="utf-8") as f:
        heading_map = json.load(f)

    # Datenstruktur für Ergebnisse
    results = []
    all_scores = []

    # Exposé-Daten sammeln
    expose_data = {}
    for entry in read_jsonl(jsonl_path):
        eid = entry["expose_id"]
        expose_data.setdefault(eid, {"title": eid.replace("_", " "), "chapters": set()})

    # Verarbeitung pro Exposé
    for eid in tqdm(expose_data.keys(), desc="Berechne Kohärenz-Scores"):
        docx_path = raw_dir / f"{eid}.docx"

        if not docx_path.exists():
            print(f"Warnung: {docx_path} nicht gefunden, überspringe {eid}")
            continue

        # Titel-Embedding
        title = expose_data[eid]["title"]
        title_embedding = sbert_sent.encode(title, convert_to_tensor=True)

        # Kapitel extrahieren und verarbeiten
        chapters = extract_chapters_efficient(docx_path, heading_map)

        expose_scores = []

        for chapter_name, chapter_text in chapters.items():
            if not chapter_text.strip():
                continue

            # Kapitel-Embedding
            chapter_embedding = sbert_long.encode(chapter_text, convert_to_tensor=True)

            # Kosinus-Ähnlichkeit berechnen
            coherence_score = cosine_similarity(chapter_embedding, title_embedding)

            # Ergebnisse sammeln
            results.append({
                'expose_id': eid,
                'chapter_name': chapter_name,
                'coherence_score': coherence_score
            })

            expose_scores.append(coherence_score)
            all_scores.append(coherence_score)

        print(f"Verarbeitet: {eid} ({len(expose_scores)} Kapitel)")

    # Ergebnisse als DataFrame
    df = pd.DataFrame(results)

    # CSV speichern
    csv_path = output_dir / "thematic_coherence_scores.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"CSV gespeichert: {csv_path}")

    # Statistiken berechnen und speichern
    if all_scores:
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)

        stats_path = output_dir / "coherence_statistics.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Thematische Kohärenz - Korpus Statistiken\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Anzahl Kapitel: {len(all_scores)}\n")
            f.write(f"Anzahl Exposés: {len(set(df['expose_id']))}\n")
            f.write(f"Durchschnittlicher Kohärenz-Score: {mean_score:.4f}\n")
            f.write(f"Standardabweichung: {std_score:.4f}\n")
            f.write(f"Minimum: {min(all_scores):.4f}\n")
            f.write(f"Maximum: {max(all_scores):.4f}\n")

        print(f"Statistiken gespeichert: {stats_path}")
        print(f"Durchschnitt: {mean_score:.4f}, Std: {std_score:.4f}")
    else:
        print("Keine Scores berechnet - prüfen Sie die Eingabedaten")


if __name__ == "__main__":
    main()