"""
encoder.py

Berechnet für jedes Exposé:
- Satz-Embeddings (SBERT)
- Kapitel-Embeddings (Long-Text-SBERT)
- Kohärenz-Scores (samirmsallem/gbert-large-coherence_evaluation)
- Zero-Shot-Scores (svalabs/gbert-large-zeroshot-nli), Label aus Kapitelüberschrift
- Adäquanz-Scores (SBERT-Kosinus mit Titel der Arbeit)
- Unkommentierte und kommentierte Sätze fließen alle in die Repräsentation ein, ohne Kommentar-Embeddings
- Fügt alles in eine Matrix M ∈ ℝ^{408×770} plus Attention-Mask zusammen und speichert sie

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import os
import json
import torch
import logging
import argparse

from pathlib import Path
from tqdm import tqdm
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from utils.utils import load_config, setup_logging, read_jsonl
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Kosinus-Ähnlichkeit zweier Tensoren (D,)."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def main(args):
    # ---- Setup ----
    config = load_config(args.config)
    setup_logging(config["logging"]["file"], config["logging"]["level"])
    logger = logging.getLogger(__name__)

    # Pfade und Hyperparameter
    jsonl_path = config["paths"]["preprocessed_jsonl"]
    out_dir = config["paths"]["encoder_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    S_MAX = config["hyperparameters"]["max_sentences_per_chapter"]  # 50
    K_MAX = config["hyperparameters"]["num_chapters"]             # 8

    # Modelle laden
    sbert_sent = SentenceTransformer(config["models"]["sbert_sentence"])
    sbert_long = SentenceTransformer(config["models"]["sbert_long"])
    coherence_model = pipeline("text-classification", model=config["models"]["coherence"], device=0)
    zeroshot_model = pipeline("zero-shot-classification", model=config["models"]["zeroshot"], device=0)

    # Lese JSONL, gruppiere nach expose_id + kapitel
    expose_data = {}
    # Temporär abspeichern, um unkommentierte Sätze später hinzuzufügen
    comments_map = {}  # (eid, kapitel, satz_text) → List[comment_text]
    for entry in read_jsonl(jsonl_path):
        eid = entry["expose_id"]
        kapit = entry["kapitel"]
        satz = entry["sentence_text"]
        ctext = entry["comment_text"]
        comments_map.setdefault((eid, kapit, satz), []).append(ctext)
        expose_data.setdefault(eid, {"chapters": {}, "title": eid.replace("_", " ")})
        expose_data[eid]["chapters"].setdefault(kapit, set()).add(satz)

    # Wir benötigen alle Kapitel-Sätze, nicht nur kommentierte.
    # Also in jedem Original-Docx-Exposé die Kapitel-Absätze erneut laden, in Sätze splitten,
    # um unkommentierte Sätze zu erfassen.
    for eid in list(expose_data.keys()):
        # Pfad zur .docx-Datei
        docx_path = Path(config["paths"]["raw_exposes_dir"]) / f"{eid}.docx"
        if not docx_path.exists():
            logger.warning(f"Docx für {eid} nicht gefunden.")
            continue

        doc = Document(str(docx_path))
        flow_lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        with open("../data/heading_terms.json", "r", encoding="utf-8") as f:
            heading_map = json.load(f)
        # Kapitel-Textblöcke extrahieren
        from preprocessing import extract_chapters  # Funktion aus preprocessing.py
        chapters_text = extract_chapters(flow_lines, heading_map)

        # Für jedes Kapitel alle Sätze extrahieren
        for title, chap_text in chapters_text.items():
            sents = [sent.strip() for sent in sent_tokenize(chap_text, language='german') if sent.strip()]
            expose_data[eid]["chapters"].setdefault(title, set()).update(sents)

    # Nun enthält expose_data[eid]["chapters"][title] alle Sätze (kommentiert und unkommentiert)
    # und comments_map nur die tatsächlich kommentierten Sätze

    for eid, data in tqdm(expose_data.items(), desc="Exposés encodieren"):
        chapters = data["chapters"]
        title_embedding = sbert_sent.encode(data["title"], convert_to_tensor=True)

        # Matrix initialisieren
        row_count = K_MAX * (S_MAX + 1)
        print(f"[DEBUG] adapter_input_dim: {config['hyperparameters']['adapter_input_dim']}")
        M = torch.zeros((row_count, config["hyperparameters"]["adapter_input_dim"]))
        mask = torch.zeros(row_count, dtype=torch.long)

        row_idx = 0
        for k_i, (kapitel_title, satz_set) in enumerate(chapters.items()):
            if k_i >= K_MAX:
                break
            satz_list = list(satz_set)
            # 1) Kapitel-Embedding (Long-SBERT)
            chap_text = " ".join(satz_list)
            e_kap = sbert_long.encode(chap_text, convert_to_tensor=True)
            # 2) Kapitel-Kohärenz
            coh_kap = float(coherence_model(chap_text, truncation=True)[0]["score"])
            # 3) Kapitel-Adäquanz (Kosinus mit Titel-Embedding)
            a_kap = cosine(e_kap, title_embedding)
            # 4) Zero-Shot-Label ist Kapitel-Titel selbst
            label = kapitel_title

            # 5) Alle Sätze (auch unkommentierte) verarbeiten, max. S_MAX
            for j, satz in enumerate(satz_list):
                if j >= S_MAX:
                    break
                # Satz-Embedding
                e_satz = sbert_sent.encode(satz, convert_to_tensor=True)
                # Kohärenz zwischen vorherigem und aktuellem Satz
                prev_satz = satz_list[j-1] if j > 0 else ""
                coh_s = float(coherence_model(f"{prev_satz} {satz}", truncation=True)[0]["score"])
                # Zero-Shot-Score
                zs = zeroshot_model(satz, [f"Dieser Satz beschreibt {label}"])["scores"][0]
                # Kommentar existiert oder nicht?
                # Keine Kommentar-Embeddings ! (nur roher Text wird in Gold-Bullet genutzt)
                # Wir setzen comment-Teil auf 0
                # Konstruktion des 770-Vektors: [e_satz (768) | coh_s (1) | zs (1)]
                row = torch.cat([
                    e_satz,                  # 768
                    torch.tensor([coh_s]),   # 1
                    torch.tensor([zs])       # 1
                ], dim=0)  # → (770,)

                M[row_idx] = row
                mask[row_idx] = 1
                row_idx += 1

            # Kapitelzeile auf Position = k_i*(S_MAX+1) + S_MAX
            chap_row = k_i * (S_MAX + 1) + S_MAX
            row = torch.cat([
                e_kap,                       # 768
                torch.tensor([coh_kap]),     # 1
                torch.tensor([a_kap])        # 1
            ], dim=0)  # → (770,)
            M[chap_row] = row
            mask[chap_row] = 1
            # Rest bis 408 bleiben Null (Padding)

        # Speichern
        out_path = Path(out_dir) / f"{eid}_M.pt"
        torch.save({"M": M, "mask": mask, "title_embedding": title_embedding}, out_path)
        logger.info(f"Encoder-Output für Exposé {eid} gespeichert → {out_path}")

    logger.info("Encoder-Vorgang abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoder: SBERT, Kohärenz, Zero-Shot, Adäquanz → M-Matrix")
    parser.add_argument("--config", type=str, default="config.json", help="Pfad zu config.json")
    args = parser.parse_args()
    main(args)
