"""
preprocessing.py

Extrahiert aus .docx-Exposés:
- Kapitelabschnitte mittels extract_chapters()
- Satz-Splitting mit SpaCy (inkl. unkommentierter Sätze)
- Word-Kommentare (python-docx) kapitelweise sammeln
- Deduplizierung der Kommentare (SBERT-Kosinus) pro Kapitel
- Ausgabe:
  1. JSONL mit (expose_id, kapitel, sentence_text, comment_text) nur für kommentierte Sätze
  2. Pro Exposé eine Gold-Bullet-Feedback .txt (Kapitelüberschrift + Bullets)

Teile des Codes wurden mit ChatGPT o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import os
import json
import logging
import argparse
import re
import zipfile
from pathlib import Path
import pypandoc
from lxml import etree
from docx.oxml.ns import qn
from sentence_transformers import SentenceTransformer
from utils.utils import load_config, setup_logging, write_jsonl
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from typing import Dict, List, Optional, Tuple

def is_bibliography_chapter(title: str, bibliography_terms: List[str]) -> bool:
    """
    Prüft, ob ein Kapiteltitel ein Literaturverzeichnis kennzeichnet.
    """
    title_lower = title.lower()
    return any(term.lower() in title_lower for term in bibliography_terms)

def extract_comments_with_context(docx_path: Path) -> List[Tuple[str, Dict[str, Optional[str]]]]:
    """
    Extrahiert Kommentare aus einer DOCX-Datei mittels XML-Parsing und behält den Kontext bei.
    Gibt eine Liste von Tupeln zurück: (kommentierter_text, kommentar_daten).
    """
    comments_with_context = []

    try:
        with zipfile.ZipFile(docx_path) as z:
            comments_xml = z.read('word/comments.xml')
            doc_xml = z.read('word/document.xml')

        # Parse Kommentare
        comments_tree = etree.XML(comments_xml)
        doc_tree = etree.XML(doc_xml)

        # Namespace-Mapping
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

        # Kommentare und ihre IDs extrahieren
        for comment in comments_tree.xpath('//w:comment', namespaces=ns):
            comment_id = comment.get(qn('w:id'))
            comment_data = {
                'author': comment.get(qn('w:author')),
                'date': comment.get(qn('w:date')),
                'text': ''.join(comment.itertext()).strip()
            }

            # 1) Finde commentRangeStart
            start_path = f"//w:commentRangeStart[@w:id='{comment_id}']"
            # 2) Sammle alle <w:t> zwischen Start und End
            raw_text_nodes = doc_tree.xpath(
                f"{start_path}/following::w:t["
                f"   preceding::w:commentRangeStart[@w:id='{comment_id}'] and "
                f"   following::w:commentRangeEnd[@w:id='{comment_id}']"
                f"]",
                namespaces=ns
            )

            # 3) Kommentierten Text zusammenbauen
            commented_text = ''.join([node.text for node in raw_text_nodes if node.text]).strip()

            if commented_text:  # Nur hinzufügen wenn Text gefunden wurde
                comments_with_context.append((commented_text, comment_data))

    except (KeyError, zipfile.BadZipFile, etree.XMLSyntaxError) as e:
        logging.warning(f"Fehler beim Extrahieren der Kommentare: {e}")
        return []

    return comments_with_context

def find_matching_sentence(commented_text: str, sentences: List[str]) -> Optional[int]:
    """
    Findet den am besten passenden Satz für den kommentierten Text.
    Gibt den Index des Satzes zurück oder None wenn kein passender Satz gefunden wurde.
    """
    # Direkte Übereinstimmung
    for idx, sentence in enumerate(sentences):
        if commented_text in sentence or sentence in commented_text:
            return idx

    # Fuzzy Matching falls nötig
    commented_words = set(commented_text.lower().split())
    best_match = None
    best_overlap = 0

    for idx, sentence in enumerate(sentences):
        sentence_words = set(sentence.lower().split())
        overlap = len(commented_words & sentence_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = idx

    # Nur zurückgeben wenn signifikante Überlappung
    if best_overlap >= min(int(len(commented_words)/3)*2, len(commented_words)):
        return best_match

    return None

def extract_chapters(
    flow_lines: list,
    terms_map: dict,
    min_heading_length: int = 50
) -> dict:
    """
    Extrahiert Kapitelabschnitte aus einer Liste von Fließtextzeilen basierend auf
    in terms_map definierten Schlagwörtern.
    term_to_key mappt jede Variante auf den Haupt-Key.
    """
    # Reverse Lookup: Stichwort → Key
    term_to_key = {}
    for key, variants in terms_map.items():
        for term in variants:
            term_to_key[term.lower()] = key

    gliederung_key = next(
        (k for k in terms_map
         if k.lower() == 'gliederung' or any(v.lower() == 'gliederung' for v in terms_map[k])),
        None
    )

    chapters = {}
    found_keys = set()  # Set zum Speichern bereits gefundener Keys
    i, n = 0, len(flow_lines)

    while i < n:

        line = flow_lines[i]
        low = line.lower()

        # Überschrift erkennen, aber nur für noch nicht gefundene Keys
        matched = next((t for t in term_to_key
                       if t in low
                       and len(line) < min_heading_length
                       and term_to_key[t] not in found_keys),
                      None)


        if not matched:
            i += 1
            continue

        key = term_to_key[matched]
        if key in chapters:
            i += 1
            found_keys.add(key)  # Key als gefunden markieren
            continue

        buffer = []
        j = i + 1

        if key == gliederung_key:
            num_regex = re.compile(r'^\s*\d+(?:\.\d+)*\s+')
            num_false = re.compile(r'^\s*\d+(?:\.\d+)*\.\s+')
            while j < n:
                nxt = flow_lines[j]
                if num_regex.match(nxt) or num_false.match(nxt):
                    buffer.append(nxt)
                    j += 1
                else:
                    low_n = nxt.lower()
                    if any(t in low_n and len(nxt) < min_heading_length and term_to_key[t] not in found_keys for t in term_to_key):
                        break
                    j += 1

            if buffer:
                chapters[key] = "\n".join(buffer).strip()
                i = j
            else:
                i += 1
        else:
            while j < n:
                nxt = flow_lines[j]
                low_n = nxt.lower()
                if any(t in low_n and len(nxt) < min_heading_length and term_to_key[t] not in found_keys for t in term_to_key):
                    break
                buffer.append(nxt)
                j += 1

            chapters[key] = "\n".join(buffer).strip()
            #print(f"key: {key} - {chapters[key]}")
            i = j

    return chapters

def deduplicate_comments(comments: list, sbert_model, threshold: float) -> list:
    """
    Entfernt Duplikate aus einer Liste von Kommentar-Strings.
    Zwei Kommentare gelten als identisch, wenn ihr SBERT-Kosinus > threshold.
    """
    if not comments:
        return []
    unique = []
    embeddings = sbert_model.encode(comments, convert_to_tensor=True)
    for idx, emb in enumerate(embeddings):
        is_duplicate = False
        for u_emb in sbert_model.encode(unique, convert_to_tensor=True):
            cos_sim = (emb @ u_emb) / (emb.norm() * u_emb.norm())
            if cos_sim.item() > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(comments[idx])
    return unique

def main(args):
    # ---- Setup ----
    config = load_config(args.config)
    setup_logging(config["logging"]["file"], config["logging"]["level"])
    logger = logging.getLogger(__name__)

    raw_dir = config["paths"]["raw_exposes_dir"]
    jsonl_path = config["paths"]["preprocessed_jsonl"]
    gold_dir = config["paths"]["gold_bullet_dir"]
    os.makedirs(Path(gold_dir), exist_ok=True)

    # Lade Kapitel-Überschriften-Mapping
    with open("../data/heading_terms.json", "r", encoding="utf-8") as f:
        heading_map = json.load(f)

    # Extrahiere Literaturverzeichnis-Begriffe aus heading_terms.json
    bibliography_terms = heading_map.get("Literaturverzeichnis", [])

    # SBERT-Modell für Dedup
    sbert_for_dedup = SentenceTransformer(config["models"]["sbert_sentence"])

    preprocessed_entries = []

    for filepath in Path(raw_dir).glob("*.docx"):
        expose_id = filepath.stem
        logger.info(f"Verarbeite Exposé {expose_id}")

        # 1) Lese alle Paragraphenzeilen
        # DOCX -> Markdown, behält alle Nummerierungen exakt bei
        md = pypandoc.convert_file(str(filepath), 'markdown')
        # Markdown-Zeilen splitten und leere sowie Kommentar-Footnotes ignorieren
        flow_lines = []
        skip_comment = False
        for line in md.splitlines():
            ln = line.strip()
            # Wechsel aus Skip-Modus, wenn eine Zeile mit '*' endet
            if skip_comment:
                if ln.endswith('*'):
                    skip_comment = False
                continue
            # Ignoriere leere Zeilen
            if not ln:
                continue
            # Ignoriere Pandoc-Kommentar-Footnotes wie [^1]: oder Inline-Verweise
            if re.match(r'^\[\^.*\]:(.*)$', ln):
                continue
            # Ignoriere Zeilen mit Fußnoten-Verweis alleine (z.B. "[^1]")
            if re.match(r'^\[\^[0-9]+\]$', ln):
                continue
            # Ignoriere HTML-Kommentare im Markdown
            if ln.startswith('<!--') or ln.endswith('-->'):
                continue
            # Beginne Skip-Modus bei Kommentarblock-Start
            if re.match(r'^\*\*\*\[Kommentar:\]\{\.underline\}\*\*.*$', ln):
                skip_comment = True
                continue
            flow_lines.append(ln)

        # 2) Extract Chapters (Text-Blöcke)
        chapters_text = extract_chapters(flow_lines, heading_map)

        # 3) Für jeden Kapiteltext: Satz-Splitting mit NLTK
        chapters = {}  # KapitelTitle → List[str] (Sätze)
        for title, chap_text in chapters_text.items():
            satz_list = [sent.strip() for sent in sent_tokenize(chap_text, language='german') if sent.strip()]
            chapters[title] = satz_list

        # 4) Kommentare mit Kontext extrahieren
        comments_with_context = extract_comments_with_context(filepath)

        # 5) Kommentare den Sätzen zuordnen
        sentence_comments = {}  # key: (kapitel_title, idx_in_chapter) → [Kommentare]
        for commented_text, comment_data in comments_with_context:
            for title, satz_list in chapters.items():
                idx = find_matching_sentence(commented_text, satz_list)
                if idx is not None:
                    key = (title, idx)
                    sentence_comments.setdefault(key, []).append(comment_data['text'])
                    break

        # 6) JSONL-Einträge für kommentierte Sätze
        comments_by_chapter = {title: [] for title in chapters.keys()
                               if not is_bibliography_chapter(title, bibliography_terms)}
        for (title, idx_in_chapter), comment_list in sentence_comments.items():

            # Überspringe Literaturverzeichnis-Kapitel
            if is_bibliography_chapter(title, bibliography_terms):
                continue

            unique = deduplicate_comments(comment_list, sbert_for_dedup, config["deduplication"]["sbert_threshold"])
            satz = chapters[title][idx_in_chapter]
            for c in unique:
                entry = {
                    "expose_id": expose_id,
                    "kapitel": title,
                    "sentence_text": satz,
                    "comment_text": c
                }
                preprocessed_entries.append(entry)
            comments_by_chapter[title].extend(unique)

        # 7) Gold-Bullet-Datei schreiben
        out_path = Path(gold_dir) / f"{expose_id}_gold_bullets.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for title, satz_list in chapters.items():
                # Überspringe Literaturverzeichnis-Kapitel
                if is_bibliography_chapter(title, bibliography_terms):
                    continue
                f.write(f"{title}:\n")
                bullets = comments_by_chapter.get(title, [])
                for b in bullets:
                    f.write(f"• {b.strip()}\n")
                f.write("\n")

    # 8) JSONL speichern
    write_jsonl(preprocessed_entries, jsonl_path)
    logger.info(f"Preprocessing abgeschlossen. JSONL unter {jsonl_path} und Gold-Bullets unter {gold_dir} gespeichert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing von Exposés (.docx) → JSONL + Gold-Bullet-Feedback")
    parser.add_argument("--config", type=str, default="config.json", help="Pfad zu config.json")
    args = parser.parse_args()
    main(args)