"""
all_coherence_eval.py

Aufgabe: Umfassende Kohärenzanalyse von Textsammlungen mit lokaler, globaler und thematischer Bewertung

- Laden von SkipFlow-Kohärenzmodell und SBERT-Embeddings für Textanalyse
- Einlesen von Originaltexten aus Verzeichnis und Themenzuordnungen aus JSONL
- Berechnung lokaler Kohärenz zwischen aufeinanderfolgenden Satzpaaren
- Bestimmung globaler Kohärenz durch SkipFlow-Modell für Gesamttext
- Ermittlung thematischer Kohärenz via Cosinus-Ähnlichkeit zwischen Text und Thema
- Statistische Analyse mit Kruskal-Wallis-Test und paarweisen Vergleichen

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import torch
import unicodedata
import re
import csv
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from coherence_model.model.edit_skipflow import SkipFlowCoherenceOnly
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import f_oneway, kruskal, mannwhitneyu
import warnings

warnings.filterwarnings('ignore')

# NLTK Daten herunterladen falls nötig
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def preprocess_text(text: str) -> str:
    """Bereinigt und normalisiert Text."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_topics(topics_file: str) -> Dict[str, str]:
    """Lädt Themenzuordnungen aus JSONL-Datei."""
    topics = {}
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # Prüfen ob es echtes JSONL (eine Zeile pro JSON) oder inline JSON ist
            if content.count('\n') > 0:
                # Echtes JSONL Format
                lines = content.split('\n')
            else:
                # Inline Format - JSON-Objekte durch } { getrennt
                # Zuerst alle }{ durch }\n{ ersetzen
                content = content.replace('} {', '}\n{')
                lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "id" in data and "topic" in data:
                        topics[data["id"]] = data["topic"]
                except json.JSONDecodeError as e:
                    print(f"Warnung: Fehler beim Parsen von Zeile {line_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Warnung: Datei {topics_file} nicht gefunden.")
    except Exception as e:
        print(f"Fehler beim Laden der Topics: {e}")

    return topics


def get_text_type(filename: str) -> str:
    """Bestimmt den Texttyp basierend auf dem Dateinamen."""
    if filename.startswith("mixed_"):
        return "mixed"
    elif filename.startswith("K"):
        return "K_text"
    elif filename.startswith("W"):
        return "W_text"
    else:
        return "other"


class CoherenceAnalyzer:
    def __init__(self, coherence_model_path: str, min_sentence_length: int = 30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_sentence_length = min_sentence_length

        # Kohärenz-Modell laden
        self.coherence_tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
        self.coherence_model = SkipFlowCoherenceOnly()

        state = torch.load(coherence_model_path, map_location=self.device, weights_only=True)
        self.coherence_model.load_state_dict(state)
        self.coherence_model.to(self.device)
        self.coherence_model.eval()

        # Embedding-Modell für thematische Kohärenz laden
        self.embedding_tokenizer = AutoTokenizer.from_pretrained("Nico97/SBERT-case-german-tng")
        self.embedding_model = AutoModel.from_pretrained("Nico97/SBERT-case-german-tng")
        self.embedding_model.to(self.device)
        self.embedding_model.eval()

    def get_embedding(self, text: str, max_len: int = 512) -> np.ndarray:
        """Erstellt Embedding für einen Text."""
        clean_text = preprocess_text(text)

        encoding = self.embedding_tokenizer(
            clean_text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.embedding_model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device)
            )
            # Mean pooling
            embeddings = outputs.last_hidden_state
            attention_mask = encoding["attention_mask"].to(self.device)
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask

        return embedding.cpu().numpy()

    def predict_coherence(self, text: str, max_len: int = 500) -> float:
        """Berechnet Kohärenz-Score für einen Text."""
        clean_text = preprocess_text(text)

        encoding = self.coherence_tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        with torch.no_grad():
            logit = self.coherence_model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
                return_matrix=True
            )
            coherence_score = torch.sigmoid(logit).item()

        return coherence_score

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenisiert Text in Sätze und filtert kurze Sätze."""
        sentences = sent_tokenize(text, language='german')
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]

    def analyze_local_coherence(self, sentences: List[str]) -> Dict:
        """Analysiert lokale Kohärenz zwischen aufeinanderfolgenden Satzpaaren."""
        if len(sentences) < 2:
            return {"avg_score": 0.0, "variance": 0.0, "num_pairs": 0}

        scores = []
        for i in range(len(sentences) - 1):
            pair_text = f"{sentences[i]} {sentences[i + 1]}"
            score = self.predict_coherence(pair_text)
            scores.append(score)

        return {
            "avg_score": np.mean(scores),
            "variance": np.var(scores),
            "num_pairs": len(scores)
        }

    def calculate_thematic_coherence(self, text: str, topic: str) -> float:
        """Berechnet thematische Kohärenz zwischen Text und Thema."""
        if not topic:
            return 0.0

        text_embedding = self.get_embedding(text)
        topic_embedding = self.get_embedding(topic)

        similarity = cosine_similarity(text_embedding, topic_embedding)[0, 0]
        return float(similarity)


def load_texts_from_directory(directory: str) -> Dict[str, str]:
    """Lädt alle .txt Dateien aus einem Verzeichnis."""
    texts = {}
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts[file_path.stem] = f.read()
    return texts


def create_mixed_texts(all_sentences: List[List[str]], num_texts: int,
                       sentences_per_text: int, available_topics: List[str]) -> Dict[str, Tuple[str, str]]:
    """Erstellt Mischtexte aus allen verfügbaren Sätzen mit zufälligen Themen."""
    mixed_texts = {}
    all_sent_list = [sent for sentences in all_sentences for sent in sentences]

    for i in range(num_texts):
        if len(all_sent_list) < sentences_per_text:
            selected_sentences = all_sent_list.copy()
            random.shuffle(selected_sentences)
        else:
            selected_sentences = random.sample(all_sent_list, sentences_per_text)

        mixed_text = " ".join(selected_sentences)
        random_topic = random.choice(available_topics)
        mixed_texts[f"mixed_text_{i + 1}"] = (mixed_text, random_topic)

    return mixed_texts


def perform_pairwise_tests(groups: Dict[str, List[float]], coherence_type: str) -> str:
    """Führt paarweise statistische Tests zwischen allen Texttypen durch."""
    group_names = list(groups.keys())
    results_text = f"\n  Paarweise Vergleiche für {coherence_type.upper()}:\n"

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            group1, group2 = groups[name1], groups[name2]

            if not group1 or not group2:
                continue

            # Mann-Whitney-U Test für paarweise Vergleiche
            try:
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

                # Effektgröße (r = Z / sqrt(N))
                n1, n2 = len(group1), len(group2)
                z_score = abs(statistic - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                effect_size = z_score / np.sqrt(n1 + n2)

                results_text += f"    {name1} vs {name2}: "
                results_text += f"U={statistic:.2f}, p={p_value:.4f}, r={effect_size:.3f}"

                # Signifikanz-Interpretation
                if p_value < 0.001:
                    sig_level = "***"
                elif p_value < 0.01:
                    sig_level = "**"
                elif p_value < 0.05:
                    sig_level = "*"
                else:
                    sig_level = "n.s."

                # Effektgröße-Interpretation
                if effect_size >= 0.5:
                    effect_desc = "großer Effekt"
                elif effect_size >= 0.3:
                    effect_desc = "mittlerer Effekt"
                elif effect_size >= 0.1:
                    effect_desc = "kleiner Effekt"
                else:
                    effect_desc = "vernachlässigbarer Effekt"

                results_text += f" ({sig_level}, {effect_desc})\n"

            except Exception as e:
                results_text += f"    {name1} vs {name2}: Test fehlgeschlagen ({e})\n"

    return results_text


def perform_statistical_analysis(results: List[Dict]) -> str:
    """Führt statistische Analyse der Kohärenz-Scores nach Texttypen durch."""
    # Daten nach Texttypen gruppieren
    text_types = {}
    for result in results:
        text_type = result["text_type"]
        if text_type not in text_types:
            text_types[text_type] = {
                "local": [], "global": [], "thematic": []
            }

        text_types[text_type]["local"].append(result["local_coherence"])
        text_types[text_type]["global"].append(result["global_coherence"])
        text_types[text_type]["thematic"].append(result["thematic_coherence"])

    # Legende erstellen
    analysis_text = "=== LEGENDE ===\n"
    analysis_text += "K_text: Texte mit K-Prefix (Kategorie K)\n"
    analysis_text += "W_text: Texte mit W-Prefix (Kategorie W)\n"
    analysis_text += "other: Texte mit anderen Prefixen\n"
    analysis_text += "mixed: Künstlich erstellte Mischtexte\n\n"
    analysis_text += "Signifikanzniveaus: *** p<0.001, ** p<0.01, * p<0.05, n.s. nicht signifikant\n"
    analysis_text += "Effektgrößen: r≥0.5 groß, r≥0.3 mittel, r≥0.1 klein, r<0.1 vernachlässigbar\n"
    analysis_text += "M = Mittelwert, SD = Standardabweichung, N = Anzahl\n"
    analysis_text += "=" * 70 + "\n\n"

    analysis_text += "=== DESKRIPTIVE STATISTIKEN ===\n\n"

    # Deskriptive Statistiken
    for text_type, scores in text_types.items():
        analysis_text += f"{text_type.upper()}:\n"
        analysis_text += f"  Lokale Kohärenz:     M={np.mean(scores['local']):.4f}, SD={np.std(scores['local']):.4f}, N={len(scores['local'])}\n"
        analysis_text += f"  Globale Kohärenz:    M={np.mean(scores['global']):.4f}, SD={np.std(scores['global']):.4f}, N={len(scores['global'])}\n"
        analysis_text += f"  Thematische Kohärenz: M={np.mean(scores['thematic']):.4f}, SD={np.std(scores['thematic']):.4f}, N={len(scores['thematic'])}\n\n"

    # Inferenzstatistik
    analysis_text += "=" * 70 + "\n"
    analysis_text += "=== INFERENZSTATISTISCHE TESTS ===\n"

    # Teste für jede Kohärenz-Art
    coherence_types = ["local", "global", "thematic"]

    for coh_type in coherence_types:
        analysis_text += f"\n{coh_type.upper()} KOHÄRENZ:\n"
        analysis_text += "-" * 30 + "\n"

        # Daten für Test sammeln
        groups = {}
        for text_type, scores in text_types.items():
            if scores[coh_type]:  # Nur wenn Daten vorhanden
                groups[text_type] = scores[coh_type]

        if len(groups) >= 2:
            # Omnibus-Test (Kruskal-Wallis für nicht-parametrische Daten)
            try:
                group_values = list(groups.values())
                h_stat, p_value = kruskal(*group_values)
                analysis_text += f"Kruskal-Wallis Test: H={h_stat:.4f}, p={p_value:.4f}\n"

                # Interpretation des Omnibus-Tests
                if p_value < 0.001:
                    analysis_text += "→ Hochsignifikanter Gesamtunterschied (p < 0.001)\n"
                elif p_value < 0.01:
                    analysis_text += "→ Sehr signifikanter Gesamtunterschied (p < 0.01)\n"
                elif p_value < 0.05:
                    analysis_text += "→ Signifikanter Gesamtunterschied (p < 0.05)\n"
                else:
                    analysis_text += "→ Kein signifikanter Gesamtunterschied (p ≥ 0.05)\n"

                # Paarweise Tests nur durchführen, wenn Omnibus-Test signifikant
                if p_value < 0.05:
                    analysis_text += perform_pairwise_tests(groups, coh_type)
                else:
                    analysis_text += "  (Keine paarweisen Tests aufgrund nicht-signifikantem Omnibus-Test)\n"

            except Exception as e:
                analysis_text += f"Omnibus-Test fehlgeschlagen: {e}\n"
        else:
            analysis_text += "Nicht genügend Gruppen für statistischen Test\n"

    return analysis_text


def main():
    # Konfiguration
    MODEL_PATH = "/home/pthn17/bachelor_tim/coherence_model/model/checkpoints/a_model_epoch30.pt"
    INPUT_DIR = "input_texts"
    TOPICS_FILE = "input_topics.jsonl"
    OUTPUT_CSV = "all_coherence_results_v2.csv"
    STATS_OUTPUT = "statistical_analysis_v2.txt"
    MIN_SENTENCE_LENGTH = 30

    # Analyzer initialisieren
    print("Initialisiere Modelle...")
    analyzer = CoherenceAnalyzer(MODEL_PATH, MIN_SENTENCE_LENGTH)

    # Texte und Themen laden
    print("Lade Texte und Themen...")
    texts = load_texts_from_directory(INPUT_DIR)
    topics = load_topics(TOPICS_FILE)
    print(f"Gefunden: {len(texts)} Texte, {len(topics)} Themen")

    if not texts:
        print(f"Keine .txt Dateien in {INPUT_DIR} gefunden!")
        return

    # Verfügbare Themen für Mischtexte sammeln
    available_topics = list(topics.values()) if topics else ["Allgemeines Thema"]

    # Originaltexte analysieren
    print("Analysiere Originaltexte...")
    results = []
    all_sentences = []

    for filename, text in texts.items():
        sentences = analyzer.tokenize_sentences(text)
        all_sentences.append(sentences)

        # Lokale und globale Kohärenz
        local_analysis = analyzer.analyze_local_coherence(sentences)
        global_coherence = analyzer.predict_coherence(text)

        # Thematische Kohärenz
        file_id = filename[:3]  # Erste drei Stellen
        topic = topics.get(file_id, "")
        thematic_coherence = analyzer.calculate_thematic_coherence(text, topic) if topic else 0.0

        text_type = get_text_type(filename)

        result = {
            "filename": filename,
            "text_type": text_type,
            "local_coherence": local_analysis["avg_score"],
            "local_variance": local_analysis["variance"],
            "global_coherence": global_coherence,
            "thematic_coherence": thematic_coherence,
            "topic": topic,
            "num_sentence_pairs": local_analysis["num_pairs"]
        }
        results.append(result)

        print(f"  {filename} ({text_type}): Lokal={local_analysis['avg_score']:.3f}, "
              f"Global={global_coherence:.3f}, Thematisch={thematic_coherence:.3f}")

    # Mischtexte erstellen und analysieren
    print("\nErstelle und analysiere Mischtexte...")
    avg_sentences_per_text = int(np.mean([len(sentences) for sentences in all_sentences]))
    mixed_texts = create_mixed_texts(all_sentences, len(texts), avg_sentences_per_text, available_topics)

    for filename, (text, assigned_topic) in mixed_texts.items():
        sentences = analyzer.tokenize_sentences(text)

        local_analysis = analyzer.analyze_local_coherence(sentences)
        global_coherence = analyzer.predict_coherence(text)
        thematic_coherence = analyzer.calculate_thematic_coherence(text, assigned_topic)

        result = {
            "filename": filename,
            "text_type": "mixed",
            "local_coherence": local_analysis["avg_score"],
            "local_variance": local_analysis["variance"],
            "global_coherence": global_coherence,
            "thematic_coherence": thematic_coherence,
            "topic": assigned_topic,
            "num_sentence_pairs": local_analysis["num_pairs"]
        }
        results.append(result)

        print(f"  {filename}: Lokal={local_analysis['avg_score']:.3f}, "
              f"Global={global_coherence:.3f}, Thematisch={thematic_coherence:.3f}")

    # Ergebnisse in CSV speichern
    print(f"\nSpeichere Ergebnisse in {OUTPUT_CSV}...")
    fieldnames = ['filename', 'text_type', 'local_coherence', 'local_variance',
                  'global_coherence', 'thematic_coherence', 'topic', 'num_sentence_pairs']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in result.items()}
                          for result in results])

    # Statistische Analyse durchführen
    print("Führe statistische Analyse durch...")
    stats_text = perform_statistical_analysis(results)

    with open(STATS_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(stats_text)

    print(f"Statistische Analyse gespeichert in {STATS_OUTPUT}")

    # Zusammenfassung
    print("\n=== ZUSAMMENFASSUNG ===")
    by_type = {}
    for result in results:
        text_type = result["text_type"]
        if text_type not in by_type:
            by_type[text_type] = {"local": [], "global": [], "thematic": []}

        by_type[text_type]["local"].append(result["local_coherence"])
        by_type[text_type]["global"].append(result["global_coherence"])
        by_type[text_type]["thematic"].append(result["thematic_coherence"])

    for text_type, scores in by_type.items():
        print(f"\n{text_type.upper()}:")
        print(f"  Lokale Kohärenz: {np.mean(scores['local']):.3f} (±{np.std(scores['local']):.3f})")
        print(f"  Globale Kohärenz: {np.mean(scores['global']):.3f} (±{np.std(scores['global']):.3f})")
        print(f"  Thematische Kohärenz: {np.mean(scores['thematic']):.3f} (±{np.std(scores['thematic']):.3f})")


if __name__ == "__main__":
    main()