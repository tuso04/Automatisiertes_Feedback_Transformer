"""
encoder_variants.py

Modularer Encoder für Ablation-Experimente mit verschiedenen Komponenten-Kombinationen.
Unterstützt die 4 strategischen Ablation-Varianten aus dem Experimentkonzept.

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import json
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from coherence_model.individual_test_model import predict_coherence
from utils.utils import load_config, setup_logging, read_jsonl
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class EncoderComponent(Enum):
    """Enum für Encoder-Komponenten"""
    SENTENCE_LEVEL = "sentence_level"
    CHAPTER_LEVEL = "chapter_level"
    ADEQUACY = "adequacy"
    COHERENCE = "coherence"


@dataclass
class AblationConfig:
    """Konfiguration für Ablation-Varianten"""
    name: str
    description: str
    enabled_components: List[EncoderComponent]

    @classmethod
    def get_predefined_variants(cls) -> Dict[str, 'AblationConfig']:
        """Vordefinierte 4 strategische Ablation-Varianten"""
        return {
            "full_model": cls(
                name="full_model",
                description="Vollmodell: Alle Komponenten aktiv",
                enabled_components=[
                    EncoderComponent.SENTENCE_LEVEL,
                    EncoderComponent.CHAPTER_LEVEL,
                    EncoderComponent.ADEQUACY,
                    EncoderComponent.COHERENCE
                ]
            ),
            "no_structure": cls(
                name="no_structure",
                description="-Strukturebenen: Nur Adäquanz + Kohärenz",
                enabled_components=[
                    EncoderComponent.SENTENCE_LEVEL,
                    EncoderComponent.ADEQUACY,
                    EncoderComponent.COHERENCE
                ]
            ),
            "no_textfeatures": cls(
                name="no_textfeatures",
                description="-Textmerkmale: Nur Satz-/Kapitelebene",
                enabled_components=[
                    EncoderComponent.SENTENCE_LEVEL,
                    EncoderComponent.CHAPTER_LEVEL
                ]
            ),
            "minimal": cls(
                name="minimal",
                description="Minimal-Encoder: Nur Basis-Embeddings",
                enabled_components=[
                    EncoderComponent.SENTENCE_LEVEL
                ]
            )
        }

    def to_serializable_dict(self) -> dict:
        """Konvertiert die Konfiguration zu einem serialisierbaren Dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "enabled_components": [comp.value for comp in self.enabled_components]
        }


class ModularEncoder:
    """Modularer Encoder mit konfigurierbaren Komponenten für Ablation-Studien"""

    def __init__(self, config: dict, ablation_config: AblationConfig):
        self.config = config
        self.ablation_config = ablation_config
        self.logger = logging.getLogger(__name__)

        # Hyperparameter
        self.S_MAX = config["hyperparameters"]["max_sentences_per_chapter"]  # 50
        self.K_MAX = config["hyperparameters"]["num_chapters"]             # 8

        # Modelle nur laden wenn benötigt
        self._init_models()

        # Dimensionen berechnen basierend auf aktiven Komponenten
        self.feature_dim = self._calculate_feature_dimension()

    def _init_models(self):
        """Initialisiert nur die für die Ablation benötigten Modelle"""
        self.models = {}

        if EncoderComponent.SENTENCE_LEVEL in self.ablation_config.enabled_components:
            self.models['sbert_sent'] = SentenceTransformer(self.config["models"]["sbert_sentence"])

        if EncoderComponent.CHAPTER_LEVEL in self.ablation_config.enabled_components:
            self.models['sbert_long'] = SentenceTransformer(self.config["models"]["sbert_long"])

        if EncoderComponent.COHERENCE in self.ablation_config.enabled_components:
            self.models['coherence'] = pipeline(
                "text-classification",
                model=self.config["models"]["coherence"],
                device=0
            )

        if EncoderComponent.ADEQUACY in self.ablation_config.enabled_components:
            # Für Adäquanz benötigen wir SBERT für Titel-Embedding
            if 'sbert_sent' not in self.models:
                self.models['sbert_sent'] = SentenceTransformer(self.config["models"]["sbert_sentence"])

            # Zero-Shot für Satz-Adäquanz in no_structure
            if self.ablation_config.name == "no_structure":
                self.models['zeroshot'] = pipeline(
                    "zero-shot-classification",
                    model=self.config["models"]["zeroshot"],
                    device=0
                )

    def _calculate_feature_dimension(self):
        """Berechnet Dimension basierend auf aktiven Komponenten und Variante"""
        dim = 768  # Basis-Embedding (SBERT)

        if EncoderComponent.COHERENCE in self.ablation_config.enabled_components:
            dim += 1

        if EncoderComponent.ADEQUACY in self.ablation_config.enabled_components:
            dim += 1

        return dim

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Kosinus-Ähnlichkeit zweier Tensoren"""
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def _encode_sentence(self, sentence: str, prev_sentence: str = "",
                        chapter_label: str = "", title_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Kodiert einen Satz basierend auf aktiven Komponenten"""
        features = []

        # Bestimme das Ziel-Device (GPU falls verfügbar)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Satz-Embedding
        sent_emb = self.models['sbert_sent'].encode(sentence, convert_to_tensor=True).to(device)
        features.append(sent_emb)

        # Kohärenz-Score
        if EncoderComponent.COHERENCE in self.ablation_config.enabled_components:
            coh_input = f"{prev_sentence} {sentence}" if prev_sentence else sentence
            # coh_label = self.models['coherence'](coh_input, truncation=True)[0]["label"]
            # coh_prob = float(self.models['coherence'](coh_input, truncation=True)[0]["score"])
            # coh_score = coh_prob if coh_label == "COHERENT" else 1 - coh_prob
            coh_score = predict_coherence(coh_input)
            features.append(torch.tensor([coh_score], device=device))

        # Zero-Shot-Score (für no_structure Variante)
        if (EncoderComponent.ADEQUACY in self.ablation_config.enabled_components and
            self.ablation_config.name == "no_structure"):
            zs_score = self.models['zeroshot'](
                sentence,
                [f"Dieser Satz beschreibt {chapter_label}"]
            )["scores"][0]
            features.append(torch.tensor([zs_score], device=device))

        return torch.cat(features, dim=0)

    def _encode_chapter(self, chapter_text: str, chapter_label: str,
                       title_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Kodiert ein Kapitel basierend auf aktiven Komponenten"""
        features = []

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Kapitel-Embedding
        if EncoderComponent.CHAPTER_LEVEL in self.ablation_config.enabled_components:
            chap_emb = self.models['sbert_long'].encode(chapter_text, convert_to_tensor=True).to(device)
        else:
            # Fallback: Nutze Satz-Encoder für Kapitel
            chap_emb = self.models['sbert_sent'].encode(chapter_text, convert_to_tensor=True).to(device)
        features.append(chap_emb)

        # Kapitel-Kohärenz
        if EncoderComponent.COHERENCE in self.ablation_config.enabled_components:
            # coh_label = self.models['coherence'](chapter_text, truncation=True)[0]["label"]
            # coh_prob = float(self.models['coherence'](chapter_text, truncation=True)[0]["score"])
            # coh_score = coh_prob if coh_label == "COHERENT" else 1-coh_prob
            coh_score = predict_coherence(chapter_text)
            features.append(torch.tensor([coh_score], device=device))

        # Kapitel-Adäquanz (Cosinus-Similarität mit Titel)
        if (EncoderComponent.ADEQUACY in self.ablation_config.enabled_components and
            title_embedding is not None):
            # Nutze das gleiche Embedding wie für Features
            if EncoderComponent.CHAPTER_LEVEL in self.ablation_config.enabled_components:
                chap_for_adequacy = self.models['sbert_long'].encode(chapter_text, convert_to_tensor=True).to(device)
            else:
                chap_for_adequacy = self.models['sbert_sent'].encode(chapter_text, convert_to_tensor=True).to(device)
            adequacy_score = self._cosine_similarity(chap_for_adequacy, title_embedding)
            features.append(torch.tensor([adequacy_score], device=device))

        return torch.cat(features, dim=0)

    def encode_expose(self, expose_data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kodiert ein komplettes Exposé basierend auf der Ablation-Variante"""
        chapters = expose_data["chapters"]

        # Titel-Embedding für Adäquanz
        topic_embedding = None
        if EncoderComponent.ADEQUACY in self.ablation_config.enabled_components:
            topic_embedding = self.models['sbert_sent'].encode(
                expose_data["topic"],
                convert_to_tensor=True
            )

        # Sammle alle Repräsentationen
        representations = []

        # Varianten-spezifische Kodierung
        if self.ablation_config.name == "full_model":
            # Schema: [Absatz1, Satz1, Satz2, ..., Absatz2, Satz1, Satz2, ...]
            representations = self._encode_full_model(chapters, topic_embedding)

        elif self.ablation_config.name == "no_structure":
            # Schema: [Satz1, Satz2, Satz3, ...] (alle Sätze sequenziell)
            representations = self._encode_no_structure(chapters, topic_embedding)

        elif self.ablation_config.name == "no_textfeatures":
            # Schema: [Absatz1, Satz1, Satz2, ..., Absatz2, Satz1, Satz2, ...]
            representations = self._encode_no_textfeatures(chapters)

        elif self.ablation_config.name == "minimal":
            # Schema: [Satz1, Satz2, Satz3, ...] (nur Satz-Embeddings)
            representations = self._encode_minimal(chapters)

        # Konvertiere zu Matrix mit Padding
        if not representations:
            M = torch.zeros((1, self.feature_dim))
            mask = torch.zeros(1, dtype=torch.long)
        else:
            # Finde maximale Dimension
            max_dim = max(r.shape[0] for r in representations)
            device = representations[0].device

            # Padding auf maximale Dimension
            padded_representations = []
            for r in representations:
                if r.shape[0] < max_dim:
                    # Padding mit Nullen
                    padding = torch.zeros(max_dim - r.shape[0], device=device)
                    padded_r = torch.cat([r, padding], dim=0)
                else:
                    padded_r = r
                padded_representations.append(padded_r)

            M = torch.stack(padded_representations)
            mask = torch.ones(len(representations), dtype=torch.long)

        return M, mask

    def _encode_full_model(self, chapters: dict, title_embedding: torch.Tensor) -> List[torch.Tensor]:
        """Kodierung für full_model: [Absatz, Satz1, Satz2, ...]"""
        representations = []

        for k_i, (kapitel_title, satz_set) in enumerate(chapters.items()):
            if k_i >= self.K_MAX:
                break

            satz_list = list(satz_set)[:self.S_MAX]

            # Zuerst Kapitel-Repräsentation
            chap_text = " ".join(satz_list)
            chap_repr = self._encode_chapter(chap_text, kapitel_title, title_embedding)
            representations.append(chap_repr)

            # Dann Satz-Repräsentationen
            for j, satz in enumerate(satz_list):
                prev_satz = satz_list[j-1] if j > 0 else ""
                sent_repr = self._encode_sentence(satz, prev_satz, kapitel_title, title_embedding)
                representations.append(sent_repr)

        return representations

    def _encode_no_structure(self, chapters: dict, title_embedding: torch.Tensor) -> List[torch.Tensor]:
        """Kodierung für no_structure: [Satz1, Satz2, ...] (alle Sätze sequenziell)"""
        representations = []

        for k_i, (kapitel_title, satz_set) in enumerate(chapters.items()):
            if k_i >= self.K_MAX:
                break

            satz_list = list(satz_set)[:self.S_MAX]

            # Nur Satz-Repräsentationen (keine Kapitel)
            for j, satz in enumerate(satz_list):
                prev_satz = satz_list[j-1] if j > 0 else ""
                sent_repr = self._encode_sentence(satz, prev_satz, kapitel_title, title_embedding)
                representations.append(sent_repr)

        return representations

    def _encode_no_textfeatures(self, chapters: dict) -> List[torch.Tensor]:
        """Kodierung für no_textfeatures: [Absatz, Satz1, Satz2, ...] (nur Embeddings)"""
        representations = []

        for k_i, (kapitel_title, satz_set) in enumerate(chapters.items()):
            if k_i >= self.K_MAX:
                break

            satz_list = list(satz_set)[:self.S_MAX]

            # Zuerst Kapitel-Embedding
            chap_text = " ".join(satz_list)
            chap_repr = self._encode_chapter(chap_text, kapitel_title)
            representations.append(chap_repr)

            # Dann Satz-Embeddings
            for satz in satz_list:
                sent_repr = self._encode_sentence(satz)
                representations.append(sent_repr)

        return representations

    def _encode_minimal(self, chapters: dict) -> List[torch.Tensor]:
        """Kodierung für minimal: [Satz1, Satz2, ...] (nur Satz-Embeddings)"""
        representations = []

        for k_i, (kapitel_title, satz_set) in enumerate(chapters.items()):
            if k_i >= self.K_MAX:
                break

            satz_list = list(satz_set)[:self.S_MAX]

            # Nur Satz-Embeddings
            for satz in satz_list:
                sent_repr = self._encode_sentence(satz)
                representations.append(sent_repr)

        return representations

    def get_info(self) -> dict:
        """Gibt Informationen über die aktuelle Konfiguration zurück"""
        return {
            "variant_name": self.ablation_config.name,
            "description": self.ablation_config.description,
            "enabled_components": [comp.value for comp in self.ablation_config.enabled_components],
            "feature_dimension": self.feature_dim,
            "loaded_models": list(self.models.keys())
        }


class ExperimentRunner:
    """Orchestriert Ablation-Experimente"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        setup_logging(self.config["logging"]["file"], self.config["logging"]["level"])
        self.logger = logging.getLogger(__name__)

        # Ausgabeverzeichnis
        self.out_dir = Path(self.config["paths"]["encoder_output_dir"])
        self.out_dir.mkdir(exist_ok=True)

        # Daten laden
        self.expose_data = self._load_expose_data()

    def _load_expose_data(self) -> dict:
        """Lädt und bereitet Exposé-Daten vor (wie im Original-Code)"""
        jsonl_path = self.config["paths"]["preprocessed_jsonl"]
        topics_path = self.config["paths"]["topics_jsonl"]

        # Topics aus expose_topics.jsonl laden
        topics_map = {}

        try:
            import json
            with open(topics_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Leere Zeile überspringen
                        continue
                    try:
                        entry = json.loads(line)
                        topics_map[entry["expose_id"]] = entry["topic"]
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Ungültiges JSON in expose_topics.jsonl Zeile {line_num}: {e}")
                    except KeyError as e:
                        self.logger.warning(f"Fehlender Schlüssel in expose_topics.jsonl Zeile {line_num}: {e}")
        except FileNotFoundError:
            self.logger.warning("expose_topics.jsonl nicht gefunden. Topics werden nicht geladen.")

        # try:
        #     for entry in read_jsonl(topics_path):
        #         topics_map[entry["expose_id"]] = entry["topic"]
        # except FileNotFoundError:
        #     self.logger.warning("expose_topics.jsonl nicht gefunden. Topics werden nicht geladen.")

        # Daten gruppieren
        expose_data = {}
        comments_map = {}

        for entry in read_jsonl(jsonl_path):
            eid = entry["expose_id"]
            kapit = entry["kapitel"]
            satz = entry["sentence_text"]
            ctext = entry["comment_text"]
            comments_map.setdefault((eid, kapit, satz), []).append(ctext)
            expose_data.setdefault(eid, {
                "chapters": {},
                "title": eid.replace("_", " "),
                "topic": topics_map.get(eid, "")
            })
            expose_data[eid]["chapters"].setdefault(kapit, set()).add(satz)

        # Unkommentierte Sätze aus DOCX hinzufügen
        for eid in list(expose_data.keys()):
            docx_path = Path(self.config["paths"]["raw_exposes_dir"]) / f"{eid}.docx"
            if not docx_path.exists():
                self.logger.warning(f"Docx für {eid} nicht gefunden.")
                continue

            doc = Document(str(docx_path))
            flow_lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

            with open("../data/heading_terms.json", "r", encoding="utf-8") as f:
                heading_map = json.load(f)

            # Kapitel extrahieren (benötigt preprocessing-Funktion)
            try:
                from preprocessing import extract_chapters
                chapters_text = extract_chapters(flow_lines, heading_map)

                for title, chap_text in chapters_text.items():
                    sents = [sent.strip() for sent in sent_tokenize(chap_text, language='german')
                             if sent.strip()]
                    expose_data[eid]["chapters"].setdefault(title, set()).update(sents)
            except ImportError as ie:
                self.logger.warning(f"extract_chapters Funktion nicht verfügbar. {ie}")

        return expose_data

    def run_single_variant(self, variant_name: str) -> None:
        """Führt ein einzelnes Ablation-Experiment aus"""
        variants = AblationConfig.get_predefined_variants()
        if variant_name not in variants:
            raise ValueError(f"Unbekannte Variante: {variant_name}")

        ablation_config = variants[variant_name]
        encoder = ModularEncoder(self.config, ablation_config)

        self.logger.info(f"Starte Ablation-Variante: {ablation_config.description}")
        self.logger.info(f"Encoder-Info: {encoder.get_info()}")

        # Ausgabeverzeichnis für diese Variante
        variant_dir = self.out_dir / variant_name
        variant_dir.mkdir(exist_ok=True)

        # Alle Exposés kodieren
        for eid, data in tqdm(self.expose_data.items(), desc=f"Encoding {variant_name}"):
            try:
                M, mask = encoder.encode_expose(data)

                # Speichern mit serialisierbarer Konfiguration
                out_path = variant_dir / f"{eid}_M.pt"
                torch.save({
                    "M": M,
                    "mask": mask,
                    "variant_name": variant_name,
                    "feature_dim": encoder.feature_dim,
                    "ablation_config": ablation_config.to_serializable_dict(),
                    "matrix_shape": M.shape,
                    "active_rows": mask.sum().item()
                }, out_path)

            except Exception as e:
                self.logger.error(f"Fehler bei Exposé {eid} in Variante {variant_name}: {e}")

        self.logger.info(f"Variante {variant_name} abgeschlossen → {variant_dir}")

    def run_all_variants(self) -> None:
        """Führt alle 4 strategischen Ablation-Varianten aus"""
        variants = AblationConfig.get_predefined_variants()

        for variant_name in variants.keys():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starte Variante: {variant_name}")
            self.logger.info(f"{'='*50}")

            try:
                self.run_single_variant(variant_name)
            except Exception as e:
                self.logger.error(f"Fehler in Variante {variant_name}: {e}")
                continue

        self.logger.info("\nAlle Ablation-Experimente abgeschlossen!")

    def run_custom_variant(self, components: List[str], name: str = "custom") -> None:
        """Führt benutzerdefinierte Ablation aus"""
        try:
            enabled_components = [EncoderComponent(comp) for comp in components]
        except ValueError as e:
            raise ValueError(f"Ungültige Komponente: {e}")

        custom_config = AblationConfig(
            name=name,
            description=f"Custom: {', '.join(components)}",
            enabled_components=enabled_components
        )

        encoder = ModularEncoder(self.config, custom_config)

        self.logger.info(f"Starte Custom-Variante: {custom_config.description}")

        # Kodierung ausführen (analog zu run_single_variant)
        variant_dir = self.out_dir / name
        variant_dir.mkdir(exist_ok=True)

        for eid, data in tqdm(self.expose_data.items(), desc=f"Encoding {name}"):
            try:
                M, mask = encoder.encode_expose(data)
                out_path = variant_dir / f"{eid}_M.pt"
                torch.save({
                    "M": M,
                    "mask": mask,
                    "variant_name": name,
                    "feature_dim": encoder.feature_dim,
                    "ablation_config": custom_config.to_serializable_dict(),
                    "matrix_shape": M.shape,
                    "active_rows": mask.sum().item()
                }, out_path)
            except Exception as e:
                self.logger.error(f"Fehler bei Exposé {eid}: {e}")


def main():
    """
    Hauptfunktion - kann direkt aus IDE gestartet werden.
    Konfiguration über globale Variablen für einfache IDE-Nutzung.
    """

    # ===== KONFIGURATION FÜR IDE-START =====
    # Diese Werte können direkt hier angepasst werden
    CONFIG_PATH = "../config/config.json"

    # Experiment-Einstellungen:
    VARIANT = "all"  # Optionen: "full_model", "no_structure", "no_textfeatures", "minimal", "all"

    # Für benutzerdefinierte Experimente:
    CUSTOM_COMPONENTS = None  # z.B. ["sentence_level", "adequacy"]
    CUSTOM_NAME = "custom"

    # ===== ENDE KONFIGURATION =====

    # Argument-Parser für Kommandozeile (optional)
    parser = argparse.ArgumentParser(description="Modularer Encoder für Ablation-Experimente")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Pfad zu config.json")
    parser.add_argument("--variant", type=str, choices=["full_model", "no_structure", "no_textfeatures", "minimal", "all"],
                       default=VARIANT, help="Ablation-Variante")
    parser.add_argument("--custom", nargs="+", choices=["sentence_level", "chapter_level", "adequacy", "coherence"],
                       default=CUSTOM_COMPONENTS, help="Benutzerdefinierte Komponenten-Kombination")
    parser.add_argument("--name", type=str, default=CUSTOM_NAME, help="Name für benutzerdefinierte Variante")

    # Versuche Argumente zu parsen, falls aus Kommandozeile gestartet
    try:
        args = parser.parse_args()
    except SystemExit:
        # Falls aus IDE gestartet ohne Argumente, nutze Default-Werte
        class DefaultArgs:
            config = CONFIG_PATH
            variant = VARIANT
            custom = CUSTOM_COMPONENTS
            name = CUSTOM_NAME
        args = DefaultArgs()

    # Experiment-Runner initialisieren
    try:
        runner = ExperimentRunner(args.config)

        print(f"🚀 Starte Ablation-Experimente...")
        print(f"📁 Konfiguration: {args.config}")
        print(f"🎯 Variante: {args.variant}")

        if args.custom:
            # Benutzerdefinierte Variante
            print(f"🔧 Custom-Komponenten: {args.custom}")
            runner.run_custom_variant(args.custom, args.name)
        elif args.variant == "all":
            # Alle 4 strategischen Varianten
            print("📊 Führe alle 4 strategischen Ablation-Varianten aus...")
            runner.run_all_variants()
        else:
            # Einzelne vordefinierte Variante
            print(f"🎯 Führe Variante '{args.variant}' aus...")
            runner.run_single_variant(args.variant)

        print("✅ Alle Experimente erfolgreich abgeschlossen!")

    except FileNotFoundError as e:
        print(f"❌ Fehler: Konfigurationsdatei nicht gefunden: {e}")
        print("💡 Tipp: Passe CONFIG_PATH in der main()-Funktion an.")
    except Exception as e:
        print(f"❌ Fehler beim Ausführen der Experimente: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()