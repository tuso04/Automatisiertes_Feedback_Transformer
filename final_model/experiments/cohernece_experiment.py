"""
coherence_analysis_export.py

Standalone-Skript zum Exportieren von BERT-basierten Kohärenz-Analysen.
Fokussiert sich nur auf Satz- und Kapitel-Ebene mit BERT-Kohärenz-Scores.

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import csv
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from tqdm import tqdm

# Imports aus dem ursprünglichen encoder_variants.py
from final_model.model_code.encoder_ablation import (
    ModularEncoder,
    AblationConfig,
    EncoderComponent,
    ExperimentRunner
)
from utils.utils import load_config, setup_logging


class BERTCoherenceAnalyzer:
    """Analysiert BERT-basierte Kohärenz-Scores auf Satz- und Kapitel-Ebene"""

    def __init__(self, config_path: str, output_dir: str = "bert_coherence_analysis"):
        """
        Initialisiert den Analyzer

        Args:
            config_path: Pfad zur Konfigurationsdatei
            output_dir: Ausgabeverzeichnis für Analysen
        """
        self.config = load_config(config_path)
        setup_logging(self.config["logging"]["file"], self.config["logging"]["level"])
        self.logger = logging.getLogger(__name__)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Lade Exposé-Daten
        self.logger.info("Lade Exposé-Daten...")
        temp_runner = ExperimentRunner(config_path)
        self.expose_data = temp_runner.expose_data
        self.logger.info(f"✅ {len(self.expose_data)} Exposés geladen")

    def analyze_variant_coherence(self, variant_name: str) -> Optional[str]:
        """
        Analysiert BERT-Kohärenz für eine Ablation-Variante

        Args:
            variant_name: Name der Variante

        Returns:
            str: Pfad zur CSV-Datei oder None wenn nicht möglich
        """
        variants = AblationConfig.get_predefined_variants()
        if variant_name not in variants:
            raise ValueError(f"Unbekannte Variante: {variant_name}. Verfügbar: {list(variants.keys())}")

        ablation_config = variants[variant_name]

        # Prüfe ob Variante Kohärenz-Komponente hat
        if EncoderComponent.COHERENCE not in ablation_config.enabled_components:
            self.logger.warning(f"Variante '{variant_name}' hat keine Kohärenz-Komponente. Überspringe...")
            return None

        self.logger.info(f"📊 Starte BERT-Kohärenz-Analyse für: {variant_name}")
        self.logger.info(f"Beschreibung: {ablation_config.description}")

        # Encoder für diese Variante laden
        encoder = ModularEncoder(self.config, ablation_config)

        if 'coherence' not in encoder.models:
            self.logger.error(f"Kohärenz-Modell nicht geladen für Variante {variant_name}")
            return None

        # Sammle Kohärenz-Daten
        coherence_data = []

        for expose_id, data in tqdm(self.expose_data.items(),
                                    desc=f"Analysiere {variant_name}"):
            try:
                chapters = data["chapters"]

                for chapter_idx, (chapter_title, sentence_set) in enumerate(chapters.items()):
                    if chapter_idx >= encoder.K_MAX:
                        break

                    sentence_list = list(sentence_set)[:encoder.S_MAX]
                    if not sentence_list:
                        continue

                    chapter_text = " ".join(sentence_list)

                    # === KAPITEL-LEVEL KOHÄRENZ ===
                    chapter_coherence = self._get_bert_coherence_score(encoder, chapter_text)

                    coherence_data.append({
                        'expose_id': expose_id,
                        'expose_title': data["title"],
                        'chapter_index': chapter_idx,
                        'chapter_title': chapter_title,
                        'structure_level': 'chapter',
                        'text_content': chapter_text,
                        'text_length_chars': len(chapter_text),
                        'text_length_words': len(chapter_text.split()),
                        'sentence_count_in_chapter': len(sentence_list),
                        'coherence_score': chapter_coherence,
                        'previous_sentence': '',  # Nicht relevant für Kapitel
                        'sentence_position_in_chapter': -1,  # -1 für Kapitel-Level
                        'variant_name': variant_name
                    })

                    # === SATZ-LEVEL KOHÄRENZ ===
                    for sent_idx, sentence in enumerate(sentence_list):
                        previous_sentence = sentence_list[sent_idx - 1] if sent_idx > 0 else ""

                        # BERT-Kohärenz: Input ist "vorheriger_satz aktueller_satz"
                        coherence_input = f"{previous_sentence} {sentence}" if previous_sentence else sentence
                        sentence_coherence = self._get_bert_coherence_score(encoder, coherence_input)

                        coherence_data.append({
                            'expose_id': expose_id,
                            'expose_title': data["title"],
                            'chapter_index': chapter_idx,
                            'chapter_title': chapter_title,
                            'structure_level': 'sentence',
                            'text_content': sentence,
                            'text_length_chars': len(sentence),
                            'text_length_words': len(sentence.split()),
                            'sentence_count_in_chapter': len(sentence_list),
                            'coherence_score': sentence_coherence,
                            'previous_sentence': previous_sentence,
                            'sentence_position_in_chapter': sent_idx,
                            'variant_name': variant_name
                        })

            except Exception as e:
                self.logger.error(f"Fehler bei Exposé {expose_id}: {e}")
                continue

        if not coherence_data:
            self.logger.warning(f"Keine Kohärenz-Daten für Variante {variant_name} gesammelt")
            return None

        # DataFrame erstellen und erweitern
        df = pd.DataFrame(coherence_data)
        df = self._add_coherence_analysis_columns(df)

        # CSV exportieren
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bert_coherence_{variant_name}_{timestamp}.csv"
        csv_path = self.output_dir / filename

        df.to_csv(csv_path, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

        # Statistiken erstellen
        self._create_coherence_summary(df, csv_path.stem)

        self.logger.info(f"✅ BERT-Kohärenz-Analyse exportiert: {csv_path}")
        self.logger.info(f"📊 {len(df)} Einträge analysiert")
        self.logger.info(f"   → {len(df[df['structure_level'] == 'chapter'])} Kapitel")
        self.logger.info(f"   → {len(df[df['structure_level'] == 'sentence'])} Sätze")

        return str(csv_path)

    def analyze_all_coherence_variants(self) -> List[str]:
        """
        Analysiert alle Varianten mit Kohärenz-Komponente

        Returns:
            List[str]: Liste der erstellten CSV-Pfade
        """
        variants = AblationConfig.get_predefined_variants()
        coherence_variants = {
            name: config for name, config in variants.items()
            if EncoderComponent.COHERENCE in config.enabled_components
        }

        if not coherence_variants:
            self.logger.warning("Keine Varianten mit Kohärenz-Komponente gefunden")
            return []

        self.logger.info(f"Analysiere {len(coherence_variants)} Varianten mit Kohärenz:")
        for name in coherence_variants.keys():
            self.logger.info(f"  - {name}")

        results = []

        for variant_name in coherence_variants.keys():
            try:
                csv_path = self.analyze_variant_coherence(variant_name)
                if csv_path:
                    results.append(csv_path)
            except Exception as e:
                self.logger.error(f"Fehler bei Variante {variant_name}: {e}")
                continue

        # Vergleichende Zusammenfassung erstellen
        if results:
            self._create_comparative_summary(results)

        return results

    def _get_bert_coherence_score(self, encoder: ModularEncoder, text: str) -> float:
        """
        Berechnet BERT-basierten Kohärenz-Score

        Args:
            encoder: ModularEncoder-Instanz
            text: Zu analysierender Text

        Returns:
            float: Kohärenz-Score zwischen 0 und 1
        """
        try:
            #result = encoder.models['coherence'](text, truncation=True)
            #result = predict_coherence(text)
            coh_label = encoder.models['coherence'](text, truncation=True)[0]["label"]
            coh_prob = float(encoder.models['coherence'](text, truncation=True)[0]["score"])
            coh_score = coh_prob if coh_label == "COHERENT" else 1 - coh_prob
            return float(coh_score)

        except Exception as e:
            self.logger.debug(f"BERT-Kohärenz Berechnung fehlgeschlagen (Länge: {len(text)}): {e}")
            return 0.0

    def _add_coherence_analysis_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erweitert DataFrame um Analyse-Spalten"""
        df = df.copy()

        # Kohärenz-Kategorien
        df['coherence_category'] = pd.cut(
            df['coherence_score'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['niedrig', 'mittel', 'hoch', 'sehr_hoch'],
            include_lowest=True
        )

        # Textlängen-Kategorien
        df['text_length_category'] = pd.cut(
            df['text_length_words'],
            bins=[0, 10, 25, 50, float('inf')],
            labels=['sehr_kurz', 'kurz', 'mittel', 'lang'],
            include_lowest=True
        )

        # Position im Kapitel (nur für Sätze)
        def get_sentence_position_category(row):
            if row['structure_level'] != 'sentence':
                return 'kapitel'
            pos = row['sentence_position_in_chapter']
            total = row['sentence_count_in_chapter']

            if pos == 0:
                return 'erster_satz'
            elif pos < total * 0.3:
                return 'anfang'
            elif pos < total * 0.7:
                return 'mitte'
            else:
                return 'ende'

        df['position_category'] = df.apply(get_sentence_position_category, axis=1)

        # Durchschnittliche Kohärenz pro Exposé und Ebene
        expose_avg = df.groupby(['expose_id', 'structure_level'])['coherence_score'].mean().reset_index()
        expose_avg.columns = ['expose_id', 'structure_level', 'expose_level_avg_coherence']
        df = df.merge(expose_avg, on=['expose_id', 'structure_level'], how='left')

        # Relative Kohärenz (im Vergleich zum Exposé-Durchschnitt dieser Ebene)
        df['relative_coherence'] = df['coherence_score'] - df['expose_level_avg_coherence']

        # Hat vorherigen Satz (nur für Sätze relevant)
        df['has_previous_sentence'] = (df['structure_level'] == 'sentence') & (df['previous_sentence'] != '')

        return df

    def _create_coherence_summary(self, df: pd.DataFrame, base_filename: str):
        """Erstellt detaillierte Zusammenfassungsstatistiken"""
        summary_path = self.output_dir / f"{base_filename}_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== BERT-KOHÄRENZ ANALYSE ZUSAMMENFASSUNG ===\n\n")
            f.write(f"Variante: {df['variant_name'].iloc[0]}\n")
            f.write(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Grundstatistiken
            f.write("--- DATENÜBERSICHT ---\n")
            f.write(f"Anzahl Exposés: {df['expose_id'].nunique()}\n")
            f.write(f"Anzahl Kapitel: {len(df[df['structure_level'] == 'chapter'])}\n")
            f.write(f"Anzahl Sätze: {len(df[df['structure_level'] == 'sentence'])}\n\n")

            # Kohärenz-Statistiken gesamt
            f.write("--- KOHÄRENZ-SCORES GESAMT ---\n")
            coherence_stats = df['coherence_score'].describe()
            for stat, value in coherence_stats.items():
                f.write(f"{stat}: {value:.4f}\n")
            f.write("\n")

            # Kohärenz nach Strukturebene
            f.write("--- KOHÄRENZ NACH STRUKTUREBENE ---\n")
            level_stats = df.groupby('structure_level')['coherence_score'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            f.write(str(level_stats))
            f.write("\n\n")

            # Kohärenz-Kategorien Verteilung
            f.write("--- KOHÄRENZ-KATEGORIEN VERTEILUNG ---\n")
            for level in ['chapter', 'sentence']:
                level_data = df[df['structure_level'] == level]
                if not level_data.empty:
                    f.write(f"\n{level.upper()}:\n")
                    cat_counts = level_data['coherence_category'].value_counts()
                    for cat, count in cat_counts.items():
                        percentage = count / len(level_data) * 100
                        f.write(f"  {cat}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Top/Bottom Exposés nach durchschnittlicher Kohärenz
            f.write("--- TOP 5 EXPOSÉS (HÖCHSTE DURCHSCHNITTLICHE KOHÄRENZ) ---\n")
            top_exposes = df.groupby('expose_id')['coherence_score'].mean().nlargest(5)
            for expose_id, score in top_exposes.items():
                f.write(f"{expose_id}: {score:.4f}\n")
            f.write("\n")

            f.write("--- BOTTOM 5 EXPOSÉS (NIEDRIGSTE DURCHSCHNITTLICHE KOHÄRENZ) ---\n")
            bottom_exposes = df.groupby('expose_id')['coherence_score'].mean().nsmallest(5)
            for expose_id, score in bottom_exposes.items():
                f.write(f"{expose_id}: {score:.4f}\n")
            f.write("\n")

            # Satz-Position Analyse
            sentence_data = df[df['structure_level'] == 'sentence']
            if not sentence_data.empty:
                f.write("--- KOHÄRENZ NACH SATZ-POSITION ---\n")
                pos_stats = sentence_data.groupby('position_category')['coherence_score'].agg([
                    'count', 'mean', 'std'
                ]).round(4)
                f.write(str(pos_stats))
                f.write("\n\n")

                # Einfluss des vorherigen Satzes
                f.write("--- EINFLUSS VORHERIGER SATZ ---\n")
                prev_stats = sentence_data.groupby('has_previous_sentence')['coherence_score'].agg([
                    'count', 'mean', 'std'
                ]).round(4)
                f.write("Kohärenz mit/ohne vorherigen Satz:\n")
                f.write(str(prev_stats))
                f.write("\n")

        self.logger.info(f"📋 Zusammenfassung erstellt: {summary_path}")

    def _create_comparative_summary(self, csv_paths: List[str]):
        """Erstellt vergleichende Zusammenfassung aller analysierten Varianten"""
        summary_path = self.output_dir / f"comparative_coherence_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # Lade alle DataFrames
        dfs = []
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Fehler beim Laden von {csv_path}: {e}")

        if not dfs:
            return

        combined_df = pd.concat(dfs, ignore_index=True)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== VERGLEICHENDE BERT-KOHÄRENZ ANALYSE ===\n\n")
            f.write(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Anzahl analysierte Varianten: {combined_df['variant_name'].nunique()}\n")
            f.write(f"Analysierte Varianten: {', '.join(combined_df['variant_name'].unique())}\n\n")

            # Vergleich nach Varianten
            f.write("--- KOHÄRENZ-VERGLEICH NACH VARIANTEN ---\n")
            variant_comparison = combined_df.groupby(['variant_name', 'structure_level'])['coherence_score'].agg([
                'count', 'mean', 'std'
            ]).round(4)
            f.write(str(variant_comparison))
            f.write("\n\n")

            # Beste/Schlechteste Variante pro Ebene
            f.write("--- BESTE VARIANTEN PRO STRUKTUREBENE ---\n")
            for level in ['chapter', 'sentence']:
                level_means = combined_df[combined_df['structure_level'] == level].groupby('variant_name')[
                    'coherence_score'].mean()
                best_variant = level_means.idxmax()
                best_score = level_means.max()
                f.write(f"{level.upper()}: {best_variant} (Score: {best_score:.4f})\n")
            f.write("\n")

        self.logger.info(f"📊 Vergleichende Zusammenfassung erstellt: {summary_path}")


def main():
    """Hauptfunktion für Standalone-Ausführung"""

    # ===== KONFIGURATION FÜR DIREKTE AUSFÜHRUNG =====
    DEFAULT_CONFIG = "config.json"
    DEFAULT_OUTPUT_DIR = "bert_coherence_analysis"
    DEFAULT_VARIANT = "full_model"  # oder "all" für alle Varianten

    # Argument-Parser
    parser = argparse.ArgumentParser(
        description="BERT-Kohärenz-Analyse für Ablation-Experimente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
    python coherence_analysis_export.py --variant full_model
    python coherence_analysis_export.py --all-variants
    python coherence_analysis_export.py --config /path/to/config.json --variant no_structure
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Pfad zur Konfigurationsdatei (default: {DEFAULT_CONFIG})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Ausgabeverzeichnis (default: {DEFAULT_OUTPUT_DIR})"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--variant",
        type=str,
        choices=["full_model", "no_structure", "no_textfeatures", "minimal"],
        help="Einzelne Variante analysieren"
    )

    group.add_argument(
        "--all-variants",
        action="store_true",
        help="Alle Varianten mit Kohärenz-Komponente analysieren"
    )

    # Parse arguments - falls aus IDE ohne Argumente gestartet, nutze Defaults
    try:
        args = parser.parse_args()
    except SystemExit:
        print("⚠️  Keine Kommandozeilen-Argumente gefunden. Nutze Default-Konfiguration.")
        print(f"📁 Config: {DEFAULT_CONFIG}")
        print(f"🎯 Variante: {DEFAULT_VARIANT}")
        print(f"📂 Output: {DEFAULT_OUTPUT_DIR}")

        class DefaultArgs:
            config = DEFAULT_CONFIG
            output_dir = DEFAULT_OUTPUT_DIR
            variant = DEFAULT_VARIANT
            all_variants = False

        args = DefaultArgs()

    # Analyzer initialisieren
    try:
        analyzer = BERTCoherenceAnalyzer(args.config, args.output_dir)

        print("\n🚀 Starte BERT-Kohärenz-Analyse...")
        print(f"📁 Konfiguration: {args.config}")
        print(f"📂 Ausgabe: {args.output_dir}")

        if args.all_variants:
            print("📊 Analysiere alle Varianten mit Kohärenz-Komponente...")
            results = analyzer.analyze_all_coherence_variants()

            if results:
                print(f"\n✅ {len(results)} Analysen erfolgreich erstellt:")
                for result in results:
                    print(f"   → {result}")
            else:
                print("❌ Keine Analysen erstellt")

        else:
            print(f"🎯 Analysiere Variante: {args.variant}")
            result = analyzer.analyze_variant_coherence(args.variant)

            if result:
                print(f"\n✅ Analyse erfolgreich erstellt: {result}")
            else:
                print("❌ Analyse konnte nicht erstellt werden")

        print("\n🎉 BERT-Kohärenz-Analyse abgeschlossen!")

    except FileNotFoundError as e:
        print(f"❌ Konfigurationsdatei nicht gefunden: {e}")
        print(f"💡 Stelle sicher, dass '{args.config}' existiert")

    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()