"""
evaluate_multivariant.py

Evaluiert Multi-Varianten-Feedbacks aus dem Inference-Engine:
- Berechnet BERTScores für alle Varianten vs. Gold-Standard
- Führt Ablationsstudie zwischen Varianten durch
- Speichert BERTScores in CSV und Ablationsstudie in TXT

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import csv
import logging
import argparse
import numpy as np
from pathlib import Path
from bert_score import score


from utils.utils import load_config, setup_logging


class MultiVariantEvaluator:
    """Evaluiert Multi-Varianten-Feedbacks und führt Ablationsstudien durch"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        setup_logging(self.config["logging"]["file"], self.config["logging"]["level"])
        self.logger = logging.getLogger(__name__)

        # Pfade
        self.gold_dir = Path(self.config["paths"]["gold_bullet_dir"])
        self.inference_dir = Path(self.config["paths"]["inference_output_dir"])
        self.eval_dir = Path(self.config["paths"].get("evaluation_dir", "evaluation_output"))
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Ausgabedateien
        self.bert_csv = self.eval_dir / "bert_scores_multivariant.csv"
        self.ablation_txt = self.eval_dir / "ablation_study.txt"

        # Datenstrukturen für Ergebnisse
        self.bert_scores = {}  # {variant: {expose_id: {p, r, f1}}}
        self.variants = []

    def discover_variants(self):
        """Entdeckt alle verfügbaren Varianten im Inference-Output-Verzeichnis"""
        if not self.inference_dir.exists():
            self.logger.error(f"Inference-Output-Verzeichnis nicht gefunden: {self.inference_dir}")
            return []

        variants = [d.name for d in self.inference_dir.iterdir() if d.is_dir()]
        self.variants = sorted(variants)
        self.logger.info(f"Gefundene Varianten: {self.variants}")
        return self.variants

    def get_expose_ids(self):
        """Ermittelt alle verfügbaren Exposé-IDs aus Gold-Standard-Dateien"""
        gold_files = list(self.gold_dir.glob("*_gold_bullets.txt"))
        expose_ids = [f.stem.replace("_gold_bullets", "") for f in gold_files]
        self.logger.info(f"Gefundene Exposés: {len(expose_ids)}")
        return sorted(expose_ids)

    def calculate_bert_scores(self):
        """Berechnet BERTScores für alle Varianten und Exposés"""
        expose_ids = self.get_expose_ids()

        if not expose_ids:
            self.logger.error("Keine Gold-Standard-Dateien gefunden!")
            return

        total_combinations = len(self.variants) * len(expose_ids)
        completed = 0

        # CSV-Header schreiben
        with open(self.bert_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["variant", "expose_id", "precision", "recall", "f1"])

            for variant in self.variants:
                self.bert_scores[variant] = {}
                variant_dir = self.inference_dir / variant

                if not variant_dir.exists():
                    self.logger.warning(f"Varianten-Verzeichnis nicht gefunden: {variant_dir}")
                    continue

                self.logger.info(f"Evaluiere Variante: {variant}")

                for expose_id in expose_ids:
                    # Gold-Standard laden
                    gold_file = self.gold_dir / f"{expose_id}_gold_bullets.txt"
                    gen_file = variant_dir / f"{expose_id}_generated.txt"

                    if not gold_file.exists():
                        self.logger.warning(f"Gold-Datei nicht gefunden: {gold_file}")
                        continue

                    if not gen_file.exists():
                        self.logger.warning(f"Generierte Datei nicht gefunden: {gen_file}")
                        continue

                    try:
                        # Texte laden
                        with open(gold_file, "r", encoding="utf-8") as f:
                            ref_text = f.read().strip()
                        with open(gen_file, "r", encoding="utf-8") as f:
                            hyp_text = f.read().strip()

                        if not ref_text or not hyp_text:
                            self.logger.warning(f"Leere Texte für {variant}/{expose_id}")
                            continue

                        # BERTScore berechnen
                        P, R, F1 = score(
                            cands=[hyp_text],
                            refs=[ref_text],
                            lang="de",
                            model_type="bert-base-multilingual-cased",
                            verbose=False
                        )

                        p, r, f1 = P.mean().item(), R.mean().item(), F1.mean().item()

                        # Ergebnisse speichern
                        self.bert_scores[variant][expose_id] = {
                            'precision': p,
                            'recall': r,
                            'f1': f1
                        }

                        # In CSV schreiben
                        writer.writerow([variant, expose_id, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])

                        completed += 1
                        if completed % 10 == 0:
                            progress = (completed / total_combinations) * 100
                            self.logger.info(f"Fortschritt: {completed}/{total_combinations} ({progress:.1f}%)")

                    except Exception as e:
                        self.logger.error(f"Fehler bei {variant}/{expose_id}: {e}")
                        continue

        self.logger.info(f"BERTScore-Berechnung abgeschlossen. Ergebnisse in {self.bert_csv}")

    def calculate_aggregate_stats(self):
        """Berechnet aggregierte Statistiken für jede Variante"""
        stats = {}

        for variant in self.variants:
            if variant not in self.bert_scores or not self.bert_scores[variant]:
                continue

            scores = self.bert_scores[variant]

            # Metriken sammeln
            precisions = [s['precision'] for s in scores.values()]
            recalls = [s['recall'] for s in scores.values()]
            f1s = [s['f1'] for s in scores.values()]

            if not f1s:  # Keine gültigen Scores
                continue

            stats[variant] = {
                'count': len(f1s),
                'precision': {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions),
                    'min': np.min(precisions),
                    'max': np.max(precisions)
                },
                'recall': {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls),
                    'min': np.min(recalls),
                    'max': np.max(recalls)
                },
                'f1': {
                    'mean': np.mean(f1s),
                    'std': np.std(f1s),
                    'min': np.min(f1s),
                    'max': np.max(f1s)
                }
            }

        return stats

    def perform_statistical_tests(self, stats):
        """Führt statistische Tests zwischen Varianten durch"""
        if len(self.variants) < 2:
            return {}

        test_results = {}

        # Alle Paarvergleiche
        for i, var1 in enumerate(self.variants):
            for var2 in self.variants[i + 1:]:
                if var1 not in self.bert_scores or var2 not in self.bert_scores:
                    continue

                # F1-Scores für beide Varianten sammeln
                # Nur gemeinsame Exposés verwenden
                common_exposes = set(self.bert_scores[var1].keys()) & set(self.bert_scores[var2].keys())

                if len(common_exposes) < 3:  # Zu wenige Datenpunkte
                    continue

                f1_var1 = [self.bert_scores[var1][exp]['f1'] for exp in common_exposes]
                f1_var2 = [self.bert_scores[var2][exp]['f1'] for exp in common_exposes]

                # Paired t-test
                try:
                    t_stat, t_pval = stats.ttest_rel(f1_var1, f1_var2)

                    # Wilcoxon signed-rank test (non-parametric alternative)
                    w_stat, w_pval = stats.wilcoxon(f1_var1, f1_var2, alternative='two-sided')

                    # Effektgröße (Cohen's d)
                    pooled_std = np.sqrt((np.var(f1_var1) + np.var(f1_var2)) / 2)
                    cohens_d = (np.mean(f1_var1) - np.mean(f1_var2)) / pooled_std if pooled_std > 0 else 0

                    test_results[f"{var1}_vs_{var2}"] = {
                        'n_samples': len(common_exposes),
                        'mean_diff': np.mean(f1_var1) - np.mean(f1_var2),
                        't_statistic': t_stat,
                        't_pvalue': t_pval,
                        'wilcoxon_statistic': w_stat,
                        'wilcoxon_pvalue': w_pval,
                        'cohens_d': cohens_d,
                        'var1_mean': np.mean(f1_var1),
                        'var2_mean': np.mean(f1_var2)
                    }

                except Exception as e:
                    self.logger.warning(f"Statistischer Test für {var1} vs {var2} fehlgeschlagen: {e}")
                    continue

        return test_results

    def write_ablation_study(self):
        """Schreibt die vollständige Ablationsstudie in eine Textdatei"""
        stats = self.calculate_aggregate_stats()
        test_results = self.perform_statistical_tests(stats)

        with open(self.ablation_txt, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATIONSSTUDIE: MULTI-VARIANTEN-FEEDBACK-EVALUATION\n")
            f.write("=" * 80 + "\n\n")

            # Übersicht
            f.write(f"Anzahl Varianten: {len(self.variants)}\n")
            f.write(f"Evaluierte Varianten: {', '.join(stats.keys())}\n")
            f.write(
                f"Datum: {self.logger.handlers[0].formatter.formatTime(logging.LogRecord('', 0, '', 0, '', (), None), '%Y-%m-%d %H:%M:%S') if self.logger.handlers else 'N/A'}\n\n")

            # Aggregierte Statistiken pro Variante
            f.write("1. AGGREGIERTE STATISTIKEN PRO VARIANTE\n")
            f.write("-" * 50 + "\n")

            # Ranking nach F1-Score
            ranked_variants = sorted(stats.items(), key=lambda x: x[1]['f1']['mean'], reverse=True)

            for rank, (variant, stat) in enumerate(ranked_variants, 1):
                f.write(f"\n{rank}. {variant}\n")
                f.write(f"   Anzahl Exposés: {stat['count']}\n")
                f.write(f"   Precision: {stat['precision']['mean']:.4f} ± {stat['precision']['std']:.4f} "
                        f"(Min: {stat['precision']['min']:.4f}, Max: {stat['precision']['max']:.4f})\n")
                f.write(f"   Recall:    {stat['recall']['mean']:.4f} ± {stat['recall']['std']:.4f} "
                        f"(Min: {stat['recall']['min']:.4f}, Max: {stat['recall']['max']:.4f})\n")
                f.write(f"   F1-Score:  {stat['f1']['mean']:.4f} ± {stat['f1']['std']:.4f} "
                        f"(Min: {stat['f1']['min']:.4f}, Max: {stat['f1']['max']:.4f})\n")

            # Statistische Tests
            if test_results:
                f.write(f"\n\n2. PAARWEISE STATISTISCHE VERGLEICHE\n")
                f.write("-" * 50 + "\n")
                f.write("Signifikanzniveau: α = 0.05\n")
                f.write("Effektgröße (Cohen's d): klein=0.2, mittel=0.5, groß=0.8\n\n")

                significant_comparisons = []

                for comparison, result in test_results.items():
                    var1, var2 = comparison.replace("_vs_", " vs. ").split(" vs. ")

                    f.write(f"{comparison.replace('_vs_', ' vs. ')}:\n")
                    f.write(f"   Stichprobengröße: {result['n_samples']}\n")
                    f.write(f"   Mittelwert-Differenz: {result['mean_diff']:.4f} "
                            f"({result['var1_mean']:.4f} - {result['var2_mean']:.4f})\n")
                    f.write(f"   T-Test: t = {result['t_statistic']:.4f}, p = {result['t_pvalue']:.4f}\n")
                    f.write(
                        f"   Wilcoxon: W = {result['wilcoxon_statistic']:.4f}, p = {result['wilcoxon_pvalue']:.4f}\n")
                    f.write(f"   Effektgröße (Cohen's d): {result['cohens_d']:.4f}\n")

                    # Signifikanz bewerten
                    is_significant = result['t_pvalue'] < 0.05
                    effect_size = abs(result['cohens_d'])

                    if effect_size < 0.2:
                        effect_desc = "vernachlässigbar"
                    elif effect_size < 0.5:
                        effect_desc = "klein"
                    elif effect_size < 0.8:
                        effect_desc = "mittel"
                    else:
                        effect_desc = "groß"

                    f.write(f"   Signifikant: {'JA' if is_significant else 'NEIN'} "
                            f"(Effekt: {effect_desc})\n")

                    if is_significant:
                        winner = var1 if result['mean_diff'] > 0 else var2
                        significant_comparisons.append((comparison, winner, result['mean_diff'], result['t_pvalue']))

                    f.write("\n")

                # Zusammenfassung signifikanter Unterschiede
                if significant_comparisons:
                    f.write("3. ZUSAMMENFASSUNG SIGNIFIKANTER UNTERSCHIEDE\n")
                    f.write("-" * 50 + "\n")

                    significant_comparisons.sort(key=lambda x: x[3])  # Nach p-Wert sortieren

                    for comp, winner, diff, pval in significant_comparisons:
                        f.write(
                            f"• {winner} übertrifft {comp.replace('_vs_', ' vs. ').replace(winner, '').replace(' vs. ', '')} "
                            f"um {abs(diff):.4f} F1-Punkte (p = {pval:.4f})\n")

                else:
                    f.write("3. ZUSAMMENFASSUNG\n")
                    f.write("-" * 50 + "\n")
                    f.write("Keine statistisch signifikanten Unterschiede zwischen den Varianten gefunden.\n")

            # Empfehlungen
            f.write(f"\n\n4. EMPFEHLUNGEN\n")
            f.write("-" * 50 + "\n")

            if ranked_variants:
                best_variant = ranked_variants[0][0]
                best_f1 = ranked_variants[0][1]['f1']['mean']
                f.write(f"• Beste Variante: {best_variant} (F1: {best_f1:.4f})\n")

                if len(ranked_variants) > 1:
                    second_best = ranked_variants[1][0]
                    second_f1 = ranked_variants[1][1]['f1']['mean']
                    diff = best_f1 - second_f1
                    f.write(f"• Zweitbeste Variante: {second_best} (F1: {second_f1:.4f})\n")
                    f.write(f"• Performance-Gap: {diff:.4f} F1-Punkte\n")

                # Variabilität bewerten
                best_std = ranked_variants[0][1]['f1']['std']
                f.write(
                    f"• Stabilität beste Variante: {'stabil' if best_std < 0.05 else 'moderat' if best_std < 0.1 else 'variabel'} "
                    f"(σ = {best_std:.4f})\n")

        self.logger.info(f"Ablationsstudie gespeichert in: {self.ablation_txt}")

    def run_evaluation(self):
        """Führt die komplette Evaluation durch"""
        self.logger.info("Starte Multi-Varianten-Evaluation...")

        # Varianten entdecken
        if not self.discover_variants():
            self.logger.error("Keine Varianten gefunden!")
            return

        # BERTScores berechnen
        self.calculate_bert_scores()

        # Ablationsstudie durchführen
        self.write_ablation_study()

        self.logger.info("Multi-Varianten-Evaluation abgeschlossen!")
        self.logger.info(f"Ergebnisse:")
        self.logger.info(f"  - BERTScores: {self.bert_csv}")
        self.logger.info(f"  - Ablationsstudie: {self.ablation_txt}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Varianten-Evaluation mit Ablationsstudie")
    parser.add_argument("--config", type=str, default="config2.json",
                        help="Pfad zur Konfigurationsdatei")
    args = parser.parse_args()

    evaluator = MultiVariantEvaluator(args.config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()