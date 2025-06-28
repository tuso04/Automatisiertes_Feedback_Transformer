"""
inference_ablation.py

Multi-Variant Inference Engine für Hybrid-Prefix-Adapter:
- Lädt automatisch alle verfügbaren trainierten Adapter-Varianten
- Für jedes Exposé: lädt {eid}_M.pt aus variantenspezifischen Ordnern
- Lädt entsprechende Chapter-Scores aus {eid}_chapterscores.pt
- Kombiniert Prefix-Embeddings + Encoder-Daten + statisches Prompt
- Führt GPT-2-Generierung für jede Adapter-Variante durch
- Speichert generierte Bullet-Feedbacks in outputs/inference/{variant}/{eid}_generated.txt
- Unterstützt Batch-Processing für alle Exposés oder einzelne Exposé-Inference
- Bietet Adapter-Übersicht und Fortschrittsanzeige

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import torch
import logging
import argparse
from pathlib import Path

from utils.utils import load_config, setup_logging
from utils.tokenizer_utils import load_gpt2_tokenizer_and_model
from final_model.model_code.adapter_prefix_tuning import PrefixAdapter

class MultiVariantInferenceEngine:
    """
    Inference Engine für alle trainierten Adapter-Varianten.
    Lädt automatisch alle verfügbaren Adapter und generiert Feedback für jede Variante.
    """

    def __init__(self, config_path: str):
        # Setup
        self.config = load_config(config_path)
        setup_logging(self.config["logging"]["file"], self.config["logging"]["level"])
        self.logger = logging.getLogger(__name__)

        # Lade Tokenizer & GPT-2 (einmalig für alle Varianten)
        self.tokenizer, self.gpt2 = load_gpt2_tokenizer_and_model(self.config["models"]["gpt2_model"])
        self.gpt2.eval()

        # GPT-2 einfrieren für Inference
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # Hyperparameter
        self.hp = self.config["hyperparameters"]

        # Prompt vorab tokenisieren (einmalig)
        self._prepare_prompt()

        # Lade alle verfügbaren Adapter
        self.adapters = {}
        self._load_all_adapters()

        self.logger.info(f"Inference Engine initialisiert für {len(self.adapters)} Varianten")

    def _prepare_prompt(self):
        """Bereitet den Standard-Prompt vor"""
        prompt_str = (
            "Beispielformat:\n"
            "Kapitel 1 – Einleitung:\n"
            "• Die Einleitung stellt die Forschungsfrage klar heraus und motiviert das Thema.\n"
            "Kapitel 2 – Theoretischer Rahmen:\n"
            "• Der theoretische Rahmen ist präzise umrissen, könnte aber um zentrale Quellen erweitert werden.\n"
            "Jetzt erstellen Sie aus den oben übergebenen strukturellen Repräsentationen (Kapitel‑ und Satz‑Embeddings sowie Scores) ein formales Feedback in deutscher Höflichkeitsform als Bullet‑List:\n"
            "Feedback (Bullet‑List):\n"
        )

        tokens = self.tokenizer.encode(
            prompt_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hp["max_prompt_tokens"]
        )
        self.prompt_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).cuda()
        self.prompt_embs = self.gpt2.transformer.wte(self.prompt_ids)  # Embedding vorberechnen

    def _load_all_adapters(self):
        """Lädt alle verfügbaren trainierten Adapter"""
        adapter_dir = Path(self.config["paths"]["adapter_checkpoint_dir"])

        if not adapter_dir.exists():
            self.logger.error(f"Adapter-Checkpoint-Verzeichnis nicht gefunden: {adapter_dir}")
            return

        # Suche nach Variant-Unterordnern
        for variant_dir in adapter_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            variant_name = variant_dir.name

            # Suche nach dem besten Checkpoint
            checkpoint_files = list(variant_dir.glob("phi_best_*.pt"))

            if not checkpoint_files:
                self.logger.warning(f"Kein Checkpoint für Variante {variant_name} gefunden")
                continue

            # Neuesten Checkpoint nehmen (falls mehrere vorhanden)
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

            try:
                # Checkpoint laden
                checkpoint = torch.load(latest_checkpoint, map_location='cuda')

                # Adapter mit den gespeicherten Parametern erstellen
                adapter = PrefixAdapter(
                    input_dim=checkpoint['input_dim'],
                    hidden_dim=self.hp["prefix_hidden_dim"],
                    num_segments=self.hp["num_chapters"],
                    prefix_static_len=self.hp["prefix_static_len"],
                    prefix_dyn_len=self.hp["prefix_dyn_len"],
                    score_dim=self.hp.get("score_dim", 2),
                    dropout_p=0.0  # Kein Dropout bei Inference
                ).cuda()

                # State Dict laden
                adapter.load_state_dict(checkpoint['model_state_dict'])
                adapter.eval()

                # Adapter speichern
                self.adapters[variant_name] = {
                    'model': adapter,
                    'input_dim': checkpoint['input_dim'],
                    'max_total_len': checkpoint['max_total_len'],
                    'checkpoint_path': latest_checkpoint
                }

                self.logger.info(f"Adapter für {variant_name} geladen: {latest_checkpoint.name}")

            except Exception as e:
                self.logger.error(f"Fehler beim Laden des Adapters für {variant_name}: {e}")
                continue

    def _load_expose_data(self, expose_id: str, variant_name: str):
        """Lädt die Daten für ein spezifisches Exposé und eine Variante"""
        # Encoder-Daten aus der spezifischen Variante laden
        enc_path = Path(self.config["paths"]["encoder_output_dir"]) / variant_name / f"{expose_id}_M.pt"
        score_path = Path(self.config["paths"]["score_matrix_dir"]) / f"{expose_id}_chapterscores.pt"

        if not enc_path.exists():
            self.logger.warning(f"Encoder-Datei für {expose_id}/{variant_name} nicht gefunden: {enc_path}")
            return None

        if not score_path.exists():
            self.logger.warning(f"Score-Datei für {expose_id} nicht gefunden: {score_path}")
            return None

        try:
            # Lade Encoder-Daten
            enc_data = torch.load(enc_path, map_location='cuda')
            M = enc_data["M"].unsqueeze(0)  # (1, seq_len, feature_dim)
            mask = enc_data["mask"].unsqueeze(0)  # (1, seq_len)

            # Lade Chapter-Scores
            chap_scores = torch.load(score_path, map_location='cuda').unsqueeze(0)  # (1, K_MAX, 2)

            return {
                'M': M,
                'mask': mask,
                'chap_scores': chap_scores,
                'actual_seq_len': M.size(1)
            }

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Daten für {expose_id}/{variant_name}: {e}")
            return None

    def _create_segment_ids(self, seq_len: int):
        """Erstellt Segment-IDs für die gegebene Sequenzlänge"""
        S_MAX = self.hp["max_sentences_per_chapter"]
        K_MAX = self.hp["num_chapters"]

        idx = torch.arange(seq_len, device='cuda').unsqueeze(0)
        segment_ids = (idx // (S_MAX + 1)).clamp(max=K_MAX - 1)

        return segment_ids

    def _generate_feedback(self, adapter_info: dict, data: dict):
        """Generiert Feedback mit einem spezifischen Adapter"""
        adapter = adapter_info['model']
        max_total_len = adapter_info['max_total_len']

        # Segment-IDs erstellen
        segment_ids = self._create_segment_ids(data['actual_seq_len'])

        with torch.no_grad():
            # Adapter-Forward Pass
            H = adapter(data['M'], segment_ids=segment_ids, chap_scores=data['chap_scores'])

            # Input für GPT-2 zusammensetzen
            inputs = torch.cat([H, self.prompt_embs], dim=1)  # (1, prefix+seq+prompt, 768)

            # Attention Mask erstellen
            prefix_len = adapter.prefix_static_len + adapter.prefix_dyn_len
            prefix_mask = torch.ones((1, prefix_len), device='cuda')
            m_mask = data['mask']
            prompt_mask = torch.ones((1, self.prompt_ids.size(1)), device='cuda')

            attn_mask = torch.cat([prefix_mask, m_mask, prompt_mask], dim=1)

            # Padding auf maximale Länge, falls nötig
            current_len = inputs.size(1)
            if current_len < max_total_len:
                padding_len = max_total_len - current_len
                padding = torch.zeros((1, padding_len, inputs.size(2)), device='cuda')
                inputs = torch.cat([inputs, padding], dim=1)

                # Attention mask entsprechend erweitern
                attn_padding = torch.zeros((1, padding_len), device='cuda')
                attn_mask = torch.cat([attn_mask, attn_padding], dim=1)
            elif current_len > max_total_len:
                # Trimmen falls zu lang
                inputs = inputs[:, :max_total_len]
                attn_mask = attn_mask[:, :max_total_len]

            # Text generieren
            vocab_size = self.gpt2.config.vocab_size
            pad_token_id = self.tokenizer.eos_token_id

            # Sicherstellen dass pad_token_id gültig ist
            if pad_token_id >= vocab_size:
                pad_token_id = vocab_size - 1

            # Generierung
            with torch.no_grad():
                out_ids = self.gpt2.generate(
                    inputs_embeds=inputs,
                    attention_mask=attn_mask,
                    pad_token_id=pad_token_id,
                    max_length=inputs.size(1) + self.hp.get("max_bullet_tokens", 150),
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    do_sample=False
                )

            # Text dekodieren
            full_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

            # Nur den generierten Teil extrahieren
            if "Feedback (Bullet-List):" in full_text:
                feedback = full_text.split("Feedback (Bullet-List):")[-1].strip()
            else:
                feedback = full_text.strip()

            return feedback

    def run_inference_for_expose(self, expose_id: str):
        """Führt Inference für ein spezifisches Exposé mit allen verfügbaren Adaptern durch"""
        if not self.adapters:
            self.logger.error("Keine Adapter geladen!")
            return

        self.logger.info(f"Starte Inference für Exposé: {expose_id}")

        results = {}

        for variant_name, adapter_info in self.adapters.items():
            self.logger.info(f"Generiere Feedback mit Adapter: {variant_name}")

            # Daten für diese Variante laden
            data = self._load_expose_data(expose_id, variant_name)
            if data is None:
                self.logger.warning(f"Überspringe {variant_name} für {expose_id} (fehlende Daten)")
                continue

            try:
                # Feedback generieren
                feedback = self._generate_feedback(adapter_info, data)
                results[variant_name] = feedback

                # Feedback speichern
                self._save_feedback(expose_id, variant_name, feedback)

                self.logger.info(f"Feedback für {expose_id}/{variant_name} erfolgreich generiert")

            except Exception as e:
                self.logger.error(f"Fehler bei Feedback-Generierung für {expose_id}/{variant_name}: {e}")
                continue

        return results

    def _save_feedback(self, expose_id: str, variant_name: str, feedback: str):
        """Speichert das generierte Feedback in variantenspezifischen Ordnern"""
        # Ausgabe-Verzeichnis für diese Variante
        out_dir = Path(self.config["paths"]["inference_output_dir"]) / variant_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Feedback speichern
        output_file = out_dir / f"{expose_id}_generated.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(feedback)

    def run_inference_for_all_exposes(self):
        """Führt Inference für alle verfügbaren Exposés durch"""
        if not self.adapters:
            self.logger.error("Keine Adapter geladen!")
            return

        # Finde alle verfügbaren Exposé-IDs (aus der ersten verfügbaren Variante)
        first_variant = next(iter(self.adapters.keys()))
        encoder_variant_dir = Path(self.config["paths"]["encoder_output_dir"]) / first_variant

        if not encoder_variant_dir.exists():
            self.logger.error(f"Encoder-Verzeichnis nicht gefunden: {encoder_variant_dir}")
            return

        # Alle _M.pt Dateien finden
        m_files = list(encoder_variant_dir.glob("*_M.pt"))
        expose_ids = [f.stem.replace("_M", "") for f in m_files]

        if not expose_ids:
            self.logger.error("Keine Exposé-Dateien gefunden!")
            return

        self.logger.info(f"Starte Inference für {len(expose_ids)} Exposés mit {len(self.adapters)} Adaptern")

        total_combinations = len(expose_ids) * len(self.adapters)
        completed = 0

        for expose_id in expose_ids:
            self.logger.info(f"\nVerarbeite Exposé {expose_id} ({expose_ids.index(expose_id) + 1}/{len(expose_ids)})")

            results = self.run_inference_for_expose(expose_id)
            completed += len(results)

            progress = (completed / total_combinations) * 100
            self.logger.info(f"Fortschritt: {completed}/{total_combinations} ({progress:.1f}%)")

        self.logger.info(f"\nInference abgeschlossen! {completed} Feedbacks generiert.")

    def get_adapter_info(self):
        """Gibt Informationen über die geladenen Adapter zurück"""
        info = {}
        for variant_name, adapter_info in self.adapters.items():
            info[variant_name] = {
                'input_dim': adapter_info['input_dim'],
                'max_total_len': adapter_info['max_total_len'],
                'checkpoint': adapter_info['checkpoint_path'].name
            }
        return info


def main():
    parser = argparse.ArgumentParser(description="Multi-Variant Inference für Hybrid-Prefix-Adapter")
    parser.add_argument("--config", type=str, default="config2.json",
                        help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--expose_id", type=str, default=None,
                        help="Spezifisches Exposé-ID für Inference (optional)")
    parser.add_argument("--list_adapters", action="store_true",
                        help="Zeigt verfügbare Adapter an")
    args = parser.parse_args()

    # Engine initialisieren
    engine = MultiVariantInferenceEngine(args.config)

    if args.list_adapters:
        # Adapter-Informationen anzeigen
        adapter_info = engine.get_adapter_info()
        print(f"\nVerfügbare Adapter ({len(adapter_info)}):")
        print("=" * 60)
        for variant, info in adapter_info.items():
            print(f"Variante: {variant}")
            print(f"  Input-Dim: {info['input_dim']}")
            print(f"  Max-Total-Len: {info['max_total_len']}")
            print(f"  Checkpoint: {info['checkpoint']}")
            print()
        return

    if args.expose_id:
        # Inference für spezifisches Exposé
        engine.run_inference_for_expose(args.expose_id)
    else:
        # Inference für alle Exposés
        engine.run_inference_for_all_exposes()


if __name__ == "__main__":
    main()