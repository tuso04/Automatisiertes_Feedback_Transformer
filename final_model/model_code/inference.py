"""
inference.py

Generiert für ein oder mehrere Exposés Bullet-Feedback:
- Lädt {eid}_M.pt
- Lädt trainierten Prefix-Adapter phi_best.pt
- Kombiniert Prefix-Embeddings + statisches Prompt
- Führt GPT-2-Generierung durch
- Speichert generiertes Bullet-Feedback in outputs/inference/{eid}_generated.txt

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import os
import torch
import logging
import argparse

from pathlib import Path
from utils.utils import load_config, setup_logging
from utils.tokenizer_utils import load_gpt2_tokenizer_and_model
from final_model.model_code.adapter import PrefixAdapter

class InferenceEngine:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        setup_logging(self.config["logging"]["file"], self.config["logging"]["level"])
        self.logger = logging.getLogger(__name__)

        # Lade Tokenizer & GPT-2
        self.tokenizer, self.gpt2 = load_gpt2_tokenizer_and_model(self.config["models"]["gpt2_model"])
        # Lade Adapter phi
        adapter_path = os.path.join(self.config["paths"]["adapter_checkpoint_dir"], "phi_best.pt")

        adapter_state = torch.load(adapter_path)  # enthält 'linear.weight', 'linear.bias'
        # Key-Umschreibung: entferne das 'linear.'-Prefix
        new_state = {}
        for k, v in adapter_state.items():
            if k.startswith("linear."):
                new_key = k.replace("linear.", "")
            else:
                new_key = k
            new_state[new_key] = v

        self.adapter = PrefixAdapter(
            input_dim=self.config["hyperparameters"]["adapter_input_dim"],
            hidden_dim=self.config["hyperparameters"]["prefix_hidden_dim"],
            num_segments=self.config["hyperparameters"]["num_chapters"]
        ).cuda()

        # Lade komplettes state_dict, kein Remapping nötig
        adapter_state = torch.load(adapter_path)
        self.adapter.load_state_dict(adapter_state)
        self.adapter.eval()

        # Prompt-Template
        # self.prompt_str = (
        #     "Bitte verfassen Sie aus den oben übergebenen strukturellen Repräsentationen "
        #     "ein formal-sprachliches Feedback in deutscher Höflichkeitsform in Form von Stichpunkten. "
        #     "Halten Sie sich an die Kapitelgliederung und vermeiden Sie Wiederholungen.\n\n"
        #     "Feedback (Bullet-List): "
        # )

        # Prompt mit Few-Shot Beispielen
        self.prompt_str = (
            "Beispielformat:\n"
            "Motivation und Problemstellung:\n"
            "• Bitte keine Fragesätze in das Exposé. ...\n"
            "• Das Methodische Vorgehen ist noch nicht komplett. ...\n"
            "\n"
            "Forschungsfrage:\n"
            "• Die Forschungsfrage ist noch nicht klar genug. ...\n"
            "\n"
            "Nun verfassen Sie aus den oben übergebenen strukturellen Repräsentationen "
            "ein formales Feedback in deutscher Höflichkeitsform als Bullet-List:\n\n"
            "Feedback (Bullet-List): ")

        self.prompt_ids = self.tokenizer.encode(
            self.prompt_str, add_special_tokens=False,
            truncation=True, max_length=self.config["hyperparameters"]["max_prompt_tokens"]
        )
        self.prompt_ids = torch.tensor(self.prompt_ids, dtype=torch.long).unsqueeze(0).cuda()

    def run_for_expose(self, expose_id: str):
        """
        1. Lädt {expose_id}_M.pt
        2. Adapter-Forward
        3. Kombiniere mit Prompt
        4. GPT-2 generate
        5. Speichere {expose_id}_generated.txt
        """
        enc_path = os.path.join(self.config["paths"]["encoder_output_dir"], f"{expose_id}_M.pt")
        if not os.path.exists(enc_path):
            self.logger.error(f"Encoder-Output {enc_path} nicht gefunden.")
            return
        data = torch.load(enc_path)
        M = data["M"].unsqueeze(0).cuda()      # (1,408,770)
        mask = data["mask"].unsqueeze(0).cuda()# (1,408)

        # ➤ Segment-IDs berechnen (Kapitelindex = row // (S_MAX+1))
        S_MAX = self.config["hyperparameters"]["max_sentences_per_chapter"]
        K_MAX = self.config["hyperparameters"]["num_chapters"]
        # row_count = K_MAX*(S_MAX+1)  # 408
        seq_len = M.size(1)  # 408
        # Erzeuge Tensor [0,1,2,...,407], forme um auf (1,408)
        idx = torch.arange(seq_len, device=M.device).unsqueeze(0)  # (1,408)
        segment_ids = (idx // (S_MAX + 1)).clamp(max=K_MAX - 1)  # (1,408)

        # Adapter-Forward
        with torch.no_grad():
            H = self.adapter(M, segment_ids=segment_ids)  # (1,408,768)

        # Prompt-Embedding
        prompt_embs = self.gpt2.transformer.wte(self.prompt_ids)  # (1, m_p,768)

        # Kombiniere
        full_embs = torch.cat([H, prompt_embs], dim=1)  # (1, 408+m_p, 768)
        attn_mask = torch.cat([
            mask,
            torch.ones((1, self.prompt_ids.size(1)), device=mask.device)
        ], dim=1)  # (1, 408+m_p)

        # Generierung
        out_ids = self.gpt2.generate(
            inputs_embeds=full_embs,
            attention_mask=attn_mask,
            max_length=408 + self.prompt_ids.size(1) + self.config["hyperparameters"]["max_bullet_tokens"],
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        generated = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        parts = generated.split("Feedback (Bullet-List):")
        feedback_text = parts[-1].strip() if len(parts) > 1 else generated.strip()

        # Speichern
        out_dir = self.config["paths"]["inference_output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{expose_id}_generated.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(feedback_text)
        self.logger.info(f"Generiertes Bullet-Feedback für {expose_id} unter {out_path}")

def main(args):
    engine = InferenceEngine(args.config)
    if args.expose_id:
        engine.run_for_expose(args.expose_id)
    else:
        for m_file in Path(engine.config["paths"]["encoder_output_dir"]).glob("*_M.pt"):
            eid = m_file.stem.replace("_M", "")
            engine.run_for_expose(eid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference: Erzeuge Bullet-Feedback für Exposés")
    parser.add_argument("--config", type=str, default="config.json", help="Pfad zu config.json")
    parser.add_argument("--expose_id", type=str, default=None, help="Optional: Nur dieses Exposé ausführen")
    args = parser.parse_args()
    main(args)
