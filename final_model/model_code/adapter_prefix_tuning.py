"""
apdapter_v3.py

- Adapter-Klasse mit Prefix-Adapter
- Training Prefix-Adapter für verschiedene Ablationsvarianten
- Kombiniert statische/dynamische Embeddings mit GPT-2 für Feedback-Generierung

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

from datetime import datetime
import torch
import logging
import argparse
from pathlib import Path

from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from utils.tokenizer_utils import load_gpt2_tokenizer_and_model
from utils.utils import load_config, setup_logging


class PrefixDataset(Dataset):
    """
    Lädt für jedes Exposé aus einer spezifischen Ablation-Variante:
      - M-Matrix und Mask (einheitlich gepaddet)
      - Kapitel-Scores (2 Werte: Kohärenz & Adäquanz)
      - Prompt-TokenIDs (einheitlich gepaddet)
      - Gold-Bullet-TokenIDs (einheitlich gepaddet)
      - Segment-IDs
    """

    def __init__(self, encoder_dir, score_dir, gold_dir, tokenizer, config, variant_name):
        self.encoder_dir = Path(encoder_dir) / variant_name
        self.score_dir = Path(score_dir)
        self.gold_dir = Path(gold_dir)
        self.tokenizer = tokenizer
        self.variant_name = variant_name

        # Maximale Längen aus Config (statisch)
        self.max_seq_len = config["hyperparameters"]["max_sequence_length"]
        self.max_prompt_len = config["hyperparameters"]["max_prompt_tokens"]
        self.max_gold_len = config["hyperparameters"]["max_gold_tokens"]
        self.S_MAX = config["hyperparameters"]["max_sentences_per_chapter"]
        self.K_MAX = config["hyperparameters"]["num_chapters"]

        # Prefix-Längen
        self.prefix_static_len = config["hyperparameters"]["prefix_static_len"]
        self.prefix_dyn_len = config["hyperparameters"]["prefix_dyn_len"]
        self.prefix_len = self.prefix_static_len + self.prefix_dyn_len

        # IDs laden
        self.ids = [p.stem.replace("_M", "")
                    for p in self.encoder_dir.glob("*_M.pt")
                    if (self.gold_dir / f"{p.stem.replace('_M', '')}_gold_bullets.txt").exists()
                    and (self.score_dir / f"{p.stem.replace('_M', '')}_chapterscores.pt").exists()]

        # Feature-Dimension validieren
        self.feature_dim = self._validate_feature_dim()

        # Logging für Dataset-Initialisierung
        logger = logging.getLogger(__name__)
        logger.info(f"[{self.variant_name}] Dataset initialisiert: {len(self.ids)} Samples, "
                    f"Feature-Dim: {self.feature_dim}, Max-Seq: {self.max_seq_len}")

    def _validate_feature_dim(self):
        """Validiert und bestimmt die konsistente Feature-Dimension"""
        if not self.ids:
            return 770  # Fallback

        logger = logging.getLogger(__name__)

        # Prüfe erstes verfügbares File
        for eid in self.ids[:3]:
            try:
                sample_data = torch.load(self.encoder_dir / f"{eid}_M.pt")
                if "M" in sample_data:
                    feature_dim = sample_data["M"].size(1)
                    logger.info(f"[{self.variant_name}] Feature-Dimension aus {eid}: {feature_dim}")
                    return feature_dim
            except Exception as e:
                logger.warning(f"[{self.variant_name}] Fehler beim Laden von {eid}: {e}")
                continue

        logger.warning(f"[{self.variant_name}] Keine gültigen Files gefunden, verwende Fallback 770")
        return 770

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        eid = self.ids[idx]
        logger = logging.getLogger(__name__)

        # 1) M-Matrix & Maske laden
        enc = torch.load(self.encoder_dir / f"{eid}_M.pt")
        M_orig = enc["M"]  # Original Shape: (N, feature_dim)
        mask_orig = enc["mask"]  # Original Shape: (N,)

        # 2) Kapitel-Scores laden
        chap_scores = torch.load(self.score_dir / f"{eid}_chapterscores.pt")

        # 3) Prompt tokenisieren
        prompt = (
            "Beispielformat:\n"
            "Kapitel 1 – Einleitung:\n"
            "• Die Einleitung stellt die Forschungsfrage klar heraus und motiviert das Thema.\n"
            "Kapitel 2 – Theoretischer Rahmen:\n"
            "• Der theoretische Rahmen ist präzise umrissen, könnte aber um zentrale Quellen erweitert werden.\n"
            "Jetzt erstellen Sie aus den oben übergebenen strukturellen Repräsentationen (Kapitel‑ und Satz‑Embeddings sowie Scores) ein formales Feedback in deutscher Höflichkeitsform als Bullet‑List:\n"
            "Feedback (Bullet‑List):\n"
        )
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False,
                                              truncation=True, max_length=self.max_prompt_len)

        # 4) Gold-Bullets tokenisieren
        with open(self.gold_dir / f"{eid}_gold_bullets.txt", "r", encoding="utf-8") as f:
            gold_text = f.read().strip()

        gold_tokens = self.tokenizer.encode(gold_text, add_special_tokens=False,
                                            truncation=True, max_length=self.max_gold_len)

        # 5) Padding auf einheitliche Größen
        # M-Matrix padden/truncate
        seq_len = min(M_orig.size(0), self.max_seq_len)
        M = torch.zeros(self.max_seq_len, self.feature_dim)
        mask = torch.zeros(self.max_seq_len, dtype=torch.long)

        M[:seq_len] = M_orig[:seq_len]
        mask[:seq_len] = mask_orig[:seq_len]

        # Segment-IDs erstellen (welches Kapitel jede Zeile repräsentiert)
        segment_ids = torch.arange(self.max_seq_len, dtype=torch.long)
        segment_ids = (segment_ids // (self.S_MAX + 1)).clamp(max=self.K_MAX - 1)

        # Prompt padden
        prompt_ids = torch.zeros(self.max_prompt_len, dtype=torch.long)
        prompt_ids[:len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long)

        # Gold padden
        gold_ids = torch.zeros(self.max_gold_len, dtype=torch.long)
        gold_ids[:len(gold_tokens)] = torch.tensor(gold_tokens, dtype=torch.long)

        # Truncation Logging (selektiv)
        if M_orig.size(0) > self.max_seq_len:
            logger.warning(f"[{self.variant_name}] {eid}: M truncated {M_orig.size(0)} -> {self.max_seq_len}")
        if len(prompt_tokens) >= self.max_prompt_len:
            logger.warning(f"[{self.variant_name}] {eid}: Prompt truncated to {self.max_prompt_len}")
        if len(gold_tokens) >= self.max_gold_len:
            logger.warning(f"[{self.variant_name}] {eid}: Gold truncated to {self.max_gold_len}")

        return {
            "M": M,
            "mask": mask,
            "chap_scores": chap_scores,
            "segment_ids": segment_ids,
            "prompt_ids": prompt_ids,
            "gold_ids": gold_ids,
            "actual_seq_len": seq_len,
            "actual_prompt_len": len(prompt_tokens),
            "actual_gold_len": len(gold_tokens)
        }


def collate_fn(batch):
    """Optimierte Collate-Funktion mit einheitlichen Tensor-Größen"""
    return {
        "M": torch.stack([b["M"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "chap_scores": torch.stack([b["chap_scores"] for b in batch], dim=0),
        "segment_ids": torch.stack([b["segment_ids"] for b in batch], dim=0),
        "prompt_ids": torch.stack([b["prompt_ids"] for b in batch], dim=0),
        "gold_ids": torch.stack([b["gold_ids"] for b in batch], dim=0),
        "actual_seq_lens": torch.tensor([b["actual_seq_len"] for b in batch], dtype=torch.long),
        "actual_prompt_lens": torch.tensor([b["actual_prompt_len"] for b in batch], dtype=torch.long),
        "actual_gold_lens": torch.tensor([b["actual_gold_len"] for b in batch], dtype=torch.long)
    }


class PrefixAdapter(nn.Module):
    """
    Hybrid-Prefix-Adapter mit statischen Dimensionen für alle Ablationsvarianten.
    """

    def __init__(self,
                 input_dim=770,
                 hidden_dim=768,
                 num_segments=8,
                 prefix_static_len=10,
                 prefix_dyn_len=2,
                 score_dim=2,
                 dropout_p=0.1
                 ):
        super().__init__()
        self.prefix_static_len = prefix_static_len
        self.prefix_dyn_len = prefix_dyn_len

        # Statischer Prefix
        self.P_stat = nn.Parameter(torch.randn(prefix_static_len, hidden_dim) * 0.02)

        # Dynamischer Prefix aus Scores
        self.W_score = nn.Linear(score_dim, prefix_dyn_len * hidden_dim)

        # Hauptprojektionen
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.segment_emb = nn.Embedding(num_segments, hidden_dim)

    def forward(self, x, segment_ids=None, chap_scores=None):
        B, L, D = x.size()  # (B, max_seq_len, input_dim)

        # Lineare Projektion
        H = self.linear(x.view(B * L, D)).view(B, L, -1)  # (B, L, 768)
        H = self.dropout(H)

        # Segment-Embeddings hinzufügen
        if segment_ids is not None:
            H = H + self.segment_emb(segment_ids)

        # Dynamischer Prefix aus Chapter-Scores
        if chap_scores is not None:
            pooled_scores = chap_scores.mean(dim=1)  # (B, 2)
            P_dyn = self.W_score(pooled_scores).view(B, self.prefix_dyn_len, -1)  # (B, 2, 768)
        else:
            P_dyn = torch.zeros((B, self.prefix_dyn_len, H.size(-1)), device=x.device)

        # Statischer Prefix expandieren
        P_stat = self.P_stat.unsqueeze(0).expand(B, -1, -1)  # (B, 10, 768)

        # Alles zusammensetzen
        result = torch.cat([P_stat, P_dyn, H], dim=1)  # (B, 12+L, 768)
        return result


def create_labels_and_masks(batch, adapter, device, max_total_len):
    """
    Erstellt einheitliche Labels und Attention-Masks mit fester Länge
    """
    B = batch["M"].size(0)
    prefix_len = adapter.prefix_static_len + adapter.prefix_dyn_len

    # Einheitliche Tensoren erstellen
    labels = torch.full((B, max_total_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_total_len), dtype=torch.long, device=device)

    for i in range(B):
        seq_len = batch["actual_seq_lens"][i].item()
        prompt_len = batch["actual_prompt_lens"][i].item()
        gold_len = batch["actual_gold_lens"][i].item()

        # Gesamtlänge für dieses Sample
        total_len = prefix_len + seq_len + prompt_len + gold_len

        # Attention mask setzen
        attention_mask[i, :total_len] = 1

        # Labels: nur Gold-Tokens am Ende
        gold_start = prefix_len + seq_len + prompt_len
        gold_end = gold_start + gold_len
        labels[i, gold_start:gold_end] = batch["gold_ids"][i, :gold_len]

    return labels, attention_mask


def train_adapter_for_variant(args, variant_name):
    """Trainiert einen Adapter für eine spezifische Ablationsvariante"""
    config = load_config(args.config)
    logger = logging.getLogger(__name__)

    logger.info(f"Training Adapter für Variante: {variant_name}")

    hp = config["hyperparameters"]
    tokenizer, gpt2 = load_gpt2_tokenizer_and_model(config["models"]["gpt2_model"])

    # Dataset für spezifische Variante
    dataset = PrefixDataset(
        encoder_dir=config["paths"]["encoder_output_dir"],
        score_dir=config["paths"]["score_matrix_dir"],
        gold_dir=config["paths"]["gold_bullet_dir"],
        tokenizer=tokenizer,
        config=config,
        variant_name=variant_name
    )

    if len(dataset) == 0:
        logger.warning(f"Keine Daten für Variante {variant_name} gefunden. Überspringe.")
        return

    # Maximale Gesamtlänge berechnen
    prefix_len = hp["prefix_static_len"] + hp["prefix_dyn_len"]
    max_total_len = prefix_len + hp["max_sequence_length"] + hp["max_prompt_tokens"] + hp["max_gold_tokens"]
    logger.info(f"[{variant_name}] Maximale Gesamtsequenzlänge: {max_total_len}")

    # Train/Val split
    val_size = int(0.2 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=hp["adapter_batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=hp["adapter_batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    # Adapter mit angepasster Input-Dimension
    adapter = PrefixAdapter(
        input_dim=dataset.feature_dim,
        hidden_dim=hp["prefix_hidden_dim"],
        num_segments=hp["num_chapters"],
        prefix_static_len=hp["prefix_static_len"],
        prefix_dyn_len=hp["prefix_dyn_len"],
        score_dim=hp.get("score_dim", 2),
        dropout_p=hp.get("adapter_dropout", 0.1)
    ).cuda()

    logger.info(f"[{variant_name}] Adapter initialisiert: Input-Dim={dataset.feature_dim}, "
                f"Max-Total-Len={max_total_len}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=hp["adapter_lr"],
        weight_decay=hp.get("adapter_weight_decay", 0.01)
    )

    # Scheduler
    total_steps = len(train_loader) * hp["adapter_epochs"]
    warmup_steps = int(hp.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # GPT-2 einfrieren
    gpt2.eval()
    for p in gpt2.parameters():
        p.requires_grad = False

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = hp["adapter_early_stopping_patience"]

    for epoch in range(hp["adapter_epochs"]):
        adapter.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Alle Tensoren auf GPU
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()

            # Adapter-Forward
            H_full = adapter(batch["M"], segment_ids=batch["segment_ids"],
                             chap_scores=batch["chap_scores"])

            # Prompt-Embeddings
            prompt_embs = gpt2.transformer.wte(batch["prompt_ids"])

            # Input zusammensetzen (einheitliche Länge)
            inputs = torch.cat([H_full, prompt_embs], dim=1)  # (B, prefix+seq+prompt, 768)

            # Labels und Masks erstellen
            labels, attn_mask = create_labels_and_masks(batch, adapter, inputs.device, max_total_len)

            # Inputs auf korrekte Länge padden/trimmen
            current_len = inputs.size(1)
            if current_len < max_total_len:
                # Padding hinzufügen
                padding = torch.zeros((inputs.size(0), max_total_len - current_len, inputs.size(2)),
                                      device=inputs.device)
                inputs = torch.cat([inputs, padding], dim=1)
            elif current_len > max_total_len:
                # Trimmen
                inputs = inputs[:, :max_total_len]

            # Debugging für erste Batches
            if batch_idx == 0 and epoch == 0:
                logger.info(f"[{variant_name}] Input shapes: inputs={inputs.shape}, "
                            f"attn_mask={attn_mask.shape}, labels={labels.shape}")

            # Forward pass
            loss = gpt2(inputs_embeds=inputs, attention_mask=attn_mask, labels=labels).loss
            loss.backward()

            # Gradient Clipping
            clip_grad_norm_(adapter.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        # Validation
        adapter.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # GPU Transfer
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda()

                H_full = adapter(batch["M"], segment_ids=batch["segment_ids"],
                                 chap_scores=batch["chap_scores"])
                prompt_embs = gpt2.transformer.wte(batch["prompt_ids"])
                inputs = torch.cat([H_full, prompt_embs], dim=1)

                labels, attn_mask = create_labels_and_masks(batch, adapter, inputs.device, max_total_len)

                # Padding/Trimming
                current_len = inputs.size(1)
                if current_len < max_total_len:
                    padding = torch.zeros((inputs.size(0), max_total_len - current_len, inputs.size(2)),
                                          device=inputs.device)
                    inputs = torch.cat([inputs, padding], dim=1)
                elif current_len > max_total_len:
                    inputs = inputs[:, :max_total_len]

                total_val_loss += gpt2(inputs_embeds=inputs, attention_mask=attn_mask, labels=labels).loss.item()

        avg_val = total_val_loss / len(val_loader)
        logger.info(f"[{variant_name}] Epoch {epoch + 1}/{hp['adapter_epochs']} – "
                    f"Train: {avg_train:.4f}, Val: {avg_val:.4f}")

        # Early Stopping & Checkpointing
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0

            ckpt_dir = Path(config["paths"]["adapter_checkpoint_dir"]) / variant_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            path = ckpt_dir / f"phi_best_{stamp}.pt"

            torch.save({
                'model_state_dict': adapter.state_dict(),
                'variant_name': variant_name,
                'input_dim': dataset.feature_dim,
                'max_total_len': max_total_len,
                'config': config
            }, path)
            logger.info(f"[{variant_name}] Checkpoint gespeichert: {path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"[{variant_name}] Early Stopping nach {epoch + 1} Epochen.")
                break

    logger.info(f"[{variant_name}] Training abgeschlossen. Best Val Loss: {best_val_loss:.4f}")


def train_all_adapters(args):
    """Trainiert Adapter für alle verfügbaren Ablationsvarianten"""
    config = load_config(args.config)
    setup_logging(config["logging"]["file"], config["logging"]["level"])
    logger = logging.getLogger(__name__)

    encoder_dir = Path(config["paths"]["encoder_output_dir"])
    variants = [d.name for d in encoder_dir.iterdir() if d.is_dir()]

    if not variants:
        logger.error("Keine Ablationsvarianten gefunden. Encoder zuerst ausführen!")
        return

    logger.info(f"Gefundene Ablationsvarianten: {variants}")

    for variant in variants:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starte Adapter-Training für: {variant}")
        logger.info(f"{'=' * 60}")

        try:
            train_adapter_for_variant(args, variant)
        except Exception as e:
            logger.error(f"Fehler beim Training für {variant}: {e}", exc_info=True)
            continue

    logger.info("\nAlle Adapter-Trainings abgeschlossen!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training der Adapter für alle Ablationsvarianten")
    parser.add_argument("--config", type=str, default="config2.json")
    parser.add_argument("--variant", type=str, default=None,
                        help="Spezifische Variante trainieren (optional)")
    args = parser.parse_args()

    if args.variant:
        config = load_config(args.config)
        setup_logging(config["logging"]["file"], config["logging"]["level"])
        train_adapter_for_variant(args, args.variant)
    else:
        train_all_adapters(args)