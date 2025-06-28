"""
adapter.py

Implementiert Adapter (phi) als einzigen linearen Layer.
ANGEPASST für encoder_variants.py Repräsentationen (full_model).
Training:
- Eingabe: M (variable_rows×770), mask (variable_rows), Prompt-TokenIDs
- Ziel: Gold-Bullet-Feedback (TokenIDs)
Speichert das trainierte phi_best.pt im adapter_checkpoint_dir.

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import os
import torch
import logging
import argparse

from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.tokenizer_utils import load_gpt2_tokenizer_and_model
from utils.utils import load_config, setup_logging
from pathlib import Path

class PrefixDataset(Dataset):
    """
    Läd:
    - {eid}_M.pt → M (variable_rows×770), mask (variable_rows)
    - gold_bullets.txt → Gold-TokenIDs
    - Prompt-String → Prompt-TokenIDs

    ANGEPASST: Unterstützt variable Matrizengrößen aus encoder_variants.py
    """

    def __init__(self, encoder_dir, gold_dir, tokenizer, config, variant_name="full_model"):
        self.encoder_dir = Path(encoder_dir) / variant_name  # ANGEPASST: Varianten-Unterordner
        self.gold_dir = Path(gold_dir)
        self.tokenizer = tokenizer
        self.max_prompt = config["hyperparameters"]["max_prompt_tokens"]
        self.max_bullets = config["hyperparameters"]["max_bullet_tokens"]
        self.config = config
        self.variant_name = variant_name  # ANGEPASST: Varianten-Name speichern

        self.ids = []
        for m_file in self.encoder_dir.glob("*_M.pt"):
            eid = m_file.stem.replace("_M", "")
            gold_path = self.gold_dir / f"{eid}_gold_bullets.txt"
            if gold_path.exists():
                self.ids.append(eid)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        eid = self.ids[idx]
        enc = torch.load(self.encoder_dir / f"{eid}_M.pt",  weights_only=False)
        M = enc["M"]                        # ANGEPASST: Variable Größe (rows×770)
        mask = enc["mask"]                  # ANGEPASST: Variable Größe (rows)

        # ANGEPASST: Segment-IDs basierend auf tatsächlicher Matrix-Struktur
        # Im full_model: [Kapitel1, Satz1_1, Satz1_2, ..., Kapitel2, Satz2_1, ...]
        # Einfache Heuristik: Jede 51. Zeile ist ein Kapitel-Start (1 Kapitel + max 50 Sätze)
        rows = M.size(0)
        segment_ids = torch.zeros(rows, dtype=torch.long)

        # Berechne Segment-IDs basierend auf Position
        chapter_size = self.config["hyperparameters"]["max_sentences_per_chapter"] + 1  # +1 für Kapitel-Repr.
        for i in range(rows):
            segment_ids[i] = min(i // chapter_size, self.config["hyperparameters"]["num_chapters"] - 1)

        # Gold-Bullets laden
        with open(self.gold_dir / f"{eid}_gold_bullets.txt", "r", encoding="utf-8") as f:
            gold_text = f.read().strip()
        gold_ids = self.tokenizer.encode(
            gold_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_bullets
        )
        gold_ids = torch.tensor(gold_ids, dtype=torch.long)

        # Prompt-TokenIDs (unverändert)
        prompt = (
            "Bitte verfassen Sie aus den oben übergebenen strukturellen Repräsentationen "
            "ein formal-sprachliches Feedback in deutscher Höflichkeitsform in Form von Stichpunkten. "
            "Halten Sie sich an die Kapitelgliederung und vermeiden Sie Wiederholungen.\n\n"
            "Feedback (Bullet-List): "
        )

        # Prompt mit Few-Shot Beispielen
        prompt = (
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

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt
        )
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)

        return {
            "M": M,                   # ANGEPASST: Variable FloatTensor(rows×770)
            "mask": mask,             # ANGEPASST: Variable LongTensor(rows)
            "prompt_ids": prompt_ids, # LongTensor(≤max_prompt)
            "gold_ids": gold_ids,      # LongTensor(≤max_bullet_tokens)
            "segment_ids": segment_ids  # ANGEPASST: Variable LongTensor(rows)
        }

def collate_fn(batch):
    """
    ANGEPASST: Pad-Funktion für variable Matrizengrößen:
    - M & mask: Pad auf maximale Zeilenzahl im Batch
    - prompt_ids und gold_ids pad auf max Länge im Batch
    """
    batch_size = len(batch)

    # ANGEPASST: Finde maximale Zeilenzahl für M und mask
    max_rows = max([item["M"].size(0) for item in batch])
    feature_dim = batch[0]["M"].size(1)  # Sollte 770 sein

    # ANGEPASST: Pad M und mask auf max_rows
    M_batch = torch.zeros((batch_size, max_rows, feature_dim))
    mask_batch = torch.zeros((batch_size, max_rows), dtype=torch.long)
    segment_ids_batch = torch.zeros((batch_size, max_rows), dtype=torch.long)

    for i, item in enumerate(batch):
        rows = item["M"].size(0)
        M_batch[i, :rows, :] = item["M"]
        mask_batch[i, :rows] = item["mask"]
        segment_ids_batch[i, :rows] = item["segment_ids"]

    # Prompt pad (unverändert)
    max_p_len = max([len(item["prompt_ids"]) for item in batch])
    prompt_ids_batch = torch.zeros((batch_size, max_p_len), dtype=torch.long)
    for i, item in enumerate(batch):
        l = len(item["prompt_ids"])
        prompt_ids_batch[i, :l] = item["prompt_ids"]

    # Gold pad (unverändert)
    max_g_len = max([len(item["gold_ids"]) for item in batch])
    gold_ids_batch = torch.zeros((batch_size, max_g_len), dtype=torch.long)
    for i, item in enumerate(batch):
        l = len(item["gold_ids"])
        gold_ids_batch[i, :l] = item["gold_ids"]

    return {
        "M": M_batch,                    # ANGEPASST: (B, max_rows, 770)
        "mask": mask_batch,              # ANGEPASST: (B, max_rows)
        "prompt_ids": prompt_ids_batch,
        "gold_ids": gold_ids_batch,
        "segment_ids": segment_ids_batch # ANGEPASST: (B, max_rows)
    }

class PrefixAdapter(nn.Module):
    """
    ANGEPASST: Lineares Modul 770 → 768 (korrigierte Input-Dimension).
    """

    def __init__(self, input_dim=770, hidden_dim=768, num_segments=8):  # ANGEPASST: input_dim=770
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Segment-Embeddings für K_MAX Kapitel
        self.segment_emb = nn.Embedding(num_segments, hidden_dim)

    def forward(self, x, segment_ids=None):
        """
        ANGEPASST: Unterstützt variable Eingabegrößen
        x: FloatTensor (B, variable_rows, 770)
        Rückgabe: FloatTensor (B, variable_rows, 768)
        """
        B, L, D = x.size()
        x_flat = x.view(B * L, D)          # (B*L, 770)
        h_flat = self.linear(x_flat)       # (B*L, 768)
        h = h_flat.view(B, L, 768)         # (B, L, 768)

        # Segment-Embedding addieren
        if segment_ids is not None:
            seg = self.segment_emb(segment_ids)  # (B,L,768)
            h = h + seg
        return h

def train_adapter(args):
    # Setup (unverändert)
    config = load_config(args.config)
    setup_logging(config["logging"]["file"], config["logging"]["level"])
    logger = logging.getLogger(__name__)

    tokenizer, gpt2 = load_gpt2_tokenizer_and_model(config["models"]["gpt2_model"])

    # ANGEPASST: Variant-Parameter hinzufügen
    variant_name = getattr(args, 'variant', 'full_model')
    dataset = PrefixDataset(
        encoder_dir=config["paths"]["encoder_output_dir"],
        gold_dir=config["paths"]["gold_bullet_dir"],
        tokenizer=tokenizer,
        config=config,
        variant_name=variant_name  # ANGEPASST: Varianten-Name übergeben
    )

    # Train/Val split 80/20 (unverändert)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=config["hyperparameters"]["adapter_batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config["hyperparameters"]["adapter_batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    # ANGEPASST: Adapter mit korrigierter Input-Dimension
    adapter = PrefixAdapter(
        input_dim=770,  # ANGEPASST: Explizit auf 770 gesetzt
        hidden_dim=config["hyperparameters"]["prefix_hidden_dim"],
        num_segments=config["hyperparameters"]["num_chapters"]
    ).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config["hyperparameters"]["adapter_lr"])

    # Friere GPT2-Parameter (unverändert)
    gpt2.eval()
    for p in gpt2.parameters():
        p.requires_grad = False

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(config["hyperparameters"]["adapter_epochs"]):
        adapter.train()
        total_train_loss = 0.0
        for batch in train_loader:
            M = batch["M"].cuda()                       # ANGEPASST: (B,variable_rows,770)
            mask = batch["mask"].cuda()                 # ANGEPASST: (B,variable_rows)
            prompt_ids = batch["prompt_ids"].cuda()     # (B, m_p)
            gold_ids = batch["gold_ids"].cuda()         # (B, m_g)
            segment_ids = batch["segment_ids"].cuda()   # ANGEPASST: (B,variable_rows)

            # 1) Adapter: phi(M) → (B,variable_rows,768)
            H = adapter(M, segment_ids=segment_ids)
            # 2) Prompt-Embedding
            prompt_embs = gpt2.transformer.wte(prompt_ids)  # (B, m_p, 768)
            # 3) Kombiniere Prefix + Prompt
            full_embs = torch.cat([H, prompt_embs], dim=1)  # (B, variable_rows+m_p, 768)
            # 4) Attention-Mask
            attn_mask = torch.cat([
                mask,
                torch.ones((mask.size(0), prompt_ids.size(1)), device=mask.device)
            ], dim=1)  # (B, variable_rows+m_p)

            B, L, _ = full_embs.size()  # B=Batch, L=Seq-Len (variable)
            device = gold_ids.device
            # 1) neues Label-Tensor full length erstellen und mit -100 füllen
            padded_labels = torch.full((B, L), -100,
                                       dtype=torch.long,
                                       device=device)
            # 2) die echten M Label-Token in den letzten M Slots platzieren
            M_tokens = gold_ids.size(1)
            padded_labels[:, -M_tokens:] = gold_ids

            outputs = gpt2(inputs_embeds=full_embs, attention_mask=attn_mask, labels=padded_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} – Train Loss: {avg_train_loss:.4f}")

        # Validierung (unverändert außer segment_ids)
        adapter.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                M = batch["M"].cuda()
                mask = batch["mask"].cuda()
                prompt_ids = batch["prompt_ids"].cuda()
                gold_ids = batch["gold_ids"].cuda()
                segment_ids = batch["segment_ids"].cuda()  # ANGEPASST

                H = adapter(M, segment_ids=segment_ids)  # ANGEPASST
                prompt_embs = gpt2.transformer.wte(prompt_ids)
                full_embs = torch.cat([H, prompt_embs], dim=1)
                attn_mask = torch.cat([
                    mask,
                    torch.ones((mask.size(0), prompt_ids.size(1)), device=mask.device)
                ], dim=1)

                B, L, _ = full_embs.size()
                device = gold_ids.device
                padded_labels = torch.full(
                    (B, L),
                    fill_value=-100,
                    dtype=torch.long,
                    device=device
                )
                M_tokens = gold_ids.size(1)
                padded_labels[:, -M_tokens:] = gold_ids

                outputs = gpt2(inputs_embeds=full_embs, attention_mask=attn_mask, labels=padded_labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1} – Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # ANGEPASST: Checkpoint mit Varianten-Name
            ckpt_dir = config["paths"]["adapter_checkpoint_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_name = f"phi_best_{variant_name}.pt"  # ANGEPASST
            torch.save(adapter.state_dict(), os.path.join(ckpt_dir, checkpoint_name))
            logger.info(f"Neuer Best-Val-Loss: {best_val_loss:.4f}. Checkpoint {checkpoint_name} gespeichert.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["hyperparameters"]["adapter_early_stopping_patience"]:
                logger.info("Early Stopping: Keine Verbesserung mehr im Validation Loss.")
                break

    logger.info("Adapter-Training abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training des Prefix-Adapters (phi)")
    parser.add_argument("--config", type=str, default="config.json", help="Pfad zu config.json")
    parser.add_argument("--variant", type=str, default="full_model",
                       choices=["full_model", "no_structure", "no_textfeatures", "minimal"],
                       help="Ablation-Variante für Training")  # ANGEPASST
    args = parser.parse_args()
    train_adapter(args)