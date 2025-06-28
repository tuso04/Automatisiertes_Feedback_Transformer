import json
import os
import re
import time

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# Beispiel-Stopwortliste (kann nach Bedarf erweitert werden)
GERMAN_STOPWORDS = nltk.corpus.stopwords.words('german')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SkipFlowCoherenceOnly(nn.Module):
    def __init__(self,
                 relevance_width: int = 100,
                 tensor_slices: int = 16,
                 hidden_dim: int = 128,
                 max_pairs: int = 100,
                 bert_model_name: str = "deepset/gbert-base"):
        super().__init__()
        # 1) Vortrainiertes GBERT laden & einfrieren
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        bert_dim = self.bert.config.hidden_size

        # 2) LSTM auf BERT-Embeddings
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=300, batch_first=True)

        # 3) Tensor-Layer für Kohärenz
        self.delta = relevance_width
        self.k = tensor_slices
        self.M = nn.Parameter(torch.randn(self.k, 300, 300) * 0.01)
        self.V = nn.Linear(2 * 300, self.k)
        self.b_tensor = nn.Parameter(torch.zeros(self.k))
        self.u = nn.Parameter(torch.randn(self.k) * 0.01)

        # 4) Nur FC auf Kohärenz-Scores
        self.max_pairs = max_pairs
        self.fc_hidden = nn.Linear(max_pairs, hidden_dim)
        self.b_hidden = nn.Parameter(torch.zeros(hidden_dim))

        # 5) Finaler Logit-Layer
        self.fc_out = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.fc_out.bias, 0.5)

    def forward(self, input_ids, attention_mask, return_matrix=False):
        B, L = input_ids.size()
        delta = self.delta

        # 1) GBERT-Forward
        with torch.no_grad():
            emb = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask
                           ).last_hidden_state  # [B, L, bert_dim]

        # 2) LSTM
        h, _ = self.lstm(emb)  # [B, L, 300]

        mask = attention_mask.bool()  # [B, L]
        lengths = mask.sum(1)  # [B] Anzahl echter Tokens

        # 3) Kohärenz-Score-Matrix
        scores = []
        for start in range(0, L, delta):
            i, j = start, (start + delta) % L
            hi, hj = h[:, i, :], h[:, j, :]

            slice_scores = torch.einsum('bd,kde,be->bk', hi, self.M, hj)  # [B,k]
            linear_term = self.V(torch.cat([hi, hj], dim=-1))           # [B,k]
            t = torch.tanh(slice_scores + linear_term + self.b_tensor)           # [B,k]
            si = t @ self.u                                              # [B]
            scores.append(si.unsqueeze(1))

        coh_matrix = torch.cat(scores, dim=1)  # [B, n_pairs]
        #if return_matrix:
            #return coh_matrix
            #print(coh_matrix)

        # 4) Padding/Trimming auf max_pairs
        n = coh_matrix.size(1)
        if n < self.max_pairs:
            pad = torch.zeros(B, self.max_pairs - n, device=coh_matrix.device)
            coh_matrix = torch.cat([coh_matrix, pad], dim=1)
        else:
            coh_matrix = coh_matrix[:, :self.max_pairs]

        # 5) FC und Logit
        h_out = torch.tanh(self.fc_hidden(coh_matrix) + self.b_hidden)  # [B, hidden_dim]
        logit = self.fc_out(h_out).squeeze(1)  # [B]
        return logit

def train_coherence_only_and_save(model, dataloader, epochs=10, lr=2e-5,
                                  device='cuda', save_dir="checkpoints"):
    """
    Trainiert das Modell und speichert nach jeder Epoche die state_dict()-Gewichte ab.
    Gibt während des Trainings Fortschrittsmeldungen aus.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Device: {device}")
    model.to(device)

    # Nur trainierbare Parameter
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    losses = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (input_ids, attention_mask, label) in enumerate(dataloader, start=1):
            input_ids     = input_ids.to(device)
            attention_mask= attention_mask.to(device)
            label         = label.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Ausgabe alle 10 Batches
            if batch_idx % 10 == 0 or batch_idx == len(dataloader):
                avg_batch_loss = running_loss / batch_idx
                print(f" Epoch {epoch:2d} | Batch {batch_idx:3d}/{len(dataloader)} "
                      f"| Avg Loss: {avg_batch_loss:.4f}")

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"→ Epoche {epoch} abgeschlossen in {epoch_time:.1f}s | "
              f"mittl. Loss: {epoch_loss:.4f}")

        # Speichere weights
        save_path = os.path.join(save_dir, f"a10_model_epoch{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"   → Gewichte gespeichert: {save_path}\n")

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


class EssayCoherenceBertDataset(Dataset):
    def __init__(self, essays, labels, tokenizer, max_len=300):
        self.essays = essays
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        text = self.essays[idx]
        enc = self.tokenizer(text,
                             truncation=True,
                             padding='max_length',
                             max_length=self.max_len,
                             return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, label


def preprocess_text(text: str) -> str:
    """
    - Normalisiert Unicode (z. B. Umlaut-Komposition)
    - Entfernt Zeilenumbrüche und mehrfachen Leerraum
    """
    # 1) Unicode-Normalisierung (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2) Zeilenumbrüche → Leerzeichen
    text = text.replace("\r\n", " ").replace("\n", " ")

    # 3) Mehrfache Leerzeichen → einzelnes Leerzeichen
    text = re.sub(r"\s+", " ", text)

    # 4) führende/trailer Leerzeichen entfernen
    return text.strip()

def load_jsonl_dataset(path):
    essays = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            raw = obj['text']
            clean = preprocess_text(raw)
            essays.append(clean)
            labels.append(float(obj['label']))
    return essays, labels

if __name__ == "__main__":
    # Pfad zu deiner JSONL-Datei
    jsonl_path = "/home/pthn17/bachelor_tim/coherence_model/data/coherence_de_train.jsonl"

    # 3.1) JSONL einlesen
    essays, labels = load_jsonl_dataset(jsonl_path)

    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

    # Dataset und DataLoader anlegen
    dataset = EssayCoherenceBertDataset(
        essays=essays,
        labels=labels,
        tokenizer=tokenizer,
        max_len=400
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,  # je nach Hardware anpassen
        pin_memory=True
    )

    model = SkipFlowCoherenceOnly()
    train_coherence_only_and_save(model, dataloader, epochs=30, lr=1e-5, device=DEVICE)
