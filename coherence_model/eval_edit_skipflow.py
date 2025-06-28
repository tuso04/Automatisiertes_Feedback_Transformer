"""
eval_edit_skipflow.py

Aufgabe: Evaluiert ein trainiertes SkipFlow-Kohärenz-Modell auf Testdaten und berechnet Vorhersage-Accuracy
Wesentliche Schritte

- Lädt SkipFlow-Modell-Checkpoint und GBERT-Tokenizer für Textverarbeitung
- Erstellt TestEssayDataset aus JSONL-Datei mit Texten und Labels
- Tokenisiert Eingabetexte mit Padding und Truncation auf MAX_LEN
- Führt Batch-Inferenz mit Sigmoid-Aktivierung und binärer Klassifikation durch
- Berechnet Accuracy durch Vergleich mit Ground-Truth-Labels über Threshold
- Exportiert Vorhersagen mit Kohärenz-Wahrscheinlichkeiten in CSV-Format

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import json
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from model.edit_skipflow import SkipFlowCoherenceOnly   # Passe den Import-Pfad zu deiner Modellklasse an

# ==== 1. Konfiguration ====
MODEL_PATH    = "/home/pthn17/bachelor_tim/coherence_model/model/checkpoints/a_model_epoch30.pt"  # Pfad zu deinem gespeicherten state_dict
TEST_JSONL    = "/home/pthn17/bachelor_tim/coherence_model/data/coherence_de_test.jsonl"               # JSONL mit Feldern "text" und "label"
OUTPUT_CSV    = "results_eval_modela.csv"                   # Datei, in die die Vorhersagen geschrieben werden
BATCH_SIZE    = 8
MAX_LEN       = 400
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD     = 0.5                             # Schwelle zum Diskretisieren der Wahrscheinlichkeit

# ==== 2. Dataset für Inferenz mit Labels ====
class TestEssayDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=300):
        self.samples   = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                text  = obj['text'].replace('\n', ' ').strip()
                label = float(obj.get('label', 0))
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }

# ==== 3. Inferenz-Funktion mit Accuracy-Berechnung ====
def run_inference(model, dataloader, device='cpu'):
    model.eval()
    results = []
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)          # [B]

            logits = model(input_ids, attention_mask)           # [B]
            probs  = torch.sigmoid(logits)                      # [B]
            preds  = (probs >= THRESHOLD).float()               # [B]

            # Accuracy zählen
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            # Ergebnisse sammeln
            for text, p, pred, label in zip(batch['text'], probs.cpu().tolist(),
                                            preds.cpu().tolist(), labels.cpu().tolist()):
                results.append({
                    'text': text,
                    'coherence_prob': p,
                    'predicted_label': int(pred),
                    'true_label': int(label)
                })

    accuracy = correct / total if total > 0 else 0.0
    return results, accuracy

# ==== 4. Hauptroutine ====
if __name__ == "__main__":
    # 4.1 Tokenizer & Modell laden
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
    model = SkipFlowCoherenceOnly(
        relevance_width=50,
        tensor_slices=16,
        hidden_dim=128,
        max_pairs=100,
        bert_model_name="deepset/gbert-base"
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    # 4.2 Test-Daten vorbereiten
    test_dataset = TestEssayDataset(TEST_JSONL, tokenizer, max_len=MAX_LEN)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4.3 Inferenz durchführen
    predictions, accuracy = run_inference(model, test_loader, device=DEVICE)

    # 4.4 Ergebnisse in CSV speichern
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['coherence_prob', 'predicted_label', 'true_label', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow({
                'coherence_prob': f"{row['coherence_prob']:.4f}",
                'predicted_label': row['predicted_label'],
                'true_label': row['true_label'],
                'text': row['text']
            })

    print(f"Inferenz abgeschlossen. Ergebnisse gespeichert in '{OUTPUT_CSV}'.")
    print(f"Test-Accuracy: {accuracy*100:.2f}%")
