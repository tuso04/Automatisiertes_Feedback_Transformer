"""
individual_test_model.py

Aufgabe: Vorhersage der Textkohärenz mittels vortrainiertem SkipFlow-Modell und GBERT-Tokenizer

- Unicode-Normalisierung und Bereinigung von Zeilenumbrüchen/Leerräumen
- Laden des GBERT-Tokenizers und SkipFlowCoherenceOnly-Modells
- Initialisierung mit gespeicherten Gewichten und GPU-Transfer
- Tokenisierung mit Truncation und Padding auf maximale Länge
- Forward-Pass durch das Modell ohne Gradientenberechnung
- Sigmoid-Transformation der Logits zu Kohärenz-Wahrscheinlichkeit

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""


import torch
import unicodedata
import re
from transformers import AutoTokenizer
from coherence_model.model.edit_skipflow import SkipFlowCoherenceOnly  # passe den Pfad zu deiner Modellklasse an

# ==== 1. Text-Vorverarbeitung ====
def preprocess_text(text: str) -> str:
    # Unicode-Normalisierung
    text = unicodedata.normalize("NFC", text)
    # Zeilenumbrüche entfernen
    text = text.replace("\r\n", " ").replace("\n", " ")
    # mehrfachen Leerraum auf ein Leerzeichen reduzieren
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==== 2. Laden von Tokenizer und Modell ====
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/pthn17/bachelor_tim/coherence_model/model/checkpoints/a_model_epoch30.pt"   # dein gespeichertes state_dict

tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
model = SkipFlowCoherenceOnly(
    # relevance_width=50,
    # tensor_slices=16,
    # hidden_dim=128,
    # max_pairs=100,
    # bert_model_name="deepset/gbert-base"
)

# sichere Variante: nur weights laden
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# ==== 3. Funktion zur Kohärenz-Vorhersage ====
def predict_coherence(text: str, max_len: int = 500) -> float:
    """
    Nimmt einen Rohtext, gibt Wahrscheinlichkeit [0..1] zurück,
    dass der Text kohärent ist.
    """
    # 1) Vorverarbeitung
    clean = preprocess_text(text)
    # 2) Tokenisierung + Padding
    enc = tokenizer(
        clean,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # 3) Forward-Durchlauf
    with torch.no_grad():
        logit = model(input_ids, attention_mask, return_matrix = True)  # [1]
        #print("Rohe Kohärenz-Scores:", logit.squeeze(0).cpu().numpy())
        prob  = torch.sigmoid(logit).item()

    return prob

# ==== 4. Beispiel-Aufruf ====
if __name__ == "__main__":
    beispiel_text = (
        "**Integration der Matrixorganisation:** Untersuchung der "
        "Implementierung einer Matrixorganisation in der FI, einschließlich"
        "der Identifikation von Herausforderungen und der Entwicklung von"
        "Strategien zur effektiven Einführung und nachhaltigen Verankerung"
        "dieser Organisationsstruktur.")
    p = predict_coherence(beispiel_text)
    print(f"Kohärenz-Wahrscheinlichkeit: {p*100:.2f}%")
