# Exposé Feedback Generation Pipeline

Dieses Repository enthält eine modulare Pipeline zur automatisierten Generierung und Evaluierung von Bullet-Feedback zu wissenschaftlichen Exposés. Die einzelnen Komponenten und Skripte sind wie folgt aufgebaut:
Sie sind im Ordner "model_code/" zu finden

---

## 1. `preprocessing.py`

- **Aufgabe**: Extrahiert aus DOCX-Exposés strukturierte Datensätze  
- **Wesentliche Schritte**  
  1. Konvertierung von `.docx` → Markdown (Pandoc)  
  2. Filtern und Gruppieren von Fließtext in Kapitelabschnitte  
  3. Satz-Splitting (NLTK/SpaCy)  
  4. Extraktion und Kontextualisierung von Word-Kommentaren (XML-Parsing)  
  5. Matching von Kommentartexten zu Sätzen und Dedup via SBERT  
  6. Ausgabe  
     - `preprocessed.jsonl`: Je Eintrag `(expose_id, kapitel, sentence_text, comment_text)`  
     - Pro Exposé: `*_gold_bullets.txt` mit kapitelweisen Feedback-Bullets  

---

## 2. `encoder.py`

- **Aufgabe**: Berechnet und speichert mehrstufige Einbettungen  
- **Wesentliche Schritte**  
  1. Einlesen von `preprocessed.jsonl` und Rekonstruktion aller Sätze (kommentiert + unkommentiert)  
  2. SBERT-Satz- und Long-Text-Kapitel-Embeddings  
  3. Kohärenz-Scores (GBert-Coherence) auf Kapitel- und Satzebene  
  4. Zero-Shot NLI-Scores (GBert-ZeroShot) mit Kapitelüberschrift als Label  
  5. Adäquanz-Scores (SBERT-Kosinus mit Exposé-Titel)  
  6. Zusammenführung aller Features in eine Matrix `M ∈ ℝ^{408×770}` und eine Attention-Maske  
  7. Speicherung von `*_M.pt` (inkl. `mask` und Titel-Embedding) 


## 2.b `encoder_ablation.py`

- **Aufgabe**: Modularer Encoder für systematische Ablation-Experimente mit 4 strategischen Feature-Kombinationen
- **Wesentliche Schritte**
  1. Definition von 4 Ablation-Varianten (full_model, no_structure, no_textfeatures, minimal)
  2. Komponentenweise Modell-Initialisierung je nach aktiver Konfiguration
  3. Adaptive Satz- und Kapitel-Encodierung basierend auf enabled_components
  4. Varianten-spezifische Matrix-Schemas und Dimensionsberechnung
  5. Batch-Verarbeitung aller Exposés pro Ablation-Variante
  6. Ausgabe: Separate Verzeichnisse mit *_M.pt (Matrix + Ablation-Metadaten)

---

## 3. `adapter.py`

- **Aufgabe**: Definition und Training des Prefix-Adapters  
- **Inhalte**  
  - **`PrefixDataset`**  
    - Lädt `M`-Matrizen, Gold-Bullet-Tokens und Prompt-Tokens  
    - Berechnet Kapitel-Segment-IDs  
  - **`PrefixAdapter`**  
    - Linearer Layer (770→768) und Kapitel-Segment-Embeddings  
  - **`train_adapter`**  
    - Prefix-Tuning: Gefrorenes GPT-2 + trainierbarer Adapter  
    - Optimizer: AdamW, Early-Stopping, Checkpointing (`phi_best.pt`)

## 3.b `adapter_prefix_tuning.py`

- **Aufgabe**: Training von Prefix-Adaptern für verschiedene Ablation-Varianten zur GPT-2-basierten Feedback-Generierung
- **Wesentliche Schritte**
  1. Laden von M-Matrizen, Kapitel-Scores und Gold-Bullets pro Ablation-Variante
  2. Einheitliches Padding aller Sequenzen auf maximale Längen (statische Dimensionen)
  3. Hybrid-Prefix-Adapter mit statischen/dynamischen Embeddings und Segment-Embeddings
  4. Training mit eingefrorenem GPT-2 und nur trainierbaren Adapter-Parametern
  5. Train/Val-Split mit Early Stopping und Gradient Clipping
  6. Ausgabe: Checkpoints pro Variante in adapter_checkpoint_dir/{variant_name}/

---

## 4. `inference.py`

- **Aufgabe**: Bullet-Feedback-Generierung für neue Exposés  
- **Wesentliche Schritte**  
  1. Laden von `*_M.pt` und `phi_best.pt`  
  2. Adapter-Forward + statischer Prompt (Few-Shot-Template)  
  3. Konkatenation von Prefix- und Prompt-Embeddings  
  4. GPT-2 Textgenerierung (Beam-Search, No-Repeat-Ngram)  
  5. Extraktion und Speicherung von `*_generated.txt`


## 4.b `inference_ablation.py`

- **Aufgabe**: Multi-Variant Inference Engine für automatische Feedback-Generierung mit verschiedenen trainierten Hybrid-Prefix-Adaptern
- **Wesentliche Schritte**
  1. Lädt alle verfügbaren trainierte Adapter-Varianten aus spezifischen Checkpoint-Verzeichnissen
  2. Bereitet GPT-2 Tokenizer und statischen Prompt für Feedback-Generierung vor
  3. Lädt variantenspezifische Encoder-Daten und Chapter-Scores für jedes Exposé
  4. Führt Adapter-Forward-Pass mit Prefix-Embeddings und Segment-IDs durch
  5. Generiert Feedback-Text via GPT-2 mit Beam-Search und Attention-Masking
  6. Speichert generierte Bullet-Point-Feedbacks in variantenspezifischen Output-Ordnern

---

## 5. `evaluate.py`

- **Aufgabe**: Automatisierte Evaluation generierten vs. Gold-Feedbacks  
- **Wesentliche Schritte**  
  1. Einlesen von `*_gold_bullets.txt` und `*_generated.txt`  
  2. Berechnung von Precision, Recall, F1 mittels BERTScore (deutsches Multilingual-BERT)  
  3. Aggregation der Scores pro Exposé in einer CSV (`evaluation_results.csv`)  

---

## 6. `utils/`

Enthält Hilfsfunktionen zum Konfigurations- und Logging-Handling sowie Tokenizer-Setup:
- **`utils/utils.py`**: Konfigurations­lade- und Logging-Setup, JSONL-IO  
- **`utils/tokenizer_utils.py`**: Laden von GPT-2-Tokenizer und Modell  

---


