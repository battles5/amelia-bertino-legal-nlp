# AMELIA + BERTino — Legal Argument Mining

Exam project for the **Text Mining & NLP** module — Master in Data Science and Statistical Learning ([MD2SL](https://www.md2sl.unifi.it/)), University of Florence, A.Y. 2025–2026.

Binary classification of argumentative components — **premise** (`prem`) vs **conclusion** (`conc`) — on Italian VAT tax-court decisions, using the [AMELIA](https://huggingface.co/datasets/nlp-unibo/AMELIA) dataset and the [BERTino](https://huggingface.co/indigo-ai/BERTino) transformer model.

---

## Results

| Model | Val Macro‑F1 | Test Macro‑F1 | Test Accuracy |
|-------|:-------------------:|:-------------:|:-------------:|
| TF‑IDF + Logistic Regression | 0.7647 | 0.7484 | 91.75 % |
| **BERTino (fine‑tuned)** | **0.9065** | **0.9221** | **96.46 %** |

Fine-tuning BERTino improves Macro-F1 by **+17.4 points** over the baseline, confirming that contextual transformer representations capture semantic and syntactic signals that a bag-of-words model misses.

### Confusion matrix (test set)

| | Pred prem | Pred conc |
|---|:---:|:---:|
| **Gold prem** | 506 | 10 |
| **Gold conc** | 11 | 67 |

*(BERTino — 21 errors out of 594 samples)*

---

## Methodology

### 1. Dataset — AMELIA

AMELIA (*Argument Mining Evaluation on Legal documents in ItAlian*) is an annotated corpus of argumentative components extracted from Italian Tax Court decisions on VAT disputes. It contains 3 311 instances split into official partitions:

| Split | Samples |
|-------|-------:|
| Train | 2 108 |
| Validation | 609 |
| Test | 594 |

Each instance is a text segment labeled as `prem` (premise) or `conc` (conclusion). The distribution is imbalanced (~80 % premises).

### 2. Baseline — TF‑IDF + Logistic Regression

scikit-learn pipeline:
1. **Preprocessing:** whitespace normalization, lowercasing
2. **Vectorization:** TF-IDF with unigrams and bigrams, max 50 000 features
3. **Classifier:** Logistic Regression (solver `lbfgs`, max 1 000 iterations, seed 42)

### 3. Model — BERTino (fine-tuning)

[BERTino](https://huggingface.co/indigo-ai/BERTino) is a DistilBERT pre-trained on Italian corpora. Fine-tuning configuration:

| Hyperparameter | Value |
|---------------|--------|
| Epochs | 3 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay | 0.01 |
| Warmup | 10 % of steps |
| Max length | 256 tokens |
| Selection metric | Macro‑F1 |
| Seed | 42 |

Hardware: NVIDIA RTX 2000 Ada (8 GB VRAM). Training time: ~4 minutes.

### 4. Metric

**Macro-F1** — arithmetic mean of per-class F1 scores. Chosen for robustness to class imbalance: it penalizes models that ignore the minority class (`conc`).

---

## Quick start

```bash
# Clone and enter
git clone https://github.com/battles5/amelia-bertino-legal-nlp.git
cd amelia-bertino-legal-nlp

# Create virtualenv and install
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

```

### Main commands

```bash
python scripts/train_baseline.py     # Baseline TF-IDF + LR (~1 min, CPU)
python scripts/train_bert.py         # Fine-tuning BERTino (~4 min, GPU)
python scripts/eval.py               # Evaluation + plots + tables
```

---

## Project structure

```
amelia-bertino-legal-nlp/
├── src/amelia_experiment/      # Reusable Python modules
│   ├── config.py               # Constants and paths
│   ├── dataset.py              # AMELIA loading from HF
│   ├── preprocess.py           # Text and label normalization
│   ├── metrics.py              # Macro-F1 and classification report
│   ├── baseline.py             # TF-IDF + LR
│   ├── bertino.py              # BERTino fine-tuning and inference
│   ├── artifacts.py            # JSON/CSV/table export
│   └── plotting.py             # Confusion matrix
├── scripts/                    # CLI entry points
│   ├── train_baseline.py       # Baseline training
│   ├── train_bert.py           # BERTino fine-tuning
│   └── eval.py                 # Evaluation and reporting
├── tests/                      # Unit tests (pytest)
├── .github/workflows/ci.yml    # CI via GitHub Actions
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Reproducibility

- **Fixed seed:** 42 (Python, NumPy, PyTorch)
- **Official splits:** AMELIA uses predefined splits on Hugging Face
- **Versionable artifacts:** JSON/CSV metrics and confusion matrices in `results/`
- **CI:** ruff (lint + format) + pytest on every push via GitHub Actions

---

## Licenses

| Resource | License | Link |
|---------|---------|------|
| AMELIA | CC BY 4.0 | [HF datasets](https://huggingface.co/datasets/nlp-unibo/AMELIA) |
| BERTino | MIT | [HF models](https://huggingface.co/indigo-ai/BERTino) |

---

## References

- *AMELIA: A dataset for argument mining in decisions on Italian VAT* (2023). In Proceedings of the Workshop on Computational Approaches to Argument Mining. [nlp-unibo/AMELIA](https://huggingface.co/datasets/nlp-unibo/AMELIA)

- *BERTino: An Italian DistilBERT model* (2022). [indigo-ai/BERTino](https://huggingface.co/indigo-ai/BERTino)

- *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019). In Proceedings of NAACL-HLT 2019.

- *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* (2019). arXiv:1910.01108.

- *Scikit-learn: Machine Learning in Python* (2011). Journal of Machine Learning Research, 12, 2825–2830.

---

## Development

```bash
ruff check .              # Lint
ruff format .             # Format
pytest                    # Test
```

