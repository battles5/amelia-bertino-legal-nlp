# AMELIA + BERTino — Legal Argument Mining

Progetto d'esame per il corso di **Text Mining & NLP** (Università di Firenze, A.A. 2025–2026).

Classificazione binaria di componenti argomentative — **premessa** (`prem`) vs **conclusione** (`conc`) — su decisioni italiane in materia di IVA, usando il dataset [AMELIA](https://huggingface.co/datasets/nlp-unibo/AMELIA) e il modello transformer [BERTino](https://huggingface.co/indigo-ai/BERTino).

---

## Risultati

| Modello | Validation Macro‑F1 | Test Macro‑F1 | Test Accuracy |
|---------|:-------------------:|:-------------:|:-------------:|
| TF‑IDF + Logistic Regression | 0.7647 | 0.7484 | 91.75 % |
| **BERTino (fine‑tuned)** | **0.9065** | **0.9221** | **96.46 %** |

Il fine‑tuning di BERTino migliora la Macro‑F1 di **+17.4 punti** rispetto alla baseline, confermando che la rappresentazione contestuale del transformer cattura segnali semantici e sintattici che il modello bag‑of‑words non coglie.

### Confusion matrix (test set)

| | Pred prem | Pred conc |
|---|:---:|:---:|
| **Gold prem** | 506 | 10 |
| **Gold conc** | 11 | 67 |

*(BERTino — 21 errori su 594 esempi)*

---

## Metodologia

### 1. Dataset — AMELIA

AMELIA (*Argument Mining in dEcisioni su IVA*) è un corpus annotato di componenti argomentative estratte da decisioni italiane della Commissione Tributaria in materia di IVA (Lippi et al., 2023). Contiene 3 311 istanze suddivise in split ufficiali:

| Split | Esempi |
|-------|-------:|
| Train | 2 108 |
| Validation | 609 |
| Test | 594 |

Ogni istanza è un segmento testuale annotato come `prem` (premessa) o `conc` (conclusione). La distribuzione è sbilanciata (~80 % premesse).

### 2. Baseline — TF‑IDF + Logistic Regression

Pipeline scikit‑learn:
1. **Preprocessing:** normalizzazione whitespace, lowercasing
2. **Vettorizzazione:** TF‑IDF con unigrammi e bigrammi, max 50 000 feature
3. **Classificatore:** Logistic Regression (solver `lbfgs`, max 1 000 iterazioni, seed 42)

### 3. Modello — BERTino (fine‑tuning)

[BERTino](https://huggingface.co/indigo-ai/BERTino) è un DistilBERT pre‑addestrato su un corpus italiano (Muffo et al., 2022). Configurazione fine‑tuning:

| Iperparametro | Valore |
|---------------|--------|
| Epoche | 3 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay | 0.01 |
| Warmup | 10 % degli step |
| Max length | 256 token |
| Metrica di selezione | Macro‑F1 |
| Seed | 42 |

Hardware: NVIDIA RTX 2000 Ada (8 GB VRAM). Tempo di training: ~4 minuti.

### 4. Metrica

**Macro‑F1** — media aritmetica delle F1 per ciascuna classe. Scelta perché robusta allo sbilanciamento: penalizza modelli che ignorano la classe minoritaria (`conc`).

---

## Quick start

```bash
# Clona ed entra
git clone https://github.com/battles5/amelia-bertino-legal-nlp.git
cd amelia-bertino-legal-nlp

# Crea virtualenv e installa
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

# Smoke test
python scripts/smoke_test.py
```

### Comandi principali

```bash
python train_baseline.py            # Baseline TF-IDF + LR (~1 min, CPU)
python train_bert.py                # Fine-tuning BERTino (~4 min, GPU)
python eval.py                      # Valutazione + plot + tabelle
python demo.py --model bertino      # Demo su 5 esempi dal test set
```

---

## Struttura del progetto

```
amelia-bertino-legal-nlp/
├── src/amelia_experiment/      # Moduli Python riusabili
│   ├── config.py               # Costanti e path
│   ├── dataset.py              # Caricamento AMELIA da HF
│   ├── preprocess.py           # Normalizzazione testo e label
│   ├── metrics.py              # Macro-F1 e classification report
│   ├── baseline.py             # TF-IDF + LR
│   ├── bertino.py              # Fine-tuning e inferenza BERTino
│   ├── artifacts.py            # Salvataggio JSON/CSV/tabelle
│   └── plotting.py             # Confusion matrix
├── tests/                      # Unit test (pytest)
├── scripts/smoke_test.py       # Smoke test per CI
├── train_baseline.py           # CLI: training baseline
├── train_bert.py               # CLI: fine-tuning BERTino
├── eval.py                     # CLI: valutazione e report
├── demo.py                     # CLI: demo interattiva
├── .github/workflows/ci.yml    # CI GitHub Actions
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Riproducibilità

- **Seed fisso:** 42 (Python, NumPy, PyTorch)
- **Split ufficiali:** AMELIA usa split predefiniti su Hugging Face
- **Artefatti versionabili:** metriche JSON/CSV e confusion matrix in `results/`
- **CI:** ruff (lint + format) + pytest su ogni push via GitHub Actions

---

## Licenze

| Risorsa | Licenza | Link |
|---------|---------|------|
| AMELIA | CC BY 4.0 | [HF datasets](https://huggingface.co/datasets/nlp-unibo/AMELIA) |
| BERTino | MIT | [HF models](https://huggingface.co/indigo-ai/BERTino) |

---

## Riferimenti

- Lippi, M., Lagioia, F., Contissa, G., Sartor, G., & Torroni, P. (2023). *AMELIA: A dataset for argument mining in decisions on Italian VAT*. In Proceedings of the Workshop on Computational Approaches to Argument Mining. [nlp-unibo/AMELIA](https://huggingface.co/datasets/nlp-unibo/AMELIA)

- Muffo, M., Ferrario, A., & Kluzer, S. (2022). *BERTino: An Italian DistilBERT model*. [indigo-ai/BERTino](https://huggingface.co/indigo-ai/BERTino)

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. In Proceedings of NAACL-HLT 2019.

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108.

- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

---

## Sviluppo

```bash
ruff check .              # Lint
ruff format .             # Format
pytest                    # Test
python scripts/smoke_test.py  # Smoke test
```

