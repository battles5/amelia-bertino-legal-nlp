# slides/

Template Beamer per la presentazione (~10 slide).

## Compilazione

Richiede una distribuzione LaTeX installata (es. TeX Live, MiKTeX).

```bash
cd slides
latexmk -pdf main.tex
```

Oppure:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Dipendenze

Le slide importano automaticamente:
- `../results/tables/results_table.tex` — tabella risultati (generata da `eval.py`)
- `../results/tables/examples.tex` — snippet 5 esempi (generato da `demo.py`)
- `../results/plots/cm_*.png` — confusion matrix (generate da `eval.py`)

**Prima di compilare**, assicurarsi che questi file esistano (eseguire `eval.py` e `demo.py`).
