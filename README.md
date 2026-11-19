# Reproducibility code and individual figures

Combined release: October 23, 2026. This contains the latest clean figure code/data release (October 14).

## Contents

- `figures/`: 30 independent `draw.py` scripts and all 53 local plotting inputs, unchanged; plus two clearly marked Figure 1 source-unavailable folders.
- `FIGURE_INDEX.md` and `FIGURE_INDEX.csv`: all 32 panel entries, manuscript filenames and individual commands.
- `metadata/`: exact source archive identities and per-file provenance.

This release contains code, plotting data and instructions.

## Generate one figure

Extract the ZIP first. Use Python with NumPy, pandas and Matplotlib. The plotting dependency versions used in validation are in `figures/requirements.txt`; they are independent of the larger computational environment.

From the combined package folder, for example:

```text
python figures/figure04_d_flow_distributions/draw.py
```

Or open any runnable panel folder and run:

```text
python draw.py
```

Each script reads only its own `raw_data/` and writes `figure.pdf` and `figure.png` in that folder. PNG export is 600 dpi. Running a script again replaces these two generated files. Choose another panel using the index. There is no need to run the computational pipeline to redraw the saved publication figures.
