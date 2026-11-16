# Reproducibility code and individual figures

Combined release: October 23, 2026. This joins the latest located computational reproducibility release (September 30 v2) and the latest clean figure code/data release (October 14).

## Contents

- `reproducibility_code/`: the complete original computational package, all 516 files unchanged, including its README, environment, configurations, source and checksums.
- `figures/`: 30 independent `draw.py` scripts and all 53 local plotting inputs, unchanged; plus two clearly marked Figure 1 source-unavailable folders.
- `FIGURE_INDEX.md` and `FIGURE_INDEX.csv`: all 32 panel entries, manuscript filenames and individual commands.
- `metadata/`: exact source archive identities and per-file provenance.

This release contains code, plotting data and instructions. Pre-rendered PDF/PNG files, manuscript TeX, raw national downloads and pretrained scientific models are not included.

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

## Reproduce the computational workflow

Open `reproducibility_code/README.md` and run its commands from inside `reproducibility_code/`. It provides the original environment, input placement, explicit run IDs/configurations and staged commands. Its references to a separate figure package now refer to the sibling `../figures/` folder. The original README is preserved byte-for-byte.

The computational code reconstructs the September 30 workflow from separately obtained inputs. It is not a newly implemented October manuscript pipeline. Later manuscript experiments and editorial changes have not been retrofitted into it. The figure scripts redraw frozen publication plotting extracts; this merge does not establish that a fresh computational run produces those exact extracts. Full national execution was not performed for this merge. The source README's download checks are dated September 30 and were not rechecked online here.

## Completeness and version notes

Figure 1(a,b) cannot be regenerated from raw data with the available releases: their original geographic data and plotting code are missing. The older full v6 package preserves artwork and artwork-preview scripts, which do not reconstruct editable geography. The two folders and the figure index state this explicitly.

The 32 v6 manuscript figure references were compared with the October 20 v8 source: 31 match byte-for-byte. The architecture PDF is a later export with identical whitespace-normalized text/equations but different spacing; the latest available editable generator is retained unchanged. Portable redraws are not universally byte- or pixel-identical to manuscript PDFs.

## Integrity

```text
python verify_combined_package.py
python reproducibility_code/scripts/verify_package.py
```

The first command checks every distributed member against the combined manifest. Additional generated files are reported but do not invalidate the preserved source files. The second checks the original computational package with its own unchanged manifest.
