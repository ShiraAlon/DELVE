# DELVE Simulation Experiments

This repository contains the simulation notebooks used for the DELVE paper experiments.

## Repository layout

- `notebooks/`: experiment notebooks
- `src/functions.py`: shared helper functions used by multiple notebooks
- `data/`: optional place for local input data (kept mostly out of git)
- `scripts/`: maintenance scripts for preparing notebooks before publishing

## Included experiments

- `Line VS. Rectangle -convergence.ipynb`
- `Line VS. Rectangle .ipynb`
- `Line VS cube.ipynb`
- `Torus - multimodal.ipynb`
- `Yoda and Rabit.ipynb`
- `Accelerometer VS gyroscope.ipynb`
- `ECG example.ipynb`
- `ECG real data.ipynb`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Prepare notebooks for GitHub

To reduce repository size and remove machine-specific outputs:

```bash
python scripts/strip_notebooks.py
```

This clears notebook cell outputs and execution counts in `notebooks/`.

## Upload to GitHub

```bash
git add .
git commit -m "Add DELVE simulation notebooks and project structure"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

## Notes

- Some notebooks may require local datasets that are not committed.
- If you have large files, use Git LFS for `.mat`, `.npy`, or media files.
