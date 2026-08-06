# DELVE

Official research code for the experiments presented in the DELVE paper.

DELVE identifies modality-specific structure in paired multimodal observations
using graph-based spectral operators. This repository contains the simulation and
data-analysis notebooks used to generate the paper’s numerical results, figures,
and tables.

## Repository structure

```text
DELVE/
├── notebooks/       Experiment notebooks
├── src/             Shared DELVE and ECG functions
├── data/            Input data and precomputed convergence arrays
├── figures/         Figures generated for the paper
├── tables/          LaTeX tables generated for the paper
├── scripts/         Notebook preparation and validation utilities
└── requirements.txt Python dependencies
```

The principal implementation is in [`src/functions.py`](src/functions.py), which
contains the kernel, graph-Laplacian, and differential-vector routines used across
the experiments. ECG-specific preprocessing and evaluation functions are collected
in [`src/ecg_functions.py`](src/ecg_functions.py).

## Experiments

| Experiment | Notebook | Data |
|---|---|---|
| Line vs. rectangle | [`Line VS. Rectangle .ipynb`](notebooks/Line%20VS.%20Rectangle%20.ipynb) | Simulated |
| Line vs. rectangle convergence | [`Line VS. Rectangle -convergence.ipynb`](notebooks/Line%20VS.%20Rectangle%20-convergence.ipynb) | Simulated / precomputed |
| Line vs. triangle | [`Line VS Triangle.ipynb`](notebooks/Line%20VS%20Triangle.ipynb) | Simulated |
| Line vs. cube | [`Line VS cube.ipynb`](notebooks/Line%20VS%20cube.ipynb) | Simulated |
| Multimodal torus | [`Torus - multimodal.ipynb`](notebooks/Torus%20-%20multimodal.ipynb) | Simulated |
| Nonlinear entangled modalities | [`Nonlinear Entangled Simulation.ipynb`](notebooks/Nonlinear%20Entangled%20Simulation.ipynb) | Simulated |
| Shared-variable robustness | [`Shared theta - robustness simulation.ipynb`](notebooks/Shared%20theta%20-%20robustness%20simulation.ipynb) | Simulated |
| Yoda and rabbit images | [`Yoda and Rabit.ipynb`](notebooks/Yoda%20and%20Rabit.ipynb) | External image data |
| Accelerometer and gyroscope | [`Accelerometer VS gyroscope.ipynb`](notebooks/Accelerometer%20VS%20gyroscope.ipynb) | OpenML HAR |
| Synthetic ECG | [`ECG example.ipynb`](notebooks/ECG%20example.ipynb) | External `.mat` files |
| Real ECG | [`ECG real data.ipynb`](notebooks/ECG%20real%20data.ipynb) | External ECG records |
| dSprites | [`dSprites Algorithm 2.ipynb`](notebooks/dSprites%20Algorithm%202.ipynb) | dSprites |

Some simulation notebooks include repeated trials or parameter-ablation studies and
may require substantial memory and computation time.

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/ShiraAlon/DELVE.git
cd DELVE

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Jupyter from the repository root:

```bash
jupyter lab
```

The notebooks locate the repository root automatically and use fixed random seeds
for stochastic experiments. Notebook outputs and execution counters are cleared in
the repository; selected paper artifacts are stored separately in `figures/` and
`tables/`.

## Data

The synthetic experiments generate their observations within the corresponding
notebook. The convergence notebook can load the included arrays
`data/l2norm_rw.npy` and `data/l2norm_delve.npy` instead of repeating the full
simulation.

The following external datasets are not distributed with the repository:

- **Synthetic ECG example:** place `fECG.mat` and `mECG.mat` in `data/`.
- **Yoda and rabbit images:** place the ordered image sequences in
  `data/yoda-rabbit/` or `data/3figures/data/`.
- **Real ECG:** place the signal and annotation files in the directory specified in
  the real-ECG notebook.
- **Human Activity Recognition:** downloaded from OpenML by the accelerometer and
  gyroscope notebook; an internet connection is required.
- **dSprites:** downloaded automatically when absent and stored locally in `data/`.

Large datasets and intermediate numerical arrays are excluded from version control.

## Reproducing paper outputs

Run the notebook corresponding to the desired experiment using **Restart Kernel and
Run All**. Selected outputs are written to:

- `figures/` for PDF, image, and animation files;
- `tables/` for LaTeX tables;
- `data/` for reusable numerical inputs or precomputed results.

The filter size used by `calc_differential_vec` is selected from the knee of the
relevant eigenspectrum. Fixed filter sizes occur only in notebooks that explicitly
perform filter-size ablations.

To check the repository notebooks before reproducing or publishing results, run:

```bash
python scripts/validate_notebooks.py
```

This checks notebook syntax, saved-output conventions, project-relative paths,
shared imports, and knee-based filter selection. Maintainers can normalize notebook
metadata and imports with:

```bash
python scripts/organize_notebooks.py
```

## Citation

If you use this repository, please cite the DELVE paper associated with this code (<https://arxiv.org/pdf/2402.18741>)
and include the repository URL:

> Shira Yoffe, *DELVE*, research code, GitHub.
> <https://github.com/ShiraAlon/DELVE>

## Contact

Questions about the code or experiments can be submitted through the repository’s
[GitHub issue tracker](https://github.com/ShiraAlon/DELVE/issues).
