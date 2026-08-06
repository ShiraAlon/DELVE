"""Project paths and plotting defaults shared by DELVE notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing both ``src`` and ``notebooks``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the DELVE project root.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"


def ensure_output_dirs() -> None:
    """Create the standard generated-artifact directories."""
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)


def configure_plots() -> None:
    """Apply the typography used by the paper figures."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )


def save_figure(filename: str, **kwargs) -> Path:
    """Save the current figure in ``figures/`` and return its path."""
    ensure_output_dirs()
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, **kwargs)
    return output_path
