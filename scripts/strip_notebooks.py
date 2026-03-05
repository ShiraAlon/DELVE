from pathlib import Path
import nbformat


def strip_notebook_outputs(nb_path: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    changed = False

    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

    if changed:
        nbformat.write(nb, nb_path)
        print(f"Stripped: {nb_path}")
    else:
        print(f"No change: {nb_path}")


def main() -> None:
    notebooks_dir = Path(__file__).resolve().parents[1] / "notebooks"
    for nb_file in sorted(notebooks_dir.glob("*.ipynb")):
        strip_notebook_outputs(nb_file)


if __name__ == "__main__":
    main()
