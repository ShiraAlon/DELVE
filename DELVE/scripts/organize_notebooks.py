"""Apply the repository's reproducible, paper-output-only notebook conventions."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

BOOTSTRAP = """# Reproducible project paths and output locations
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "src").is_dir():
    raise FileNotFoundError("Run this notebook from inside the DELVE repository.")

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_utils import DATA_DIR, FIGURES_DIR, TABLES_DIR, configure_plots, ensure_output_dirs, save_figure

configure_plots()
ensure_output_dirs()
"""

CORE_FUNCTIONS = {
    "Kernel_matrix", "LG_K", "LG_RW", "LG_sym", "calc_differential_vec",
    "calc_differential_vec_cutoff", "calc_distance", "calc_sig_to_noise",
    "circ_convolution", "diffusion_map", "spectral_mapping",
}
ECG_FUNCTIONS = {
    "add_tolerance", "collapse_windows_to_centers", "const_lag", "const_lags",
    "coverage_PR_AUC", "lowpass_filter", "preprocess_ecg",
    "remove_baseline_median", "window_vector_to_signal",
}

def source(cell) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def remove_top_level_imports(text: str) -> tuple[str, list[ast.stmt]]:
    """Remove and return imports defined at a cell's top level."""
    tree = ast.parse(text)
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    lines = text.splitlines(keepends=True)
    for node in sorted(imports, key=lambda item: item.lineno, reverse=True):
        del lines[node.lineno - 1 : node.end_lineno]
    return "".join(lines).strip("\n"), imports


def normalized_imports(import_nodes: list[ast.stmt]) -> list[str]:
    """Deduplicate imports and merge imports from the same module."""
    plain_imports = []
    seen_plain = set()
    from_imports = {}
    from_order = []

    for node in import_nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = (alias.name, alias.asname)
                if key not in seen_plain:
                    seen_plain.add(key)
                    plain_imports.append(
                        f"import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else "")
                    )
            continue

        key = (node.level, node.module)
        if key not in from_imports:
            from_imports[key] = []
            from_order.append(key)
        seen_aliases = {(alias.name, alias.asname) for alias in from_imports[key]}
        for alias in node.names:
            if (alias.name, alias.asname) not in seen_aliases:
                from_imports[key].append(alias)
                seen_aliases.add((alias.name, alias.asname))

    result = plain_imports
    for level, module in from_order:
        prefix = "." * level + (module or "")
        names = ", ".join(
            alias.name + (f" as {alias.asname}" if alias.asname else "")
            for alias in from_imports[(level, module)]
        )
        result.append(f"from {prefix} import {names}")
    return result


def consolidate_notebook_imports(notebook) -> None:
    """Move every top-level import to the notebook's first code cell."""
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        return

    import_nodes = []
    for cell in code_cells:
        cleaned_source, cell_imports = remove_top_level_imports(source(cell))
        cell.source = cleaned_source
        import_nodes.extend(cell_imports)

    combined_source = "\n".join(source(cell) for cell in code_cells)
    used_names = {
        node.id
        for node in ast.walk(ast.parse(combined_source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    used_import_nodes = []
    for node in import_nodes:
        aliases = []
        for alias in node.names:
            if isinstance(node, ast.Import):
                binding = alias.asname or alias.name.split(".")[0]
            else:
                binding = alias.asname or alias.name
            if binding == "*" or binding in used_names:
                aliases.append(alias)
        if aliases:
            node.names = aliases
            used_import_nodes.append(node)

    imports = normalized_imports(used_import_nodes)
    local_prefixes = ("from project_utils import", "from functions import", "from ecg_functions import")
    external_imports = "\n".join(
        statement for statement in imports if not statement.startswith(local_prefixes)
    )
    local_imports = "\n".join(
        statement for statement in imports if statement.startswith(local_prefixes)
    )
    first_code = code_cells[0]
    setup_source = first_code.source.lstrip()
    if local_imports and "\nconfigure_plots()" in setup_source:
        setup_source = setup_source.replace(
            "\nconfigure_plots()",
            f"\n\n{local_imports}\n\nconfigure_plots()",
            1,
        )
    elif local_imports:
        setup_source = local_imports + "\n\n" + setup_source
    first_code.source = external_imports + "\n\n" + setup_source


def organize_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    obsolete_intermediate_cells = {
        'np.save("line_vs_cube_over_w.npy",acc_cube_over_w)',
        'np.save("triangle_res_for_different_l.npy",acc_triangle)',
        'np.save("triangle_running_results.npy",acc_triangle)',
        'acc_triangle = np.load("triangle_running_results.npy")',
    }
    notebook.cells = [
        cell
        for cell in notebook.cells
        if source(cell).strip() not in obsolete_intermediate_cells
    ]

    if path.name in {"ECG example.ipynb", "ECG real data.ipynb"}:
        cleaned_cells = []
        for cell in notebook.cells:
            text = source(cell)
            if cell.cell_type != "code":
                cleaned_cells.append(cell)
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                cleaned_cells.append(cell)
                continue
            definitions = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            only_definitions = tree.body and all(
                isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
                for node in tree.body
            )
            if only_definitions and definitions and definitions <= ECG_FUNCTIONS:
                continue
            cleaned_cells.append(cell)
        notebook.cells = cleaned_cells

    all_code = "\n".join(
        source(cell) for cell in notebook.cells if cell.cell_type == "code"
    )
    used_core_functions = sorted(
        name for name in CORE_FUNCTIONS if re.search(rf"\b{name}\b", all_code)
    )

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None
        text = source(cell)
        text = text.replace("plt.savefig(", "save_figure(")
        text = re.sub(
            r"(?P<object>\b\w+)\.savefig\(\s*(?P<quote>['\"])(?P<name>[^'\"]+)(?P=quote)",
            lambda match: (
                f'{match.group("object")}.savefig('
                f'FIGURES_DIR / "{match.group("name")}"'
            ),
            text,
        )
        if path.name == "Shared theta - robustness simulation.ipynb":
            text = text.replace(
                'fig.savefig(FIGURES_DIR / "robustness_simulation.pdf",',
                'save_figure("robustness_simulation.pdf",',
            )
        text = re.sub(
            r"(?P<object>\b\w+)\.save\(\s*(?P<quote>['\"])(?P<name>[^'\"]+\.(?:mp4|gif))(?P=quote)",
            lambda match: (
                f'{match.group("object")}.save('
                f'FIGURES_DIR / "{match.group("name")}"'
            ),
            text,
        )
        text = re.sub(
            r"(?<!TABLES_DIR / )(?P<quote>['\"])(?P<name>[^'\"]+\.tex)(?P=quote)",
            lambda match: f'TABLES_DIR / "{match.group("name")}"',
            text,
        )
        if "from functions import *" in text:
            text = text.replace(
                "from functions import *",
                "from functions import " + ", ".join(used_core_functions),
            )
        if 'exec(open("Functions.py").read())' in text:
            text = text.replace(
                'exec(open("Functions.py").read())',
                "from functions import " + ", ".join(used_core_functions),
            )
        if path.name == "ECG example.ipynb":
            text = text.replace(
                "DATA_CANDIDATES = [\n    PROJECT_ROOT,\n    Path('/Users/shiraalon/Documents/Thesis/Different examples for FKT comparation'),\n]",
                "DATA_CANDIDATES = [DATA_DIR, PROJECT_ROOT]",
            )
            if "s1, u1 = calc_differential_vec(L2, v1, 4)" in text:
                text = text.replace(
                    "s1, u1 = calc_differential_vec(L2, v1, 4)\n"
                    "s2, u2 = calc_differential_vec(L1, v2, 4)",
                    "tau1 = KneeLocator(np.arange(len(d1)), d1, curve=\"convex\", "
                    "direction=\"decreasing\", S=0.5).knee\n"
                    "tau2 = KneeLocator(np.arange(len(d2)), d2, curve=\"convex\", "
                    "direction=\"decreasing\", S=0.5).knee\n\n"
                    "s1, u1 = calc_differential_vec(L2, v1, tau1)\n"
                    "s2, u2 = calc_differential_vec(L1, v2, tau2)",
                )
        elif path.name == "Line VS. Rectangle -convergence.ipynb":
            text = text.replace(
                "candidates = [\n    PROJECT_ROOT,\n    Path('/Users/shiraalon/Documents/Thesis/Different examples for FKT comparation'),\n]",
                "candidates = [DATA_DIR, PROJECT_ROOT]",
            )
            text = text.replace(
                "np.save(PROJECT_ROOT / 'l2norm_rw.npy', rw_l_convergence)",
                "np.save(DATA_DIR / 'l2norm_rw.npy', rw_l_convergence)",
            )
            text = text.replace(
                "np.save(PROJECT_ROOT / 'l2norm_delve.npy', delve_w_convergence)",
                "np.save(DATA_DIR / 'l2norm_delve.npy', delve_w_convergence)",
            )
            text = text.replace(
                "def rectangle_converge(L: float, W: float, N: int, C_vec: np.ndarray, k: int = 5):",
                "def rectangle_converge(L: float, W: float, N: int, C_vec: np.ndarray):",
            )
            text = text.replace(
                "_, _, v1_sym = laplacian_sym(K1)\n"
                "        L2_sym, _, _ = laplacian_sym(K2)\n\n"
                "        _, u1 = calc_differential_vec(L2_sym, v1_sym, k)",
                "_, d1_sym, v1_sym = laplacian_sym(K1)\n"
                "        L2_sym, _, _ = laplacian_sym(K2)\n\n"
                "        tau1 = KneeLocator(\n"
                "            np.arange(len(d1_sym)), d1_sym,\n"
                "            curve=\"convex\", direction=\"decreasing\", S=0.5,\n"
                "        ).knee\n"
                "        _, u1 = calc_differential_vec(L2_sym, v1_sym, tau1)",
            )
            text = text.replace("                k=5,\n", "")
        elif path.name == "Yoda and Rabit.ipynb":
            text = text.replace(
                "DATA_CANDIDATES = [\n    PROJECT_ROOT / '3figures' / 'data',\n    Path('/Users/shiraalon/Documents/Thesis/Different examples for FKT comparation/3figures/data'),\n]",
                "DATA_CANDIDATES = [DATA_DIR / 'yoda-rabbit', DATA_DIR / '3figures' / 'data']",
            )
            if "# 5. Plot cumulative explained variance" in text:
                text = text.split("# ---------------------------------------------------------\n# 5. Plot cumulative explained variance", 1)[0].rstrip() + "\n"
        elif path.name == "ECG real data.ipynb" and "# 2) run FastICA" in text:
            text = text.split("import matplotlib.pyplot as plt", 1)[0].rstrip() + "\n"
        elif path.name == "ECG real data.ipynb" and "df_results_filtered =" in text:
            text = """df_results_filtered = df_results[
    (~df_results["method_vector"].str.startswith("DELVE"))
    | df_results["method_vector"].str.contains("_k20_")
].copy()

df_results_filtered
"""
        if path.name == "Line VS. Rectangle -convergence.ipynb":
            text = text.replace(
                "print(f'Loading precomputed arrays from:\n  {rw_path}\n  {delve_path}')",
                "print(f'Loading precomputed arrays from:\\n  {rw_path}\\n  {delve_path}')",
            )
            text = text.replace("textstr = '\n'.join((", "textstr = '\\n'.join((")
        if path.name == "ECG real data.ipynb":
            if text.strip() == "L1, d1, v1 = LG_sym(K1)\nL2, d2, v2 = LG_sym(K2)":
                text += (
                    "\n\ntau1 = KneeLocator(np.arange(len(d1)), d1, curve=\"convex\", "
                    "direction=\"decreasing\", S=0.5).knee\n"
                    "tau2 = KneeLocator(np.arange(len(d2)), d2, curve=\"convex\", "
                    "direction=\"decreasing\", S=0.5).knee"
                )
            text = text.replace(
                "calc_differential_vec(L2,v1,10)",
                "calc_differential_vec(L2, v1, tau1)",
            ).replace(
                "calc_differential_vec(L1,v2,10)",
                "calc_differential_vec(L1, v2, tau2)",
            )
            text = text.replace(
                ", tolerances=[50, 100, 150, 200], k=10,K=1500, fs=1000):",
                ", tolerances=[50, 100, 150, 200], K=1500, fs=1000):",
            )
            text = text.replace(
                "s1, u1 = calc_differential_vec(L2,v1,k)\n"
                "        s2, u2 = calc_differential_vec(L1,v2,k)",
                "tau1 = KneeLocator(np.arange(len(d1)), d1, curve=\"convex\", "
                "direction=\"decreasing\", S=0.5).knee\n"
                "        tau2 = KneeLocator(np.arange(len(d2)), d2, curve=\"convex\", "
                "direction=\"decreasing\", S=0.5).knee\n\n"
                "        s1, u1 = calc_differential_vec(L2, v1, tau1)\n"
                "        s2, u2 = calc_differential_vec(L1, v2, tau2)",
            )
            if "def ecg_save(" in text and "# Filter-size ablation" not in text:
                text = text.replace(
                    "    for record_id in ids:",
                    "    # Filter-size ablation: k1-k4 are intentionally fixed.\n"
                    "    for record_id in ids:",
                    1,
                )
            if "def ecg_save_diff(" in text:
                text = text.replace(
                    ", adaptive=5500, k1=15,k2=20,k3=25, fs=1000):",
                    ", adaptive=5500, fs=1000):",
                )
                text = text.replace(
                    "s1, u11 = calc_differential_vec(L2,v1,k1)\n"
                    "        s2, u12 = calc_differential_vec(L1,v2,k1)",
                    "tau1 = KneeLocator(np.arange(len(d1)), d1, curve=\"convex\", "
                    "direction=\"decreasing\", S=0.5).knee\n"
                    "        tau2 = KneeLocator(np.arange(len(d2)), d2, curve=\"convex\", "
                    "direction=\"decreasing\", S=0.5).knee\n\n"
                    "        s1, u11 = calc_differential_vec(L2, v1, tau1)\n"
                    "        s2, u12 = calc_differential_vec(L1, v2, tau2)",
                )
        if path.name == "dSprites Algorithm 2.ipynb":
            text = text.replace(
                'data_file = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"',
                'data_file = DATA_DIR / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"',
            )
            if "# Save\nnp.savez_compressed(" in text:
                text = text.split("# Save\nnp.savez_compressed(", 1)[0].rstrip() + "\n"
        if path.name == "Line VS cube.ipynb" and "3d differentials cube.pdf" in text:
            text = text.replace(
                "fig = plt.figure(figsize=(7,8))",
                "fig = plt.figure(figsize=(9, 8))",
            )
            text = text.replace(
                'fig = plt.figure(figsize=(10, 8), layout="constrained")',
                "fig = plt.figure(figsize=(9, 8))",
            )
            text = text.replace("labelpad=-10", "labelpad=0")
            text = text.replace("labelpad=-9", "labelpad=0")
            text = text.replace("labelpad=6", "labelpad=0")
            text = re.sub(
                r"\nplt\.subplots_adjust\(.*?\)\n",
                "\n",
                text,
                flags=re.DOTALL,
            )
            margin_block = (
                "plt.subplots_adjust(left=0.03, right=0.97, bottom=0.04, "
                "top=0.94, wspace=0.05, hspace=0.08)\n\n"
            )
            text = re.sub(
                r'\s*save_figure\("3d differentials cube\.pdf"[^\n]*\)',
                "\n\n"
                + margin_block
                + 'save_figure("3d differentials cube.pdf", '
                'bbox_inches="tight", pad_inches=0.4)',
                text,
            )
        if path.name == "Line VS cube.ipynb":
            text = text.replace(
                "calc_differential_vec(L2, v1, tau2)",
                "calc_differential_vec(L2, v1, tau1)",
            )
            text = text.replace(
                "calc_differential_vec(L1, v2, tau1)",
                "calc_differential_vec(L1, v2, tau2)",
            )
            text = text.replace(
                'calc_differential_vec(L2,v_V1,10,"yes")',
                'calc_differential_vec(L2, v_V1, tauV1, "yes")',
            )
        if path.name == "Torus - multimodal.ipynb" and "def torus_corr_k(" in text:
            if "# Filter-size ablation" not in text:
                text = "# Filter-size ablation: k is intentionally varied.\n" + text
        if path.name == "Yoda and Rabit.ipynb" and "K_values =" in text:
            if "# Filter-size ablation" not in text:
                text = "# Filter-size ablation: K_values are intentionally varied.\n" + text
        cell.source = text

    first_code = next(
        (cell for cell in notebook.cells if cell.cell_type == "code"), None
    )
    if first_code is None:
        first_code = nbformat.v4.new_code_cell(BOOTSTRAP)
        notebook.cells.insert(0, first_code)
    elif "from project_utils import" not in source(first_code):
        first_code.source = BOOTSTRAP + "\n" + source(first_code)

    if "ensure_output_dirs" not in first_code.source:
        first_code.source += "\nfrom project_utils import ensure_output_dirs\n"
        first_code.source = first_code.source.replace(
            "configure_plots()", "configure_plots()\nensure_output_dirs()", 1
        )

    if path.name in {"ECG example.ipynb", "ECG real data.ipynb"}:
        ecg_import = "from ecg_functions import " + ", ".join(sorted(ECG_FUNCTIONS))
        if ecg_import not in first_code.source:
            first_code.source += "\n" + ecg_import + "\n"

    all_code = "\n".join(
        source(cell) for cell in notebook.cells if cell.cell_type == "code"
    )
    required_project_names = [
        name
        for name in (
            "DATA_DIR",
            "FIGURES_DIR",
            "TABLES_DIR",
            "configure_plots",
            "ensure_output_dirs",
            "save_figure",
        )
        if re.search(rf"\b{name}\b", all_code)
    ]
    first_code.source += (
        "\nfrom project_utils import " + ", ".join(required_project_names) + "\n"
    )
    if "KneeLocator" in all_code:
        first_code.source += "\nfrom kneed import KneeLocator\n"

    consolidate_notebook_imports(notebook)

    notebook.metadata.setdefault("kernelspec", {})
    notebook.metadata["kernelspec"].update(
        {"display_name": "Python 3", "language": "python", "name": "python3"}
    )
    notebook.metadata.setdefault("language_info", {})["name"] = "python"
    nbformat.write(notebook, path)
    print(f"Organized: {path.name}")


def main() -> None:
    for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        organize_notebook(path)


if __name__ == "__main__":
    main()
