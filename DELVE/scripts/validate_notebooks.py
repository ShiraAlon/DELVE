"""Check DELVE notebooks for reproducibility and output-policy violations."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_PATTERN = re.compile(
    r"(?:plt\.(?!cm\.)|sns\.|\.plot\s*\(|\.hist\s*\(|\.scatter\s*\(|\.imshow\s*\()"
)
SAVE_PATTERN = re.compile(r"(?:save_figure|savefig|\.save\s*\()")
ABSOLUTE_PATH_PATTERN = re.compile(r"['\"]/(?:Users|home|mnt|Volumes)/")
UNSCOPED_FIGURE_SAVE_PATTERN = re.compile(
    r"\b\w+\.(?:savefig|save)\(\s*['\"][^'\"]+\.(?:pdf|png|jpg|jpeg|svg|mp4|gif)['\"]"
)


def validate(path: Path) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    problems = []
    all_code = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code = cell.source
        all_code.append(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            problems.append(f"cell {index}: syntax error: {error.msg}")
            continue
        is_filter_ablation = "# Filter-size ablation:" in code
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = getattr(node.func, "id", None)
            if function_name != "calc_differential_vec" or len(node.args) < 3:
                continue
            filter_argument = node.args[2]
            uses_knee = isinstance(filter_argument, ast.Name) and filter_argument.id.lower().startswith("tau")
            if not uses_knee and not is_filter_ablation:
                problems.append(
                    f"cell {index}: calc_differential_vec filter size is not knee-derived"
                )
            if uses_knee and isinstance(node.args[1], ast.Name):
                vector_stem = re.sub(
                    r"[^a-z0-9]", "", node.args[1].id.lower()[1:]
                ).removesuffix("sym")
                knee_stem = re.sub(
                    r"[^a-z0-9]", "", filter_argument.id.lower()[3:]
                )
                if vector_stem != knee_stem:
                    problems.append(
                        f"cell {index}: knee {filter_argument.id} does not match "
                        f"basis {node.args[1].id}"
                    )
        if cell.outputs or cell.execution_count is not None:
            problems.append(f"cell {index}: committed output or execution count")
        if ABSOLUTE_PATH_PATTERN.search(code):
            problems.append(f"cell {index}: machine-specific absolute path")
        if UNSCOPED_FIGURE_SAVE_PATTERN.search(code):
            problems.append(f"cell {index}: figure save does not use FIGURES_DIR")
        if PLOT_PATTERN.search(code) and not SAVE_PATTERN.search(code):
            executable = [
                node
                for node in tree.body
                if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            ]
            if executable and "rcParams" not in code:
                problems.append(f"cell {index}: plot is not saved")

    if "from project_utils import" not in "\n".join(all_code):
        problems.append("missing shared project-path setup")
    tree = ast.parse("\n".join(all_code))
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "project_utils"
        for alias in node.names
    }
    for name in ("DATA_DIR", "FIGURES_DIR", "TABLES_DIR"):
        if name in loaded_names and name not in imported_names:
            problems.append(f"uses {name} without importing it")
    return problems


def main() -> None:
    failed = False
    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        problems = validate(path)
        if problems:
            failed = True
            print(path.name)
            for problem in problems:
                print(f"  - {problem}")
    if failed:
        raise SystemExit(1)
    print("All notebooks satisfy the reproducibility checks.")


if __name__ == "__main__":
    main()
