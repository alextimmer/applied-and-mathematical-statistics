"""
Add 'hide-input' tags to notebook code cells that are:
  1. The setup cell (first code cell in the notebook)
  2. Plot/figure cells (contain plt.show, %%manim, fig., .savefig, etc.)

ONLY modifies cell["metadata"]["tags"] — never touches code, text, or outputs.

Usage:
    python scripts/tag_hide_input.py --dry-run    # preview changes
    python scripts/tag_hide_input.py --apply       # apply changes
"""

import json
import sys
from pathlib import Path

# Patterns that indicate a cell produces a plot/figure
PLOT_PATTERNS = [
    "plt.show()",
    "%%manim",
    ".savefig(",
    "fig.suptitle(",
    "fig.subplots_adjust(",
    "plt.tight_layout()",
    "plt.subplots(",
    "fig, ax",
    "fig =",
    "display(Video(",
    "display(Image(",
    "ax.imshow(",
    "sns.heatmap(",
    "plt.imshow(",
]

# Patterns that indicate a setup/config cell
SETUP_PATTERNS = [
    "class Cfg",
    "cfg = Cfg(",
    "cfg.apply_manim_config()",
    "from manim import",
    "IN_COLAB",
]


def is_setup_cell(source: str) -> bool:
    return any(p in source for p in SETUP_PATTERNS)


def is_plot_cell(source: str) -> bool:
    return any(p in source for p in PLOT_PATTERNS)


def process_notebook(nb_path: Path, apply: bool) -> list[str]:
    """Process a single notebook. Returns list of change descriptions."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changes = []
    modified = False
    found_first_code = False

    for i, cell in enumerate(nb.get("cells", [])):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        # Ensure metadata and tags exist
        meta = cell.setdefault("metadata", {})
        tags = meta.setdefault("tags", [])

        should_hide = False
        reason = ""

        # First code cell = setup
        if not found_first_code:
            found_first_code = True
            if is_setup_cell(source):
                should_hide = True
                reason = "setup cell"

        # Plot/figure cells
        if not should_hide and is_plot_cell(source):
            should_hide = True
            # Find which pattern matched
            for p in PLOT_PATTERNS:
                if p in source:
                    reason = f"plot cell (matched: {p})"
                    break

        if should_hide and "hide-input" not in tags:
            changes.append(f"  cell {i}: + hide-input  ({reason})")
            if apply:
                tags.append("hide-input")
                modified = True

    if apply and modified:
        with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")

    return changes


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("--dry-run", "--apply"):
        print("Usage: python scripts/tag_hide_input.py --dry-run|--apply")
        sys.exit(1)

    apply = sys.argv[1] == "--apply"
    mode = "APPLYING" if apply else "DRY RUN"
    print(f"=== {mode} ===\n")

    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    nb_files = sorted(notebooks_dir.rglob("*.ipynb"))

    total_changes = 0
    for nb_path in nb_files:
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        changes = process_notebook(nb_path, apply)
        if changes:
            rel = nb_path.relative_to(notebooks_dir.parent)
            print(f"{rel}:")
            for c in changes:
                print(c)
            print()
            total_changes += len(changes)

    print(f"--- Total: {total_changes} cells {'tagged' if apply else 'would be tagged'} ---")


if __name__ == "__main__":
    main()
