"""
Tag notebook code cells for visibility in the Jupyter Book HTML build.

Since _config.yml sets nb_code_source_hidden: true, ALL code cells are hidden
by default (behind a toggle). This script goes further:

  - Cells to REMOVE completely:  get 'remove-input' tag  (no toggle, invisible)
  - Cells to SHOW:               get 'show-input' tag    (visible by default)
  - Everything else:             stays hidden behind toggle (default behavior)

Rules:
  - Manim cells (%%manim)           → remove-input  (IP: completely invisible)
  - Pure plotting cells             → remove-input  (IP: completely invisible)
  - Setup cells (class Cfg, etc.)   → (no tag)      (hidden behind toggle)
  - Plotting cells with educational
    code (def, class, significant
    logic beyond the plot)          → show-input    (visible)
  - All other code cells            → show-input    (visible)

ONLY modifies cell["metadata"]["tags"] — never touches code, text, or outputs.

Usage:
    python scripts/tag_hide_input.py --dry-run    # preview changes
    python scripts/tag_hide_input.py --apply       # apply changes
"""

import json
import sys
from pathlib import Path

# Patterns that indicate a Manim cell — NEVER show
MANIM_PATTERNS = [
    "%%manim",
]

# Patterns that indicate a setup/config cell — NEVER show
SETUP_PATTERNS = [
    "class Cfg",
    "cfg = Cfg(",
    "cfg.apply_manim_config()",
    "from manim import",
    "IN_COLAB",
]

# Patterns that indicate a cell produces a plot/figure
PLOT_PATTERNS = [
    "plt.show()",
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

# Patterns that indicate educational value worth showing even in a plot cell
EDUCATIONAL_PATTERNS = [
    "def ",
    "class ",
    "lambda ",         # non-trivial function definitions
    "for ",            # loops that teach an algorithm
    "while ",
]


def is_manim_cell(source: str) -> bool:
    return any(p in source for p in MANIM_PATTERNS)


def is_setup_cell(source: str) -> bool:
    return any(p in source for p in SETUP_PATTERNS)


def is_plot_cell(source: str) -> bool:
    return any(p in source for p in PLOT_PATTERNS)


def has_educational_content(source: str) -> bool:
    """Check if a cell has significant educational code beyond plotting."""
    lines = source.strip().splitlines()
    # Count lines that are educational (not blank, not comments, not pure plot)
    educational_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Skip lines that are purely plot setup
        if any(p in stripped for p in PLOT_PATTERNS):
            continue
        if any(stripped.startswith(kw) for kw in ["ax.", "plt.", "fig.", "sns."]):
            continue
        educational_lines += 1

    # Has function/class definitions → educational
    if any(p in source for p in ["def ", "class "]):
        return True

    # More than half of non-blank code lines are non-plot → educational
    total_code_lines = sum(
        1 for l in lines
        if l.strip() and not l.strip().startswith("#")
    )
    if total_code_lines > 0 and educational_lines / total_code_lines > 0.5:
        return True

    return False


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

        # --- Decide visibility ---

        def set_tag(desired, reason):
            """Ensure cell has exactly the desired tag (or no tag).
            desired: 'remove-input', 'show-input', or None (no tag)."""
            nonlocal modified
            for t in ("remove-input", "show-input"):
                if t in tags and t != desired:
                    changes.append(f"  cell {i}: - {t}  ({reason})")
                    if apply:
                        tags.remove(t)
                        modified = True
            if desired and desired not in tags:
                changes.append(f"  cell {i}: + {desired}  ({reason})")
                if apply:
                    tags.append(desired)
                    modified = True

        # Manim cells: completely remove from HTML (IP protection)
        if is_manim_cell(source):
            set_tag("remove-input", "manim cell")
            continue

        # Setup cells: hidden behind toggle (no tag needed)
        if not found_first_code:
            found_first_code = True
            if is_setup_cell(source):
                set_tag(None, "setup cell")
                continue

        # Plot cells: remove pure plots (IP), show educational ones
        if is_plot_cell(source):
            if has_educational_content(source):
                set_tag("show-input", "plot cell with educational code")
            else:
                set_tag("remove-input", "pure plot cell")
            continue

        # All other code cells: SHOW
        set_tag("show-input", "educational code")

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

    total_show = 0
    total_remove = 0
    total_cleared = 0
    for nb_path in nb_files:
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        changes = process_notebook(nb_path, apply)
        if changes:
            rel = nb_path.relative_to(notebooks_dir.parent)
            print(f"{rel}:")
            for c in changes:
                print(c)
                if "+ show-input" in c:
                    total_show += 1
                elif "+ remove-input" in c:
                    total_remove += 1
                elif c.startswith("  cell") and "- " in c:
                    total_cleared += 1
            print()

    action = "tagged" if apply else "would be tagged"
    print(f"--- Total: {total_show} cells {action} show-input, "
          f"{total_remove} cells {action} remove-input, "
          f"{total_cleared} stale tags cleared ---")


if __name__ == "__main__":
    main()
