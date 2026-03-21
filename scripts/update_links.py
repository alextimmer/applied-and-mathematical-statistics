"""Update navigation links in notebooks for the two new additions."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def update_nb(path, old_text, new_text):
    """Replace old_text with new_text in any markdown cell source."""
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        if old_text in src:
            src = src.replace(old_text, new_text)
            cell["source"] = [line + "\n" for line in src.split("\n")]
            # Fix last line (no trailing \n)
            if cell["source"] and cell["source"][-1].endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"  ✓ Updated {path.relative_to(ROOT)}")
    else:
        print(f"  - No match in {path.relative_to(ROOT)}")


def add_nav_cell(path, nav_text):
    """Append a navigation markdown cell before the last code cell (save_gifs)."""
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the save_gifs cell and insert a markdown cell before it
    for i in range(len(nb["cells"]) - 1, -1, -1):
        cell = nb["cells"][i]
        if cell["cell_type"] == "code" and "save_gifs" in "".join(cell.get("source", [])):
            nav_cell = {
                "cell_type": "markdown",
                "id": "nav_next",
                "metadata": {},
                "source": [line + "\n" for line in nav_text.split("\n")],
            }
            # Fix last line
            nav_cell["source"][-1] = nav_cell["source"][-1].rstrip("\n")
            # Insert after save_gifs
            nb["cells"].insert(i + 1, nav_cell)
            break

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"  ✓ Added nav cell to {path.relative_to(ROOT)}")


# 1. CLT notebook: "Next" should point to Monte Carlo instead of Module 03
print("1. Updating CLT notebook:")
update_nb(
    ROOT / "notebooks" / "02_distributions" / "03_central_limit_theorem.ipynb",
    "**This completes Module 02: Distributions.** You now know the major discrete and continuous families, their properties, and the CLT that connects them all through the Normal distribution.\n\n**Next:** [Module 03 — Descriptive Statistics](../03_descriptive_stats/01_summary_statistics.ipynb) — Summary statistics, data exploration, and the tools we use to describe data before modelling it.",
    "**Next:** [04 — Monte Carlo Sampling](04_monte_carlo_sampling.ipynb) — How to generate random samples from arbitrary distributions: the inverse transform, acceptance-rejection, and importance sampling.",
)

# 2. GP notebook: add "Next" link pointing to Module 12
print("2. Updating Gaussian Processes notebook:")
add_nav_cell(
    ROOT / "notebooks" / "11_machine_learning" / "03_gaussian_processes.ipynb",
    "---\n\n**Next:** [Module 12 — Time Series Fundamentals](../12_advanced_topics/01_time_series_fundamentals.ipynb) — Stationarity, autocorrelation, ARIMA models, and forecasting.",
)

print("\nDone.")
