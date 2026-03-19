# Installation Guide

This guide walks you through setting up everything you need to run the course notebooks on your own machine.

## 1. Install Python via Conda (Miniconda)

We use **Conda** to manage Python and all dependencies in an isolated environment.

1. Download **Miniconda** from <https://docs.anaconda.com/miniconda/install/>.
   - Choose the installer for your OS (Windows, macOS, or Linux).
   - Use the default options during installation.
2. Open a terminal (or "Anaconda Prompt" on Windows).
3. Verify the install:
   ```bash
   conda --version
   ```

## 2. Create the Course Environment

From the **root of this repository**, run:

```bash
conda env create -f environment.yml
conda activate amstats
```

This installs Python 3.11, Jupyter, NumPy, SciPy, Matplotlib, Manim, PyMC, and all other course dependencies in one step.

To update the environment after we add new packages:

```bash
conda env update -f environment.yml --prune
```

## 3. Install LaTeX (required for Manim)

Manim uses LaTeX to render mathematical text in animations. Without it, animations fall back to plain text.

### Windows — MiKTeX

1. Download MiKTeX from <https://miktex.org/download>.
2. Run the installer. Choose "Install missing packages on the fly: Yes".
3. The notebook setup cell automatically adds MiKTeX to the PATH, so you do not need to configure anything else.

### macOS

```bash
brew install --cask mactex-no-gui
```

### Linux (Debian/Ubuntu)

```bash
sudo apt-get install texlive-full
```

### Verify LaTeX

```bash
pdflatex --version
```

## 4. Install the Course Package

The `src/amstats/` package provides shared utilities (color palettes, plotting helpers, Manim constructors). Install it in development mode:

```bash
pip install -e ".[dev]"
```

## 5. Launch Jupyter

```bash
jupyter lab
```

Navigate to `notebooks/` and open any notebook. Run the first cell — if it prints no errors, you're ready.

## 6. Optional: Bayesian Stack

Modules 07–09 use PyMC and optionally CmdStanPy. These are included in the conda environment, but CmdStan itself needs a one-time install:

```bash
python -m cmdstanpy.install_cmdstan
```

## 7. Building the Course Website (Jupyter Book)

All notebooks can be rendered to a static HTML site using [Jupyter Book](https://jupyterbook.org/). From the **repo root**:

```bash
jb build .
```

This produces `_build/html/`. Open `_build/html/index.html` in a browser to preview.

### Option A: Build from saved outputs (default)

Notebooks must be **run and saved with outputs** before building. The workflow:

1. Open each notebook in Jupyter and run it (Kernel → Restart & Run All).
2. Save the notebook — the outputs (plots, GIFs) are now stored inside the `.ipynb` file.
3. Run `jb build .` — the HTML will include all plots and GIFs.

This is the default (`execute_notebooks: "off"` in `_config.yml`). The build machine doesn't need Manim or LaTeX — it just uses whatever outputs are already in the files.

### Option B: Re-execute during build

If you have the full environment (Manim, LaTeX, all dependencies), you can have Jupyter Book run the notebooks during build. Edit `_config.yml` and change:

```yaml
execute_notebooks: "auto"    # only re-run notebooks that have no saved outputs
# or
execute_notebooks: "force"   # re-run ALL notebooks unconditionally
```

Then `jb build .` will execute the notebooks and capture fresh outputs. This is slower but ensures everything is up to date.

### Rendering a single notebook to HTML

If you just need one notebook as a standalone HTML file:

```bash
jupyter nbconvert --to html notebooks/01_probability_basics/01_sample_spaces.ipynb
```

This creates `01_sample_spaces.html` next to the notebook with all outputs (plots, GIFs) embedded inline.

## Troubleshooting

| Problem                                          | Solution                                                                             |
|--------------------------------------------------|--------------------------------------------------------------------------------------|
| `MathTex` shows plain text instead of LaTeX      | LaTeX not found. Install MiKTeX/MacTeX/texlive and restart the kernel.               |
| `FileNotFoundError` when re-running a Manim cell | Restart the Jupyter kernel. This is a known Manim 0.18 caching issue.                |
| `MAX_PATH` errors on Windows                     | Keep the repo path short (e.g., `C:\dev\amstats\`).                                  |
| `conda activate` doesn't work                    | Run `conda init bash` (or `conda init powershell`) first, then restart the terminal. |
