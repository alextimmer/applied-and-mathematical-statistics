# Applied and Mathematical Statistics

An interactive, notebook-first course in applied statistics.

## What is this course?

A complete set of Jupyter notebooks for teaching (and learning) statistics — from probability basics through Bayesian inference and causal reasoning. Every key concept is accompanied by animated visualisations that show the idea in motion, not just as a static formula.

## Course Philosophy

This course follows a **"from scratch, then professional"** approach:

1. **Theory first** — rigorous mathematical development with intuitive explanations.
2. **Build it yourself** — implement every method by hand in Python and NumPy.
3. **Then use the real tool** — learn the professional library (SciPy, statsmodels, PyMC) with full understanding of what it does internally.

By the end, you'll never use a statistical tool you don't understand.

## Modules

| #  | Module                           | Topics                                                                |
|----|----------------------------------|-----------------------------------------------------------------------|
| 00 | Prerequisites                    | Python, NumPy, Matplotlib refresher                                   |
| 01 | Probability Basics               | Sample spaces, combinatorics, conditional probability, Bayes' theorem |
| 02 | Distributions                    | Discrete & continuous families, CLT                                   |
| 03 | Descriptive Statistics           | Summary stats, data exploration                                       |
| 04 | Estimation                       | MLE, confidence intervals, bootstrap                                  |
| 05 | Hypothesis Testing               | p-values, power, ROC curves                                           |
| 06 | Linear Models                    | Regression, ANOVA, diagnostics                                        |
| 07 | Bayesian Inference               | Priors, posteriors, MCMC                                              |
| 08 | Bayesian Regression              | PyMC, model comparison                                                |
| 09 | Hierarchical Models              | Partial pooling, varying effects                                      |
| 10 | Causal Inference                 | DAGs, confounds, do-calculus                                          |
| 11 | Machine Learning & Generative AI | ML foundations, neural nets, generative models                        |
| 12 | Advanced Topics                  | GPs, HMMs, missing data                                               |
| 13 | Projects                         | Capstone exercises                                                    |

## Getting Started

See the [installation guide](notebooks/00_prerequisites/INSTALL.md) for full setup instructions. The short version:

```bash
conda env create -f environment.yml
conda activate amstats
pip install -e .
jupyter lab
```

Then open `notebooks/00_prerequisites/01_python_and_tools.ipynb` and start from there.

## License

MIT
