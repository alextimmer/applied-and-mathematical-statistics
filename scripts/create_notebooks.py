"""Create the Monte Carlo Sampling and Time Series Fundamentals notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def nb(cells):
    """Wrap cells in a valid notebook structure."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }


def md(id_, source):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": source.split("\n")}


def code(id_, source, tags=None):
    meta = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "id": id_,
        "metadata": meta,
        "source": source.split("\n"),
        "execution_count": None,
        "outputs": [],
    }


# ═══════════════════════════════════════════════════════════════════
#  NOTEBOOK 1: Monte Carlo Sampling Methods
# ═══════════════════════════════════════════════════════════════════

mc_cells = [
    # ── Title ──
    md("m1", """# Monte Carlo Sampling Methods

## Learning Objectives

- Understand why we need methods to generate random samples from arbitrary distributions.
- Implement the **inverse transform method** from scratch and prove why it works.
- Implement **acceptance-rejection sampling** and understand its efficiency trade-offs.
- Apply **importance sampling** for estimating expectations under difficult distributions.
- Connect these methods to the MCMC techniques used in Bayesian inference (Module 07).

## Prerequisites

- Module 02.01–02.03 (discrete and continuous distributions, CDFs)
- Basic calculus (integration, change of variables)"""),

    # ── Setup ──
    code("m2", """import sys, os, shutil
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.system(
        "sudo apt-get update -qq && sudo apt-get install -y -qq "
        "libcairo2-dev libpango1.0-dev && pip install -q manim ipython==8.21.0"
    )

_miktex_bin = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64"
if _miktex_bin.exists() and str(_miktex_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + str(_miktex_bin)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.abspath("../../src"))
from amstats.plotting import apply_style

apply_style()

rng = np.random.default_rng(42)


class Cfg:
    root = Path("../../").resolve()
    gif_dir = root / "media" / "gifs"
    has_latex: bool = (
        shutil.which("latex") is not None or shutil.which("pdflatex") is not None
    )

    def __init__(self):
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        if not self.has_latex:
            print("⚠ LaTeX not found — MathTex will fall back to Text().")

    def apply_manim_config(self):
        from manim import config as mcfg
        mcfg.format = "gif"

    def math_text(self, expr, **kwargs):
        from manim import MathTex, Text
        if self.has_latex:
            return MathTex(expr, **kwargs)
        return Text(expr, **kwargs)

    def save_gifs(self, clean=True):
        local_media = Path("media")
        found = list(local_media.rglob("*.gif")) if local_media.exists() else []
        if not found:
            print("  No new GIFs to save.")
            return
        for gif in found:
            dest = self.gif_dir / gif.name
            shutil.copy2(gif, dest)
            print(f"  ✓ media/gifs/{gif.name}")
        if clean:
            for sub in ("videos", "images", "Tex"):
                d = local_media / sub
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            print("  Cleaned up local temp render files.")


cfg = Cfg()""", tags=["hide-input"]),

    # ── Manim setup ──
    code("m3", """from manim import *

cfg.apply_manim_config()
math_text = cfg.math_text

from amstats.manim_utils import C, COLORS""", tags=["hide-input"]),

    # ═══ Section 1: Introduction ═══
    md("m4", """---

## 1. Why Monte Carlo Sampling?

In the previous notebooks we studied probability distributions — their PMFs, PDFs, CDFs, means, and variances. We can *describe* these distributions analytically. But in practice we often need to **generate random samples** from a distribution, and this turns out to be a surprisingly deep problem.

**Why do we need samples?**

1. **Estimating expectations.** If $X \\sim f$, then $E[g(X)] = \\int g(x) f(x)\\, dx$. For many choices of $g$ and $f$, this integral has no closed-form solution. But if we can draw samples $X_1, \\ldots, X_n \\sim f$, then by the Law of Large Numbers:

$$\\hat{\\mu}_n = \\frac{1}{n} \\sum_{i=1}^n g(X_i) \\xrightarrow{\\text{a.s.}} E[g(X)] \\quad \\text{as } n \\to \\infty$$

2. **Simulation.** Monte Carlo simulation lets us study complex systems (queuing networks, financial models, physical processes) by repeatedly sampling random inputs and observing the outputs.

3. **Bayesian inference.** In Module 07 we will need to draw samples from posterior distributions that have no analytical form. The MCMC methods we'll use there are sophisticated extensions of the basic sampling techniques we develop in this notebook.

**The core problem:** Your computer can generate $U \\sim \\text{Uniform}(0, 1)$ — that's built into every programming language. But how do you turn these uniform random numbers into samples from an *arbitrary* distribution $f$?

This notebook presents three fundamental answers:

| Method | Idea | When to use |
|---|---|---|
| **Inverse transform** | Map uniform samples through $F^{-1}$ | CDF inverse available in closed form |
| **Acceptance-rejection** | Propose from an easy distribution, accept/reject | Any bounded density, no $F^{-1}$ needed |
| **Importance sampling** | Sample from a different distribution, reweight | Estimating expectations, rare events |"""),

    # ═══ Section 2: Inverse Transform Method ═══
    md("m5", """---

## 2. The Inverse Transform Method

### 2.1 The key theorem

The inverse transform method is the most elegant connection between uniform random variables and arbitrary distributions. The idea is beautifully simple: if you can invert the CDF, you can sample from the distribution.

**Theorem (Inverse Transform).** Let $F$ be a continuous, strictly increasing CDF. If $U \\sim \\text{Uniform}(0, 1)$, then

$$X = F^{-1}(U) \\sim F$$

That is, $X$ has CDF $F$.

**Proof.** We need to show that $P(X \\leq x) = F(x)$ for all $x$.

$$P(X \\leq x) = P\\big(F^{-1}(U) \\leq x\\big)$$

Since $F$ is strictly increasing, $F^{-1}(U) \\leq x$ if and only if $U \\leq F(x)$. Therefore:

$$P\\big(F^{-1}(U) \\leq x\\big) = P\\big(U \\leq F(x)\\big)$$

Since $U \\sim \\text{Uniform}(0, 1)$ and $F(x) \\in [0, 1]$, we have $P(U \\leq F(x)) = F(x)$. Hence:

$$P(X \\leq x) = F(x) \\quad \\blacksquare$$

The proof works because a uniform random variable has the remarkable property that $P(U \\leq u) = u$ for any $u \\in [0, 1]$. This makes the CDF "transparent" — it passes right through.

### 2.2 Geometric intuition

Think of it this way: the CDF $F(x)$ maps values on the $x$-axis to probabilities on the $y$-axis (between 0 and 1). The inverse $F^{-1}$ reverses this — it maps probabilities back to values. So if you feed in a *uniformly distributed* probability $U$, you get back a value $X$ whose distribution matches $F$.

Regions where $F$ is steep (high density) correspond to many uniform values being mapped to a narrow range of $x$ — exactly producing more samples where the density is high."""),

    # ── Inverse transform: from scratch for Exponential ──
    md("m6", """### 2.3 From scratch: Exponential distribution

Let's apply the inverse transform to the **Exponential($\\lambda$)** distribution.

**Step 1 — CDF:**

$$F(x) = 1 - e^{-\\lambda x}, \\quad x \\geq 0$$

**Step 2 — Invert:** Set $u = F(x) = 1 - e^{-\\lambda x}$ and solve for $x$:

$$u = 1 - e^{-\\lambda x} \\implies e^{-\\lambda x} = 1 - u \\implies -\\lambda x = \\ln(1 - u) \\implies x = -\\frac{\\ln(1 - u)}{\\lambda}$$

**Step 3 — Since $U$ and $1 - U$ have the same distribution** (both Uniform(0,1)), we can simplify:

$$X = -\\frac{\\ln U}{\\lambda}$$

This is our sampling formula. Let's implement it and verify it produces the correct distribution."""),

    code("m7", """# From scratch: inverse transform sampling for Exponential(λ)
def sample_exponential(lam, n, rng):
    \"\"\"Generate n samples from Exponential(λ) using inverse transform.\"\"\"
    u = rng.uniform(size=n)
    return -np.log(u) / lam


lam = 2.0
n_samples = 50_000
samples = sample_exponential(lam, n_samples, rng)

# Verify: compare sample statistics to theory
print(f"Exponential(λ={lam}):")
print(f"  Theory:  E[X] = {1/lam:.4f},   Var(X) = {1/lam**2:.4f}")
print(f"  Sample:  mean = {samples.mean():.4f},   var  = {samples.var():.4f}")"""),

    md("m8", """The sample statistics match the theoretical values closely. Now let's compare the histogram to the true PDF visually."""),

    code("m9", """# Histogram vs theoretical PDF
fig, ax = plt.subplots(figsize=(9, 4.5))

x = np.linspace(0, 4, 300)
ax.hist(samples, bins=80, density=True, alpha=0.6, label="Inverse transform samples")
ax.plot(x, stats.expon(scale=1/lam).pdf(x), "k-", lw=2, label=f"Exponential(λ={lam}) PDF")

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Inverse Transform Sampling — Exponential Distribution")
ax.legend()
plt.tight_layout()
plt.show()""", tags=["hide-input"]),

    # ── Manim animation: inverse transform ──
    md("m10", """### 2.4 Visualising the inverse transform

The following animation shows the inverse transform in action. Uniform samples on the $y$-axis (left) are mapped through the inverse CDF curve to produce Exponential samples on the $x$-axis (bottom). Watch how the density of points on the $x$-axis is highest near zero — exactly where the Exponential PDF is largest."""),

    code("m11", """%%manim -qm -v WARNING InverseTransformDemo


class InverseTransformDemo(Scene):
    \"\"\"Animate uniform samples being mapped through the inverse CDF
    to produce exponential samples.\"\"\"

    def construct(self):
        lam = 1.5
        title = Text("Inverse Transform: U → F⁻¹(U) → X", font_size=28).to_edge(UP)
        self.play(Write(title), run_time=0.6)

        # Axes: x-axis = sample value, y-axis = uniform [0,1]
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1, 0.2],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "font_size": 16},
        ).shift(DOWN * 0.3)
        x_lbl = axes.get_x_axis_label(Text("x (Exponential samples)", font_size=16),
                                       edge=DOWN, direction=DOWN)
        y_lbl = axes.get_y_axis_label(Text("u (Uniform)", font_size=16),
                                       edge=LEFT, direction=LEFT)
        self.play(Create(axes), Write(x_lbl), Write(y_lbl), run_time=0.8)

        # Draw the CDF curve: F(x) = 1 - exp(-λx)
        cdf_curve = axes.plot(lambda x: 1 - np.exp(-lam * x), x_range=[0, 4],
                              color=C.GOLD, stroke_width=3)
        cdf_label = Text("CDF: F(x)", font_size=16, color=C.GOLD).next_to(
            axes.c2p(3, 0.95), LEFT
        )
        self.play(Create(cdf_curve), Write(cdf_label), run_time=0.8)

        # Animate 15 samples flowing through the CDF
        np.random.seed(42)
        u_vals = np.random.uniform(0.02, 0.98, 15)

        for i, u in enumerate(u_vals[:15]):
            x_val = -np.log(1 - u) / lam  # inverse CDF

            # Dot on y-axis
            y_dot = Dot(axes.c2p(0, u), color=C.CYAN, radius=0.06)
            # Horizontal line to CDF
            h_line = DashedLine(axes.c2p(0, u), axes.c2p(x_val, u),
                                color=C.CYAN, stroke_width=1.5)
            # Vertical line down to x-axis
            v_line = DashedLine(axes.c2p(x_val, u), axes.c2p(x_val, 0),
                                color=C.SALMON, stroke_width=1.5)
            # Dot on x-axis
            x_dot = Dot(axes.c2p(x_val, 0), color=C.SALMON, radius=0.06)

            speed = 0.25 if i < 5 else 0.12
            self.play(FadeIn(y_dot), run_time=speed)
            self.play(Create(h_line), run_time=speed)
            self.play(Create(v_line), FadeIn(x_dot), run_time=speed)
            self.play(FadeOut(h_line), FadeOut(v_line), FadeOut(y_dot),
                      run_time=speed * 0.5)

        self.wait(1)""", tags=["hide-input"]),

    # ── From scratch: Triangular distribution ──
    md("m12", """### 2.5 From scratch: Triangular distribution

To reinforce the method, let's apply it to a less standard distribution. The **Triangular distribution** on $[0, 2]$ with peak at 1 has PDF:

$$f(x) = \\begin{cases} x & 0 \\leq x \\leq 1 \\\\ 2 - x & 1 < x \\leq 2 \\end{cases}$$

**Step 1 — CDF (by integration):**

$$F(x) = \\begin{cases} \\frac{x^2}{2} & 0 \\leq x \\leq 1 \\\\ 1 - \\frac{(2-x)^2}{2} & 1 < x \\leq 2 \\end{cases}$$

**Step 2 — Invert each piece:**

- For $0 \\leq u \\leq 0.5$: $u = x^2/2 \\implies x = \\sqrt{2u}$
- For $0.5 < u \\leq 1$: $u = 1 - (2-x)^2/2 \\implies x = 2 - \\sqrt{2(1-u)}$"""),

    code("m13", """# From scratch: inverse transform for Triangular(0, 1, 2)
def sample_triangular(n, rng):
    \"\"\"Sample from Triangular(0, peak=1, 2) using inverse transform.\"\"\"
    u = rng.uniform(size=n)
    x = np.where(u <= 0.5, np.sqrt(2 * u), 2 - np.sqrt(2 * (1 - u)))
    return x


tri_samples = sample_triangular(50_000, rng)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.linspace(0, 2, 300)
pdf_tri = np.where(x <= 1, x, 2 - x)

ax.hist(tri_samples, bins=80, density=True, alpha=0.6, label="Inverse transform samples")
ax.plot(x, pdf_tri, "k-", lw=2, label="Triangular PDF")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Inverse Transform — Triangular Distribution")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Theory:  E[X] = 1.0,   Var(X) = {1/6:.4f}")
print(f"Sample:  mean = {tri_samples.mean():.4f},   var = {tri_samples.var():.4f}")""", tags=["hide-input"]),

    # ── Professional comparison ──
    md("m14", """### 2.6 Professional tools

In practice, you rarely implement the inverse transform yourself. SciPy and NumPy provide optimised samplers for all standard distributions:"""),

    code("m15", """# Professional: scipy and numpy use optimised methods internally
from scipy.stats import expon, triang

# SciPy Exponential
scipy_samples = expon(scale=1/lam).rvs(size=50_000, random_state=42)
print(f"SciPy Exponential: mean = {scipy_samples.mean():.4f}")

# NumPy Exponential (uses Ziggurat algorithm — faster than inverse transform!)
numpy_samples = rng.exponential(scale=1/lam, size=50_000)
print(f"NumPy Exponential: mean = {numpy_samples.mean():.4f}")

# SciPy Triangular: triang(c, loc, scale) where c = (peak - loc) / scale
scipy_tri = triang(c=0.5, loc=0, scale=2).rvs(size=50_000, random_state=42)
print(f"SciPy Triangular:  mean = {scipy_tri.mean():.4f}")"""),

    md("m16", """### 2.7 Limitations

The inverse transform is mathematically elegant but has a practical limitation: **you need $F^{-1}$ in closed form.** For many important distributions — the Normal, the Beta, the Gamma — the CDF involves special functions (error function, incomplete Beta, etc.) whose inverses are not available as simple formulae.

What do we do then? We need a method that works with the **PDF** directly, without requiring the CDF inverse. This is exactly what acceptance-rejection sampling provides."""),

    # ═══ Section 3: Acceptance-Rejection Sampling ═══
    md("m17", """---

## 3. Acceptance-Rejection Sampling

### 3.1 The algorithm

Suppose we want to sample from a **target** distribution with PDF $f(x)$, but we cannot compute $F^{-1}$. The acceptance-rejection method works as follows:

1. Choose a **proposal** distribution with PDF $g(x)$ that is easy to sample from.
2. Find a constant $M \\geq 1$ such that $f(x) \\leq M \\cdot g(x)$ for all $x$. The function $M \\cdot g(x)$ is called the **envelope**.
3. Repeat:
   - Draw $Y \\sim g$ (a proposal)
   - Draw $U \\sim \\text{Uniform}(0, 1)$ (independently)
   - If $U \\leq \\dfrac{f(Y)}{M \\cdot g(Y)}$, **accept** $Y$; otherwise **reject** and try again.

**Claim:** The accepted values are i.i.d. samples from $f$.

### 3.2 Why does it work?

**Proof.** We need to show that $P(Y \\leq x \\mid \\text{accept}) = F(x)$.

The probability of accepting a proposal $Y = y$ is:

$$P(\\text{accept} \\mid Y = y) = P\\left(U \\leq \\frac{f(y)}{M \\cdot g(y)}\\right) = \\frac{f(y)}{M \\cdot g(y)}$$

The overall acceptance probability is:

$$P(\\text{accept}) = \\int_{-\\infty}^{\\infty} \\frac{f(y)}{M \\cdot g(y)} \\cdot g(y)\\, dy = \\frac{1}{M} \\int f(y)\\, dy = \\frac{1}{M}$$

By Bayes' theorem, the density of accepted values is:

$$f_{Y \\mid \\text{accept}}(y) = \\frac{P(\\text{accept} \\mid Y = y) \\cdot g(y)}{P(\\text{accept})} = \\frac{\\frac{f(y)}{M \\cdot g(y)} \\cdot g(y)}{\\frac{1}{M}} = f(y) \\quad \\blacksquare$$

### 3.3 Efficiency

The acceptance probability is $1/M$. This means:
- On average, we need $M$ proposals to get one accepted sample.
- **We want $M$ as small as possible** — the tighter the envelope $M \\cdot g(x)$ fits around $f(x)$, the more efficient the sampler.
- A bad choice of $g$ (e.g., a very wide Uniform) wastes most proposals."""),

    # ── Accept-reject: from scratch for Beta ──
    md("m18", """### 3.4 From scratch: sampling from Beta(2.7, 6.3)

The Beta distribution's CDF involves the incomplete Beta function — no closed-form inverse. This makes it a perfect test case for acceptance-rejection.

**Target:** $f(x) = \\text{Beta}(2.7, 6.3)$ on $[0, 1]$.

**Proposal 1 — Uniform(0, 1):** The simplest choice. We need $M = \\max_x f(x)$, which is the mode height of the Beta PDF."""),

    code("m19", """# From scratch: acceptance-rejection for Beta(2.7, 6.3) with Uniform proposal
a, b = 2.7, 6.3
target = stats.beta(a, b)

# M = max of the Beta PDF (at the mode)
mode = (a - 1) / (a + b - 2)
M_uniform = target.pdf(mode)  # envelope height
print(f"Beta({a},{b}) mode at x = {mode:.4f}, M = {M_uniform:.4f}")
print(f"Acceptance rate = 1/M = {1/M_uniform:.4f} = {100/M_uniform:.1f}%")


def accept_reject_beta_uniform(n_target, a, b, rng):
    \"\"\"Sample n_target values from Beta(a,b) using Uniform proposal.\"\"\"
    target = stats.beta(a, b)
    M = target.pdf((a - 1) / (a + b - 2))  # max of PDF

    samples = []
    n_proposed = 0
    while len(samples) < n_target:
        y = rng.uniform()                    # proposal from g = Uniform(0,1)
        u = rng.uniform()                    # acceptance threshold
        n_proposed += 1
        if u <= target.pdf(y) / M:           # accept?
            samples.append(y)

    return np.array(samples), n_proposed


samples_ar, n_proposed = accept_reject_beta_uniform(20_000, a, b, rng)
print(f"\\nGenerated {len(samples_ar)} samples from {n_proposed} proposals")
print(f"Empirical acceptance rate: {len(samples_ar)/n_proposed:.4f}")
print(f"Theory:  E[X] = {a/(a+b):.4f},  Var(X) = {a*b/((a+b)**2*(a+b+1)):.4f}")
print(f"Sample:  mean = {samples_ar.mean():.4f},  var  = {samples_ar.var():.4f}")"""),

    md("m20", """The histogram below shows our accepted samples against the true Beta PDF. The shaded region between $f(x)$ and $M \\cdot g(x)$ represents the "wasted" proposals — the rejection zone."""),

    code("m21", """# Visualise: target, envelope, and accepted samples
fig, ax = plt.subplots(figsize=(9, 5))
x = np.linspace(0, 1, 300)

ax.fill_between(x, target.pdf(x), M_uniform, alpha=0.15, color="red",
                label=f"Rejection zone ({100*(1-1/M_uniform):.0f}% of area)")
ax.fill_between(x, 0, target.pdf(x), alpha=0.15, color="green",
                label=f"Acceptance zone ({100/M_uniform:.0f}% of area)")
ax.plot(x, target.pdf(x), "k-", lw=2, label=f"Target: Beta({a},{b})")
ax.axhline(M_uniform, color="red", ls="--", lw=1.5,
           label=f"Envelope: M·g(x) = {M_uniform:.2f}")
ax.hist(samples_ar, bins=60, density=True, alpha=0.4, color=C.CYAN,
        label="Accepted samples")

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Acceptance-Rejection: Beta(2.7, 6.3) with Uniform Proposal")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()""", tags=["hide-input"]),

    # ── Manim: accept-reject animation ──
    md("m22", """### 3.5 Visualising acceptance and rejection

The animation below generates random points under the envelope $M \\cdot g(x)$. Points that fall below $f(x)$ (the target PDF) are **accepted** (green); points between $f(x)$ and the envelope are **rejected** (red). The accepted points, projected onto the $x$-axis, form samples from $f$."""),

    code("m23", """%%manim -qm -v WARNING AcceptRejectDemo


class AcceptRejectDemo(Scene):
    \"\"\"Animate acceptance-rejection sampling for Beta(2.7, 6.3).\"\"\"

    def construct(self):
        a, b = 2.7, 6.3
        from scipy.stats import beta as beta_dist
        target = beta_dist(a, b)
        M = target.pdf((a - 1) / (a + b - 2))

        title = Text("Accept-Reject Sampling: Beta(2.7, 6.3)", font_size=26).to_edge(UP)
        self.play(Write(title), run_time=0.5)

        axes = Axes(
            x_range=[0, 1, 0.2], y_range=[0, 3, 0.5],
            x_length=10, y_length=5,
            axis_config={"include_numbers": True, "font_size": 16},
        ).shift(DOWN * 0.3)
        self.play(Create(axes), run_time=0.5)

        # Draw target PDF
        pdf_curve = axes.plot(lambda x: target.pdf(x), x_range=[0.01, 0.99],
                              color=C.GOLD, stroke_width=3)
        # Draw envelope
        envelope = DashedLine(axes.c2p(0, M), axes.c2p(1, M),
                              color=C.SALMON, stroke_width=2)
        self.play(Create(pdf_curve), Create(envelope), run_time=0.6)

        # Generate and animate points
        np.random.seed(123)
        n_points = 40
        accepted = 0
        rejected = 0

        counter_text = Text(f"Accepted: 0 / 0", font_size=20, color=WHITE
                           ).to_corner(UR).shift(DOWN * 0.5)
        self.play(Write(counter_text), run_time=0.3)

        for i in range(n_points):
            y = np.random.uniform(0.02, 0.98)
            h = np.random.uniform(0, M)
            is_accept = h <= target.pdf(y)

            dot = Dot(axes.c2p(y, h), radius=0.05,
                      color=C.EMERALD if is_accept else C.SALMON,
                      fill_opacity=0.8)

            if is_accept:
                accepted += 1
            else:
                rejected += 1

            speed = 0.15 if i < 10 else 0.06
            self.play(FadeIn(dot), run_time=speed)

            new_counter = Text(
                f"Accepted: {accepted} / {accepted + rejected}",
                font_size=20, color=WHITE
            ).to_corner(UR).shift(DOWN * 0.5)
            self.remove(counter_text)
            self.add(new_counter)
            counter_text = new_counter

        self.wait(1.5)""", tags=["hide-input"]),

    # ── Better proposal ──
    md("m24", """### 3.6 Improving efficiency with a better proposal

With the Uniform proposal, our acceptance rate was only about $\\frac{1}{M} \\approx 38\\%$. We can do much better by choosing a proposal $g(x)$ that more closely matches the shape of $f(x)$.

For a Beta(2.7, 6.3) target, a natural choice is a Beta(2, 5) proposal — it has a similar shape but is wider, so it can act as an envelope."""),

    code("m25", """# Better proposal: Beta(2, 5) — closer shape to the target
proposal = stats.beta(2, 5)

# Find M: max of f(x)/g(x) over [0, 1]
x_grid = np.linspace(0.001, 0.999, 10_000)
ratio = target.pdf(x_grid) / proposal.pdf(x_grid)
M_beta = ratio.max() * 1.001  # tiny safety margin
print(f"Better proposal: M = {M_beta:.4f}")
print(f"Acceptance rate = 1/M = {1/M_beta:.4f} = {100/M_beta:.1f}%")
print(f"(vs {100/M_uniform:.1f}% with Uniform proposal — {M_uniform/M_beta:.1f}× improvement)")"""),

    code("m26", """# Compare the two envelopes visually
fig, axes_plt = plt.subplots(1, 2, figsize=(13, 5))

x = np.linspace(0.001, 0.999, 300)

for ax, (M, g_pdf, g_name) in zip(axes_plt, [
    (M_uniform, np.ones_like(x), "Uniform(0,1)"),
    (M_beta, proposal.pdf(x), "Beta(2,5)"),
]):
    envelope_vals = M * g_pdf
    ax.fill_between(x, target.pdf(x), envelope_vals, alpha=0.2, color="red")
    ax.fill_between(x, 0, target.pdf(x), alpha=0.2, color="green")
    ax.plot(x, target.pdf(x), "k-", lw=2, label="Target f(x)")
    ax.plot(x, envelope_vals, "r--", lw=1.5, label=f"M·g(x), M={M:.2f}")
    ax.set_title(f"Proposal: {g_name}\\nAcceptance ≈ {100/M:.0f}%", fontsize=11)
    ax.set_xlabel("x")
    ax.legend(fontsize=9)

axes_plt[0].set_ylabel("Density")
fig.suptitle("Tighter Envelope → Higher Acceptance Rate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()""", tags=["hide-input"]),

    # ═══ Section 4: Importance Sampling ═══
    md("m27", """---

## 4. Importance Sampling

### 4.1 The problem

Sometimes we don't need *samples* from $f$ per se — we need to compute an **expectation**:

$$\\mu = E_f[h(X)] = \\int h(x) f(x)\\, dx$$

If $f$ is hard to sample from, we can use a trick: sample from a *different* distribution $g$ (the **importance distribution**) and reweight.

### 4.2 The key identity

$$E_f[h(X)] = \\int h(x) f(x)\\, dx = \\int h(x) \\frac{f(x)}{g(x)} g(x)\\, dx = E_g\\left[h(X) \\cdot \\frac{f(X)}{g(X)}\\right]$$

The ratio $w(x) = f(x) / g(x)$ is called the **importance weight**. The estimator becomes:

$$\\hat{\\mu}_n = \\frac{1}{n} \\sum_{i=1}^n h(X_i) \\cdot w(X_i), \\quad X_i \\sim g$$

This is unbiased for any $g$ with $g(x) > 0$ wherever $f(x) > 0$. But the **variance** depends critically on how well $g$ matches $f \\cdot |h|$.

### 4.3 Why the choice of $g$ matters

The variance of the importance sampling estimator is:

$$\\text{Var}_g\\left[h(X) \\cdot w(X)\\right] = E_g\\left[\\left(h(X) \\cdot \\frac{f(X)}{g(X)}\\right)^2\\right] - \\mu^2$$

If $g$ is very different from $f$ in the regions where $h(x) \\cdot f(x)$ is large, some importance weights $w(X_i)$ will be enormous, making the estimator highly variable. The ideal importance distribution is $g^*(x) \\propto |h(x)| \\cdot f(x)$, but this is usually impractical to sample from (if we could, we wouldn't need importance sampling!).

**Rule of thumb:** Choose $g$ with **heavier tails** than $f$ in the region of interest. It's better to over-sample the tails than to miss them."""),

    md("m28", """### 4.4 Worked example: estimating a tail probability

Let $X \\sim N(0, 1)$. We want to estimate $P(X > 4) = E[\\mathbf{1}_{X > 4}]$.

This probability is extremely small ($\\approx 3.17 \\times 10^{-5}$). With naive Monte Carlo (sampling from $N(0,1)$ and counting how often $X > 4$), most samples contribute nothing — we need millions of draws for a decent estimate.

**Importance sampling idea:** Sample from $g = N(4, 1)$ (centred on the region of interest) and reweight."""),

    code("m29", """# Importance sampling: estimate P(X > 4) for X ~ N(0,1)
true_prob = stats.norm.sf(4)  # survival function = P(X > 4)
print(f"True P(X > 4) = {true_prob:.6e}")

n_samples = 10_000

# Method 1: Naive Monte Carlo
naive_samples = rng.standard_normal(n_samples)
naive_estimate = np.mean(naive_samples > 4)
print(f"\\nNaive MC ({n_samples:,} samples):   P̂ = {naive_estimate:.6e}")

# Method 2: Importance sampling with g = N(4, 1)
g_samples = rng.normal(loc=4, scale=1, size=n_samples)
h_values = (g_samples > 4).astype(float)  # h(x) = 1_{x > 4}
weights = stats.norm.pdf(g_samples) / stats.norm(loc=4, scale=1).pdf(g_samples)
is_estimate = np.mean(h_values * weights)
print(f"Importance ({n_samples:,} samples):   P̂ = {is_estimate:.6e}")"""),

    md("m30", """Let's see how the two estimators converge as we increase the number of samples. Importance sampling should converge much faster because it concentrates its samples in the region that matters."""),

    code("m31", """# Convergence comparison: naive vs importance sampling
n_values = np.arange(100, n_samples + 1, 50)

naive_running = np.cumsum(naive_samples > 4) / np.arange(1, n_samples + 1)
is_running = np.cumsum(h_values * weights) / np.arange(1, n_samples + 1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n_values, naive_running[n_values - 1], alpha=0.7, label="Naive Monte Carlo")
ax.plot(n_values, is_running[n_values - 1], alpha=0.7, label="Importance Sampling (g = N(4,1))")
ax.axhline(true_prob, color="k", ls="--", lw=1.5, label=f"True value = {true_prob:.2e}")

ax.set_xlabel("Number of samples")
ax.set_ylabel("Estimated P(X > 4)")
ax.set_title("Convergence: Naive MC vs Importance Sampling")
ax.legend()
ax.set_yscale("symlog", linthresh=1e-6)
plt.tight_layout()
plt.show()""", tags=["hide-input"]),

    md("m32", """### 4.5 Connection to Bayesian inference

Importance sampling is the foundation of several key algorithms in Bayesian computation:

- **Sequential Monte Carlo** (particle filters) use importance sampling to track a posterior distribution as data arrives sequentially.
- **PSIS-LOO** (Pareto-Smoothed Importance Sampling Leave-One-Out), used in ArviZ for model comparison (Module 08.03), is an importance sampling method.
- The transition from importance sampling to **MCMC** (Module 07) can be understood as moving from "reweight existing samples" to "generate new samples that are already from the target." """),

    # ═══ Section 5: Comparing Methods ═══
    md("m33", """---

## 5. Comparing the Three Methods

| | Inverse Transform | Accept-Reject | Importance Sampling |
|---|---|---|---|
| **Requires** | $F^{-1}$ in closed form | Proposal $g$ with $f \\leq Mg$ | Proposal $g$ with matching support |
| **Output** | Exact i.i.d. samples | Exact i.i.d. samples | Weighted samples (for expectations) |
| **Efficiency** | 100% (one uniform → one sample) | $1/M$ acceptance rate | Depends on weight variance |
| **Best for** | Standard distributions | Complex PDFs, bounded support | Tail probabilities, rare events |
| **Limitation** | Needs invertible CDF | Efficiency degrades in high dimensions | High weight variance if $g$ is poor |

All three methods start from the same building block: uniform random numbers. They differ in how they transform those uniforms into samples from the target. In Module 07, we'll see how **Markov Chain Monte Carlo** extends these ideas to sample from distributions that are only known up to a normalising constant — exactly the situation in Bayesian inference."""),

    # ═══ Section 6: Key Takeaways ═══
    md("m34", """---

## Key Takeaways

1. **The inverse transform method** converts $U \\sim \\text{Uniform}(0,1)$ into $X \\sim F$ via $X = F^{-1}(U)$. It is exact and efficient, but requires the CDF inverse in closed form.

2. **Acceptance-rejection sampling** works with the PDF directly: propose from an easy distribution, accept with probability proportional to the target-to-envelope ratio. The tighter the envelope, the higher the efficiency.

3. **Importance sampling** reweights samples from an easy distribution to estimate expectations under a hard-to-sample distribution. The choice of importance distribution critically affects variance.

4. **Efficiency matters.** A bad proposal in accept-reject (large $M$) or a bad importance distribution (high weight variance) can make these methods practically useless. This is the same challenge that motivates the sophisticated MCMC algorithms in Module 07.

5. **These are the foundations of computational statistics.** Every modern sampling algorithm — Metropolis-Hastings, Hamiltonian Monte Carlo, Sequential Monte Carlo — builds on the ideas in this notebook."""),

    # ═══ Section 7: Exercises ═══
    md("m35", """---

## Exercises

**Exercise 4.1 (Inverse transform for Cauchy).** The standard Cauchy distribution has CDF $F(x) = \\frac{1}{2} + \\frac{1}{\\pi} \\arctan(x)$. Derive $F^{-1}(u)$ and implement inverse transform sampling. Generate 10,000 samples, plot the histogram, and overlay the true PDF. Why is the histogram noisy even with so many samples? *(Hint: think about the variance of the Cauchy distribution.)*

**Exercise 4.2 (Accept-reject for semicircle).** The Wigner semicircle distribution on $[-1, 1]$ has PDF $f(x) = \\frac{2}{\\pi}\\sqrt{1 - x^2}$. Implement acceptance-rejection using a Uniform(-1, 1) proposal. What is $M$? What is the theoretical acceptance rate? Verify empirically.

**Exercise 4.3 (Importance sampling for rare events).** Estimate $P(X > 5)$ where $X \\sim N(0, 1)$ using (a) naive Monte Carlo with $n = 100{,}000$ and (b) importance sampling with $g = N(5, 1)$. Report the estimates and their standard errors. Which is more precise?

**Exercise 4.4 (Proposal design).** For the acceptance-rejection method applied to $\\text{Beta}(5, 2)$: (a) compute the optimal $M$ using a Uniform proposal; (b) compute $M$ using a Beta(4, 2) proposal; (c) compare the acceptance rates. Why is the Beta proposal so much better?"""),

    # ── Final cells ──
    code("m36", """cfg.save_gifs(clean=True)"""),

    md("m37", """---

**Next:** [Module 03 — Descriptive Statistics](../03_descriptive_stats/01_summary_statistics.ipynb) — Summarising data: measures of centre, spread, and shape."""),
]


# ═══════════════════════════════════════════════════════════════════
#  NOTEBOOK 2: Time Series Fundamentals
# ═══════════════════════════════════════════════════════════════════

ts_cells = [
    md("t1", """# Time Series Fundamentals

## Learning Objectives

- Understand what makes time series data fundamentally different from cross-sectional data.
- Define and test for **stationarity** — the key assumption underlying most time series models.
- Compute and interpret **autocorrelation** and **partial autocorrelation** functions.
- Fit and diagnose **AR**, **MA**, and **ARIMA** models using statsmodels.
- Produce and evaluate **forecasts** with confidence intervals.
- Recognise the connections between time series methods and the regression/Bayesian tools from earlier modules.

## Prerequisites

- Module 06 (Linear Models — regression, residual analysis)
- Module 04.03 (Maximum Likelihood Estimation)
- Familiarity with NumPy, matplotlib, pandas"""),

    code("t2", """import sys, os, shutil
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.system(
        "sudo apt-get update -qq && sudo apt-get install -y -qq "
        "libcairo2-dev libpango1.0-dev && pip install -q manim ipython==8.21.0"
    )

_miktex_bin = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64"
if _miktex_bin.exists() and str(_miktex_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + str(_miktex_bin)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, os.path.abspath("../../src"))
from amstats.plotting import apply_style

apply_style()

rng = np.random.default_rng(42)


class Cfg:
    root = Path("../../").resolve()
    gif_dir = root / "media" / "gifs"
    has_latex: bool = (
        shutil.which("latex") is not None or shutil.which("pdflatex") is not None
    )

    def __init__(self):
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        if not self.has_latex:
            print("⚠ LaTeX not found — MathTex will fall back to Text().")

    def apply_manim_config(self):
        from manim import config as mcfg
        mcfg.format = "gif"

    def math_text(self, expr, **kwargs):
        from manim import MathTex, Text
        if self.has_latex:
            return MathTex(expr, **kwargs)
        return Text(expr, **kwargs)

    def save_gifs(self, clean=True):
        local_media = Path("media")
        found = list(local_media.rglob("*.gif")) if local_media.exists() else []
        if not found:
            print("  No new GIFs to save.")
            return
        for gif in found:
            dest = self.gif_dir / gif.name
            shutil.copy2(gif, dest)
            print(f"  ✓ media/gifs/{gif.name}")
        if clean:
            for sub in ("videos", "images", "Tex"):
                d = local_media / sub
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            print("  Cleaned up local temp render files.")


cfg = Cfg()""", tags=["hide-input"]),

    code("t3", """from manim import *

cfg.apply_manim_config()
math_text = cfg.math_text

from amstats.manim_utils import C, COLORS""", tags=["hide-input"]),

    # ═══ Section 1: What Makes Time Series Special? ═══
    md("t4", """---

## 1. What Makes Time Series Special?

A **time series** is an ordered sequence of observations indexed by time:

$$\\{Y_t\\}_{t=1}^{T} = Y_1, Y_2, \\ldots, Y_T$$

This sounds deceptively simple — after all, Module 06's regression models also handle ordered data. But time series data has a fundamental property that changes everything: **temporal dependence**. Nearby observations are correlated with each other because they share common underlying causes, trends, and momentum.

In cross-sectional data (e.g., heights of 200 random people), observations are independent — knowing person 47's height tells you nothing about person 48's. In time series data (e.g., daily temperatures), knowing today's temperature tells you a lot about tomorrow's.

### 1.1 Three components of a time series

Most time series can be decomposed (conceptually or mathematically) into three components:

$$Y_t = T_t + S_t + R_t$$

where:
- $T_t$ is the **trend** — the long-run direction (upward, downward, or flat)
- $S_t$ is the **seasonality** — regular periodic patterns (daily, weekly, yearly)
- $R_t$ is the **residual** (noise) — the irregular fluctuations left over

The figure below shows a synthetic time series with all three components visible, then decomposes it into its parts."""),

    code("t5", """# Generate a synthetic time series with trend, seasonality, and noise
T = 200
t = np.arange(T)

trend = 0.05 * t                                    # linear upward trend
seasonality = 3 * np.sin(2 * np.pi * t / 25)        # period = 25
noise = rng.normal(0, 1, T)                          # random noise
y = trend + seasonality + noise

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

axes[0].plot(t, y, color="black", lw=1)
axes[0].set_title("Observed Time Series:  Y_t = Trend + Seasonality + Noise", fontweight="bold")
axes[0].set_ylabel("Y_t")

axes[1].plot(t, trend, color=C.CYAN, lw=2)
axes[1].set_title("Trend Component", fontweight="bold")
axes[1].set_ylabel("T_t")

axes[2].plot(t, seasonality, color=C.GOLD, lw=2)
axes[2].set_title("Seasonal Component (period = 25)", fontweight="bold")
axes[2].set_ylabel("S_t")

axes[3].plot(t, noise, color=C.SALMON, lw=1, alpha=0.8)
axes[3].set_title("Residual (Noise)", fontweight="bold")
axes[3].set_ylabel("R_t")
axes[3].set_xlabel("Time (t)")

plt.tight_layout()
plt.show()""", tags=["hide-input"]),

    md("t6", """The fundamental assumption behind all time series modelling is that **the past contains information about the future**. If the residuals were all that existed (pure noise), there would be nothing to model or forecast. It is the trend and seasonal structure — plus any additional serial correlation in the residuals — that makes time series analysis productive."""),

    # ═══ Section 2: Stationarity ═══
    md("t7", """---

## 2. Stationarity

Most time series models (AR, MA, ARIMA) assume a property called **stationarity**. Intuitively, a stationary process looks the same no matter when you observe it — its statistical properties don't drift over time.

### 2.1 Formal definitions

**Strict stationarity.** A time series $\\{Y_t\\}$ is strictly stationary if the joint distribution of $(Y_{t_1}, Y_{t_2}, \\ldots, Y_{t_k})$ is the same as $(Y_{t_1+h}, Y_{t_2+h}, \\ldots, Y_{t_k+h})$ for all choices of times and all shifts $h$.

This is a very strong condition. In practice, we use a weaker version:

**Weak (covariance) stationarity.** A time series is weakly stationary if:

1. **Constant mean:** $E[Y_t] = \\mu$ for all $t$
2. **Constant variance:** $\\text{Var}(Y_t) = \\sigma^2$ for all $t$
3. **Autocovariance depends only on lag:** $\\text{Cov}(Y_t, Y_{t+h}) = \\gamma(h)$ for all $t$

Condition 3 is the crucial one: the correlation between two observations depends only on how far apart they are in time ($h$), not on when they occur ($t$).

### 2.2 Why stationarity matters

If a process is non-stationary, its mean or variance changes over time. This means:
- The relationship between $Y_t$ and $Y_{t-1}$ changes depending on *when* you look
- Parameter estimates from one part of the series don't apply to another part
- Forecasts are unreliable because the model was fitted to statistics that no longer hold

**The strategy:** Transform non-stationary data into stationary data (typically by differencing), model the stationary residuals, then invert the transformation to produce forecasts."""),

    md("t8", """### 2.3 Stationary vs non-stationary: a visual comparison

The figure below shows two processes side by side:
- **Left:** a stationary AR(1) process — it fluctuates around a constant mean
- **Right:** a random walk $Y_t = Y_{t-1} + \\varepsilon_t$ — it wanders without returning to a mean

We also plot rolling mean and standard deviation to highlight the difference."""),

    code("t9", """# Compare stationary vs non-stationary processes
n = 300
eps = rng.normal(0, 1, n)

# Stationary: AR(1) with phi = 0.7
ar1 = np.zeros(n)
for t in range(1, n):
    ar1[t] = 0.7 * ar1[t - 1] + eps[t]

# Non-stationary: random walk
rw = np.cumsum(eps)

fig, axes = plt.subplots(2, 2, figsize=(13, 7))

window = 50
for col, (series, name) in enumerate([(ar1, "Stationary: AR(1), φ=0.7"),
                                       (rw, "Non-stationary: Random Walk")]):
    ax_ts, ax_stats = axes[0, col], axes[1, col]

    ax_ts.plot(series, lw=1, color="black")
    ax_ts.axhline(0, color="gray", ls="--", lw=0.8)
    ax_ts.set_title(name, fontweight="bold")
    ax_ts.set_ylabel("Y_t")

    roll_mean = pd.Series(series).rolling(window).mean()
    roll_std = pd.Series(series).rolling(window).std()
    ax_stats.plot(roll_mean, label=f"Rolling mean (w={window})", color=C.CYAN, lw=2)
    ax_stats.plot(roll_std, label=f"Rolling std (w={window})", color=C.SALMON, lw=2)
    ax_stats.legend(fontsize=9)
    ax_stats.set_xlabel("Time")
    ax_stats.set_ylabel("Value")

fig.suptitle("Rolling Statistics Reveal Non-Stationarity", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()""", tags=["hide-input"]),

    md("t10", """The AR(1) process has a rolling mean that stays near zero and a roughly constant rolling standard deviation. The random walk's rolling mean drifts arbitrarily, and its rolling standard deviation grows — classic signs of non-stationarity.

### 2.4 The Augmented Dickey-Fuller test

The **Augmented Dickey-Fuller (ADF)** test is a formal hypothesis test for stationarity:

- $H_0$: The series has a unit root (is non-stationary)
- $H_1$: The series is stationary

A **small p-value** (typically $< 0.05$) rejects $H_0$ and provides evidence for stationarity."""),

    code("t11", """# Augmented Dickey-Fuller test
for series, name in [(ar1, "AR(1)"), (rw, "Random Walk")]:
    result = adfuller(series, autolag="AIC")
    print(f"ADF test for {name}:")
    print(f"  Test statistic: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.6f}")
    print(f"  Conclusion:     {'Stationary ✓' if result[1] < 0.05 else 'Non-stationary ✗'}")
    print()"""),

    md("t12", """### 2.5 Differencing: making non-stationary data stationary

The most common transformation is **differencing**: replacing $Y_t$ with

$$\\Delta Y_t = Y_t - Y_{t-1}$$

For a random walk $Y_t = Y_{t-1} + \\varepsilon_t$, differencing gives $\\Delta Y_t = \\varepsilon_t$ — pure white noise, which is stationary. Sometimes we need to difference twice ($d = 2$) if the trend is quadratic, but $d = 1$ handles most cases."""),

    code("t13", """# Differencing the random walk
rw_diff = np.diff(rw)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(rw, lw=1, color="black")
axes[0].set_title("Random Walk (non-stationary)", fontweight="bold")
axes[0].set_ylabel("Y_t")

axes[1].plot(rw_diff, lw=1, color=C.EMERALD)
axes[1].axhline(0, color="gray", ls="--", lw=0.8)
axes[1].set_title("After Differencing: ΔY_t = Y_t − Y_{t−1} (stationary)", fontweight="bold")
axes[1].set_ylabel("ΔY_t")

for ax in axes:
    ax.set_xlabel("Time")

plt.tight_layout()
plt.show()

# Verify with ADF test
result = adfuller(rw_diff, autolag="AIC")
print(f"ADF test on differenced series: p-value = {result[1]:.6e} → Stationary ✓")""", tags=["hide-input"]),

    # ═══ Section 3: Autocorrelation ═══
    md("t14", """---

## 3. Autocorrelation

Now that we know what stationarity is, we can define the tools that describe the **memory structure** of a stationary time series.

### 3.1 The autocovariance and autocorrelation functions

For a weakly stationary series with mean $\\mu$ and variance $\\sigma^2$:

**Autocovariance function (ACVF):**
$$\\gamma(h) = \\text{Cov}(Y_t, Y_{t+h}) = E[(Y_t - \\mu)(Y_{t+h} - \\mu)]$$

**Autocorrelation function (ACF):**
$$\\rho(h) = \\frac{\\gamma(h)}{\\gamma(0)} = \\frac{\\text{Cov}(Y_t, Y_{t+h})}{\\text{Var}(Y_t)}$$

The ACF is just the normalised version of the ACVF, with $\\rho(0) = 1$ always.

### 3.2 Partial autocorrelation function (PACF)

The ACF at lag $h$ measures the *total* correlation between $Y_t$ and $Y_{t+h}$, including indirect effects through intermediate values $Y_{t+1}, \\ldots, Y_{t+h-1}$.

The **PACF** at lag $h$ measures the *direct* correlation between $Y_t$ and $Y_{t+h}$, after removing the linear effects of the intermediate values. Formally, it is the coefficient $\\phi_{hh}$ in the regression:

$$Y_{t+h} = \\phi_{h1} Y_{t+h-1} + \\phi_{h2} Y_{t+h-2} + \\cdots + \\phi_{hh} Y_t + \\varepsilon_t$$

**Intuition:**
- **ACF** tells you "how far does memory reach?" — it may decay slowly because of indirect chains of correlation.
- **PACF** tells you "what is the *direct* effect at each lag?" — it isolates the contribution of each specific lag."""),

    md("t15", """### 3.3 ACF and PACF in practice

The following animation builds up an ACF plot bar by bar for a simulated AR(2) process, showing at each lag how correlated the series is with its past. Significant lags (those outside the 95% confidence band) are highlighted."""),

    code("t16", """%%manim -qm -v WARNING ACFBuildUp


class ACFBuildUp(Scene):
    \"\"\"Build up an ACF plot bar by bar, highlighting significant lags.\"\"\"

    def construct(self):
        title = Text("Autocorrelation Function (ACF) — AR(2) Process", font_size=26).to_edge(UP)
        self.play(Write(title), run_time=0.5)

        # Generate AR(2) data: Y_t = 0.5 Y_{t-1} - 0.3 Y_{t-2} + eps
        n = 500
        np.random.seed(42)
        eps = np.random.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.5 * y[t - 1] - 0.3 * y[t - 2] + eps[t]

        from statsmodels.tsa.stattools import acf as acf_func
        acf_vals = acf_func(y, nlags=15, fft=True)
        conf = 1.96 / np.sqrt(n)

        axes = Axes(
            x_range=[0, 16, 1], y_range=[-0.5, 1.1, 0.25],
            x_length=10, y_length=4.5,
            axis_config={"include_numbers": True, "font_size": 14},
        ).shift(DOWN * 0.3)
        x_lbl = axes.get_x_axis_label(Text("Lag h", font_size=16), edge=DOWN, direction=DOWN)
        y_lbl = axes.get_y_axis_label(Text("ρ(h)", font_size=16), edge=LEFT, direction=LEFT)
        self.play(Create(axes), Write(x_lbl), Write(y_lbl), run_time=0.6)

        # Confidence band
        conf_upper = DashedLine(axes.c2p(0, conf), axes.c2p(16, conf),
                                color=GREY, stroke_width=1.5)
        conf_lower = DashedLine(axes.c2p(0, -conf), axes.c2p(16, -conf),
                                color=GREY, stroke_width=1.5)
        self.play(Create(conf_upper), Create(conf_lower), run_time=0.4)

        # Build bars one by one
        for h in range(16):
            val = acf_vals[h]
            is_sig = abs(val) > conf
            color = C.SALMON if is_sig else C.PERIWINKLE

            bar_height = val * (4.5 / 1.6)
            bar = Line(axes.c2p(h, 0), axes.c2p(h, val),
                       stroke_width=6, color=color)
            dot = Dot(axes.c2p(h, val), radius=0.06, color=color)

            speed = 0.3 if h < 5 else 0.12
            self.play(Create(bar), FadeIn(dot), run_time=speed)

        self.wait(1.5)""", tags=["hide-input"]),

    code("t17", """# ACF and PACF for the AR(2) process using statsmodels
ar2 = np.zeros(500)
eps_ar2 = rng.normal(0, 1, 500)
for t in range(2, 500):
    ar2[t] = 0.5 * ar2[t - 1] - 0.3 * ar2[t - 2] + eps_ar2[t]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(ar2, lags=20, ax=ax1, title="ACF of AR(2): φ₁=0.5, φ₂=−0.3")
plot_pacf(ar2, lags=20, ax=ax2, title="PACF of AR(2): φ₁=0.5, φ₂=−0.3", method="ywm")
plt.tight_layout()
plt.show()

print("Notice: PACF cuts off sharply after lag 2 — the signature of an AR(2) process.")
print("The ACF decays gradually because the indirect correlations persist through the chain.")""", tags=["hide-input"]),

    # ═══ Section 4: AR Models ═══
    md("t18", """---

## 4. Autoregressive (AR) Models

### 4.1 Definition

An **autoregressive model of order $p$**, denoted AR($p$), expresses each observation as a linear combination of its $p$ most recent values plus noise:

$$Y_t = c + \\phi_1 Y_{t-1} + \\phi_2 Y_{t-2} + \\cdots + \\phi_p Y_{t-p} + \\varepsilon_t$$

where $\\varepsilon_t \\sim N(0, \\sigma^2)$ is white noise and $\\phi_1, \\ldots, \\phi_p$ are the AR coefficients.

**Connection to regression (Module 06):** This is literally a regression of $Y_t$ on its own lagged values! The "auto" in autoregressive means "self-regressing." The design matrix has columns $Y_{t-1}, Y_{t-2}, \\ldots, Y_{t-p}$, and we estimate the $\\phi$ coefficients by least squares or MLE.

### 4.2 Stationarity conditions

An AR($p$) process is stationary if and only if all roots of the **characteristic polynomial**

$$1 - \\phi_1 z - \\phi_2 z^2 - \\cdots - \\phi_p z^p = 0$$

lie **outside the unit circle** in the complex plane ($|z| > 1$).

For AR(1), this simplifies to $|\\phi_1| < 1$. When $\\phi_1 = 1$, we get a random walk (non-stationary). When $|\\phi_1| > 1$, the process explodes.

### 4.3 ACF/PACF signature of AR models

This is the key diagnostic tool for identifying AR models from data:

| Model | ACF | PACF |
|---|---|---|
| AR($p$) | Decays gradually (exponential/oscillating) | Cuts off after lag $p$ |

The PACF cutoff makes intuitive sense: in an AR($p$) model, $Y_t$ depends *directly* on $Y_{t-1}, \\ldots, Y_{t-p}$ only. The PACF, which measures direct effects, should therefore be zero beyond lag $p$."""),

    md("t19", """### 4.4 The effect of $\\phi$: memory and persistence

The animation below shows how the AR(1) coefficient $\\phi$ controls the "memory" of the process. With $\\phi$ close to zero, each observation is nearly independent; with $\\phi$ close to 1, the process moves slowly and has long-range correlation."""),

    code("t20", """%%manim -qm -v WARNING ARPhiEffect


class ARPhiEffect(Scene):
    \"\"\"Show AR(1) realisations for different phi values.\"\"\"

    def construct(self):
        title = Text("AR(1): How φ Controls Memory", font_size=28).to_edge(UP)
        self.play(Write(title), run_time=0.5)

        phi_values = [0.2, 0.8, 0.95]
        colors = [C.EMERALD, C.GOLD, C.SALMON]
        n = 150

        axes = Axes(
            x_range=[0, n, 25], y_range=[-6, 6, 2],
            x_length=10, y_length=5,
            axis_config={"include_numbers": True, "font_size": 14},
        ).shift(DOWN * 0.3)
        x_lbl = axes.get_x_axis_label(Text("Time", font_size=16), edge=DOWN, direction=DOWN)
        self.play(Create(axes), Write(x_lbl), run_time=0.5)

        np.random.seed(42)
        eps = np.random.normal(0, 1, n)

        legend_items = VGroup()

        for phi, col in zip(phi_values, colors):
            y = np.zeros(n)
            for t in range(1, n):
                y[t] = phi * y[t - 1] + eps[t]

            points = [axes.c2p(t, y[t]) for t in range(n)]
            line = VMobject(color=col, stroke_width=2)
            line.set_points_smoothly(points)

            label = Text(f"φ = {phi}", font_size=18, color=col)
            legend_items.add(label)

            self.play(Create(line), run_time=1.2)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UR).shift(DOWN * 0.5)
        for item in legend_items:
            self.play(Write(item), run_time=0.2)

        self.wait(1.5)""", tags=["hide-input"]),

    # ── Fitting AR models ──
    md("t21", """### 4.5 Fitting AR models with statsmodels

Let's simulate an AR(2) process with known parameters and see how well we can recover them."""),

    code("t22", """# Simulate and fit AR(2): Y_t = 2 + 0.5*Y_{t-1} - 0.3*Y_{t-2} + eps
n = 1000
true_c, true_phi1, true_phi2, true_sigma = 2.0, 0.5, -0.3, 1.0

eps = rng.normal(0, true_sigma, n)
y = np.zeros(n)
y[0], y[1] = true_c, true_c
for t in range(2, n):
    y[t] = true_c + true_phi1 * y[t - 1] + true_phi2 * y[t - 2] + eps[t]

# Fit using statsmodels ARIMA(p=2, d=0, q=0) = AR(2)
model = ARIMA(y, order=(2, 0, 0))
result = model.fit()
print(result.summary().tables[1])
print(f"\\nTrue parameters:     c={true_c}, φ₁={true_phi1}, φ₂={true_phi2}, σ={true_sigma}")"""),

    # ═══ Section 5: MA Models ═══
    md("t23", """---

## 5. Moving Average (MA) Models

### 5.1 Definition

A **moving average model of order $q$**, denoted MA($q$), expresses each observation as a linear combination of the current and past $q$ noise terms:

$$Y_t = \\mu + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1} + \\theta_2 \\varepsilon_{t-2} + \\cdots + \\theta_q \\varepsilon_{t-q}$$

where $\\varepsilon_t \\sim N(0, \\sigma^2)$ is white noise. Unlike AR models, which regress on past *values*, MA models depend on past *shocks*.

**Intuition:** An MA($q$) process has a **finite memory** of exactly $q$ periods. After a shock $\\varepsilon_t$ occurs, it affects $Y_t, Y_{t+1}, \\ldots, Y_{t+q}$ and then vanishes completely.

### 5.2 ACF/PACF signature of MA models

| Model | ACF | PACF |
|---|---|---|
| MA($q$) | **Cuts off after lag $q$** | Decays gradually |

This is the mirror image of the AR pattern. The ACF cutoff makes sense: after $q$ lags, the shared $\\varepsilon$ terms have no overlap.

### 5.3 Invertibility

Just as AR models need stationarity (roots outside unit circle), MA models need **invertibility**: the roots of $1 + \\theta_1 z + \\cdots + \\theta_q z^q = 0$ must lie outside the unit circle. This ensures a unique representation and well-behaved estimation."""),

    code("t24", """# Simulate MA(2) and check ACF/PACF signature
theta1, theta2 = 0.7, -0.4
eps_ma = rng.normal(0, 1, 500)
ma2 = np.zeros(500)
for t in range(2, 500):
    ma2[t] = eps_ma[t] + theta1 * eps_ma[t - 1] + theta2 * eps_ma[t - 2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(ma2, lags=20, ax=ax1, title="ACF of MA(2): θ₁=0.7, θ₂=−0.4")
plot_pacf(ma2, lags=20, ax=ax2, title="PACF of MA(2): θ₁=0.7, θ₂=−0.4", method="ywm")
plt.tight_layout()
plt.show()

print("Notice: ACF cuts off after lag 2 — the signature of an MA(2) process.")
print("The PACF decays gradually — the mirror image of what we saw for AR(2).")""", tags=["hide-input"]),

    # ═══ Section 6: ARIMA ═══
    md("t25", """---

## 6. ARIMA Models

### 6.1 From ARMA to ARIMA

An **ARMA($p, q$)** model combines both AR and MA components:

$$Y_t = c + \\sum_{i=1}^{p} \\phi_i Y_{t-i} + \\varepsilon_t + \\sum_{j=1}^{q} \\theta_j \\varepsilon_{t-j}$$

This handles stationary data. For non-stationary data, we first difference $d$ times and then fit an ARMA to the differenced series. This is the **ARIMA($p, d, q$)** model:

- $p$ = order of the AR component (number of lagged values)
- $d$ = degree of differencing (number of times we subtract $Y_{t-1}$ from $Y_t$)
- $q$ = order of the MA component (number of lagged noise terms)

### 6.2 The Box-Jenkins methodology

George Box and Gwilym Jenkins (1970) established the standard workflow for ARIMA modelling:

**Step 1 — Identify:**
- Plot the series. Is there a trend? → differencing needed ($d \\geq 1$)
- Apply ADF test. Still non-stationary after differencing? → increase $d$
- Examine ACF and PACF of the (differenced) series:
  - PACF cuts off at lag $p$ → AR($p$) component
  - ACF cuts off at lag $q$ → MA($q$) component
  - Both decay gradually → mixed ARMA, use information criteria

**Step 2 — Estimate:**
- Fit the ARIMA($p, d, q$) model via MLE. `statsmodels.tsa.arima.model.ARIMA` does this.
- Compare candidate models using **AIC** (Akaike Information Criterion) or **BIC** (Bayesian Information Criterion) — these penalise model complexity, exactly as we discussed in Module 06.05.

**Step 3 — Diagnose:**
- Check residuals: they should look like white noise
  - ACF of residuals should show no significant lags
  - **Ljung-Box test**: $H_0$: residuals are white noise
- If residuals show structure, revise the model (increase $p$, $q$, or $d$)"""),

    md("t26", """### 6.3 Worked example: the full Box-Jenkins pipeline

Let's generate a synthetic ARIMA(1,1,1) process and walk through the entire identification → estimation → diagnostics pipeline."""),

    code("t27", """# Generate ARIMA(1,1,1) data
# First create stationary ARMA(1,1), then integrate (cumsum) to make it non-stationary
n = 400
true_phi, true_theta = 0.6, 0.4
eps_arima = rng.normal(0, 1, n)

# ARMA(1,1) for the differenced series
z = np.zeros(n)
for t in range(1, n):
    z[t] = true_phi * z[t - 1] + eps_arima[t] + true_theta * eps_arima[t - 1]

# Integrate to get ARIMA(1,1,1)
y_arima = np.cumsum(z) + 50  # add offset for realism

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(y_arima, lw=1, color="black")
ax1.set_title("ARIMA(1,1,1) — Simulated (non-stationary)", fontweight="bold")
ax1.set_ylabel("Y_t")

y_diff = np.diff(y_arima)
ax2.plot(y_diff, lw=1, color=C.EMERALD)
ax2.axhline(0, color="gray", ls="--", lw=0.8)
ax2.set_title("After differencing: ΔY_t (stationary)", fontweight="bold")
ax2.set_ylabel("ΔY_t")
ax2.set_xlabel("Time")

plt.tight_layout()
plt.show()

# ADF test on original and differenced
adf_orig = adfuller(y_arima, autolag="AIC")
adf_diff = adfuller(y_diff, autolag="AIC")
print(f"ADF on original series: p = {adf_orig[1]:.4f} → {'Stationary' if adf_orig[1] < 0.05 else 'Non-stationary'}")
print(f"ADF on differenced:     p = {adf_diff[1]:.6f} → {'Stationary' if adf_diff[1] < 0.05 else 'Non-stationary'}")
print("\\n→ d = 1 differencing is sufficient.")""", tags=["hide-input"]),

    code("t28", """# Step 1: Identify p and q from ACF/PACF of the differenced series
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(y_diff, lags=20, ax=ax1, title="ACF of ΔY_t")
plot_pacf(y_diff, lags=20, ax=ax2, title="PACF of ΔY_t", method="ywm")
plt.tight_layout()
plt.show()

print("Both ACF and PACF decay gradually → suggests a mixed ARMA model.")
print("We'll compare ARIMA(1,1,0), ARIMA(0,1,1), and ARIMA(1,1,1) using AIC.")""", tags=["hide-input"]),

    code("t29", """# Step 2: Estimate — compare candidate models via AIC
candidates = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]

print(f"{'Model':<20s} {'AIC':>10s} {'BIC':>10s}")
print("-" * 42)
best_aic, best_order = np.inf, None
for order in candidates:
    try:
        m = ARIMA(y_arima, order=order).fit()
        aic, bic = m.aic, m.bic
        marker = ""
        if aic < best_aic:
            best_aic = aic
            best_order = order
            marker = " ← best"
        print(f"ARIMA{order!s:<14s} {aic:10.1f} {bic:10.1f}{marker}")
    except Exception as e:
        print(f"ARIMA{order!s:<14s} {'failed':>10s}")

print(f"\\nSelected: ARIMA{best_order}")"""),

    code("t30", """# Step 3: Diagnose — fit the best model and check residuals
best_model = ARIMA(y_arima, order=best_order).fit()
print(best_model.summary().tables[1])
print(f"\\nTrue parameters: φ = {true_phi}, θ = {true_theta}")"""),

    code("t31", """# Residual diagnostics
resid = best_model.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time series of residuals
axes[0, 0].plot(resid, lw=0.8, color="black")
axes[0, 0].axhline(0, color="gray", ls="--")
axes[0, 0].set_title("Residuals Over Time")
axes[0, 0].set_ylabel("Residual")

# Histogram
axes[0, 1].hist(resid, bins=40, density=True, alpha=0.7)
x_norm = np.linspace(resid.min(), resid.max(), 100)
axes[0, 1].plot(x_norm, stats.norm(resid.mean(), resid.std()).pdf(x_norm), "k-", lw=2)
axes[0, 1].set_title("Residual Distribution")

# ACF of residuals
plot_acf(resid, lags=20, ax=axes[1, 0], title="ACF of Residuals")

# QQ plot
stats.probplot(resid, plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot")

fig.suptitle("Residual Diagnostics — Should Look Like White Noise", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
print("Ljung-Box test (H₀: residuals are white noise):")
print(lb.to_string())""", tags=["hide-input"]),

    # ═══ Section 7: Forecasting ═══
    md("t32", """---

## 7. Forecasting and Evaluation

### 7.1 Producing forecasts

Once we have a fitted ARIMA model, we can forecast future values. The model produces:
- **Point forecasts**: the conditional expectation $\\hat{Y}_{T+h} = E[Y_{T+h} \\mid Y_1, \\ldots, Y_T]$
- **Prediction intervals**: reflecting the uncertainty, which grows with the forecast horizon

### 7.2 Evaluation metrics

To assess forecast quality, we use:

$$\\text{MAE} = \\frac{1}{H} \\sum_{h=1}^{H} |Y_{T+h} - \\hat{Y}_{T+h}|$$

$$\\text{RMSE} = \\sqrt{\\frac{1}{H} \\sum_{h=1}^{H} (Y_{T+h} - \\hat{Y}_{T+h})^2}$$

$$\\text{MAPE} = \\frac{100}{H} \\sum_{h=1}^{H} \\left|\\frac{Y_{T+h} - \\hat{Y}_{T+h}}{Y_{T+h}}\\right|$$

### 7.3 Walk-forward validation

In cross-sectional data, we use $k$-fold cross-validation. For time series, we **cannot randomly split** the data because that would break the temporal order. Instead, we use **walk-forward validation**:

1. Train on $Y_1, \\ldots, Y_t$
2. Forecast $\\hat{Y}_{t+1}$
3. Expand the training set to include $Y_{t+1}$
4. Repeat

This respects the temporal structure and gives an honest estimate of out-of-sample performance."""),

    code("t33", """# Forecasting: train on first 350 points, forecast the last 50
train = y_arima[:350]
test = y_arima[350:]

model_fc = ARIMA(train, order=best_order).fit()
forecast = model_fc.get_forecast(steps=len(test))
fc_mean = forecast.predicted_mean
fc_ci = forecast.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(12, 5))
t_train = np.arange(len(train))
t_test = np.arange(len(train), len(y_arima))

ax.plot(t_train[-100:], train[-100:], "k-", lw=1, label="Training data")
ax.plot(t_test, test, "k--", lw=1.5, label="Actual (held out)")
ax.plot(t_test, fc_mean, color=C.CYAN, lw=2, label="Forecast")
ax.fill_between(t_test, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                color=C.CYAN, alpha=0.2, label="95% prediction interval")
ax.axvline(len(train), color="gray", ls=":", lw=1)
ax.set_xlabel("Time")
ax.set_ylabel("Y_t")
ax.set_title(f"ARIMA{best_order} Forecast vs Actual")
ax.legend()
plt.tight_layout()
plt.show()

# Compute metrics
mae = np.mean(np.abs(test - fc_mean))
rmse = np.sqrt(np.mean((test - fc_mean) ** 2))
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")""", tags=["hide-input"]),

    code("t34", """# Walk-forward validation
start_idx = 250
forecasts_wf = []
actuals_wf = []

for t in range(start_idx, len(y_arima) - 1):
    train_wf = y_arima[:t + 1]
    model_wf = ARIMA(train_wf, order=best_order).fit()
    fc = model_wf.forecast(steps=1)
    forecasts_wf.append(fc.iloc[0])
    actuals_wf.append(y_arima[t + 1])

forecasts_wf = np.array(forecasts_wf)
actuals_wf = np.array(actuals_wf)

fig, ax = plt.subplots(figsize=(12, 5))
t_wf = np.arange(start_idx + 1, len(y_arima))
ax.plot(t_wf, actuals_wf, "k-", lw=1, label="Actual")
ax.plot(t_wf, forecasts_wf, color=C.SALMON, lw=1.5, alpha=0.8, label="1-step forecast")
ax.set_xlabel("Time")
ax.set_ylabel("Y_t")
ax.set_title("Walk-Forward Validation: 1-Step-Ahead Forecasts")
ax.legend()
plt.tight_layout()
plt.show()

wf_mae = np.mean(np.abs(actuals_wf - forecasts_wf))
wf_rmse = np.sqrt(np.mean((actuals_wf - forecasts_wf) ** 2))
print(f"Walk-forward MAE  = {wf_mae:.3f}")
print(f"Walk-forward RMSE = {wf_rmse:.3f}")""", tags=["hide-input"]),

    # ═══ Section 8: Connections ═══
    md("t35", """---

## 8. Connections to the Broader Course

Time series methods connect deeply to topics you've already studied:

- **AR models are regression** (Module 06) with lagged values as predictors. The OLS estimator for AR coefficients is a special case of linear regression.
- **MLE** (Module 04.03) is how `statsmodels` fits ARIMA models. The log-likelihood of a Gaussian ARIMA model has a specific form involving the innovation variance.
- **Model selection via AIC/BIC** (Module 06.05) applies directly — we used it above to choose between candidate ARIMA orders.
- **Bayesian time series:** In Modules 07–08, you learned Bayesian regression. Bayesian structural time series models (BSTS) put priors on trend, seasonality, and regression coefficients. PyMC can fit these models directly.
- **State-space models** generalise ARIMA — they express the same models in a recursive form that naturally handles missing data, time-varying parameters, and multiple time series. This connects to hidden Markov models (a possible future Module 12 topic)."""),

    # ═══ Key Takeaways ═══
    md("t36", """---

## Key Takeaways

1. **Time series data is fundamentally different** from cross-sectional data because observations are temporally dependent. This dependence is what makes modelling and forecasting possible.

2. **Stationarity** is the key assumption: constant mean, constant variance, and autocovariance that depends only on lag. Non-stationary data must be transformed (typically by differencing) before modelling.

3. **ACF and PACF** are the primary diagnostic tools. AR($p$) models have PACF that cuts off at lag $p$; MA($q$) models have ACF that cuts off at lag $q$.

4. **ARIMA($p, d, q$)** unifies differencing, autoregression, and moving averages into a flexible framework. The Box-Jenkins methodology (identify → estimate → diagnose) is the standard workflow.

5. **Forecasting** requires walk-forward validation — you cannot randomly split time series data. Prediction intervals grow with the forecast horizon because uncertainty compounds.

6. **Time series methods build on regression and MLE** from earlier modules. Bayesian extensions (structural time series, state-space models) connect to Module 07–08."""),

    # ═══ Exercises ═══
    md("t37", """---

## Exercises

**Exercise 1.1 (Simulating and identifying).** Simulate 500 observations from each of the following processes. For each, plot the ACF and PACF and explain how you would identify the correct model order from the plots alone:
- (a) AR(1) with $\\phi = 0.9$
- (b) MA(1) with $\\theta = -0.5$
- (c) ARMA(1,1) with $\\phi = 0.6, \\theta = 0.3$

**Exercise 1.2 (Real data).** Download a real-world time series dataset (e.g., monthly airline passengers, daily stock prices, or weekly COVID cases). Apply the full Box-Jenkins pipeline: test for stationarity, difference if needed, examine ACF/PACF, fit candidate ARIMA models, select via AIC, and diagnose residuals.

**Exercise 1.3 (Forecast evaluation).** Using the airline passengers dataset (or your choice), split the data 80/20 into train/test. Fit ARIMA models of varying orders and compare their out-of-sample RMSE. Does more complex always mean better?

**Exercise 1.4 (Connection to regression).** Fit an AR(2) model to simulated data using (a) `statsmodels.tsa.arima.model.ARIMA` and (b) `statsmodels.OLS` with manually constructed lagged columns. Compare the coefficient estimates. Why are they not exactly the same? *(Hint: think about the likelihood vs OLS and the treatment of initial values.)*"""),

    # ── Final cells ──
    code("t38", """cfg.save_gifs(clean=True)"""),

    md("t39", """---

**Next:** [Module 13 — Digital Chemistry](../13_digital_chemistry/01_qsar_molecular_property_prediction.ipynb) — Applying statistical and machine learning methods to molecular data."""),
]


# ═══════════════════════════════════════════════════════════════════
#  WRITE BOTH NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════

def fix_sources(cells):
    """Ensure each source line ends with \\n (notebook convention)."""
    for cell in cells:
        src = cell["source"]
        fixed = []
        for i, line in enumerate(src):
            if i < len(src) - 1 and not line.endswith("\n"):
                fixed.append(line + "\n")
            else:
                fixed.append(line)
        cell["source"] = fixed
    return cells


mc_path = ROOT / "notebooks" / "02_distributions" / "04_monte_carlo_sampling.ipynb"
ts_path = ROOT / "notebooks" / "12_advanced_topics" / "01_time_series_fundamentals.ipynb"

mc_nb = nb(fix_sources(mc_cells))
ts_nb = nb(fix_sources(ts_cells))

mc_path.parent.mkdir(parents=True, exist_ok=True)
ts_path.parent.mkdir(parents=True, exist_ok=True)

with open(mc_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(mc_nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"✓ {mc_path}")

with open(ts_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(ts_nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"✓ {ts_path}")
