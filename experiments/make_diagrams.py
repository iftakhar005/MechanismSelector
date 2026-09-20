"""Draw the architecture and workflow diagrams as PNG (and PDF for LaTeX).

Layout is explicit: every box is placed on a 16x10 grid, arrows run in reserved
corridors, and nothing overlaps. Run after changing either diagram.

    .venv/Scripts/python.exe experiments/make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "presentation"

INK = "#12171c"
MUTED = "#55606b"
LINE = "#c3cabf"
PAPER = "#f6f7f4"
PANEL = "#ffffff"
BAND = "#eef1ee"
ACCENT = "#1f5a82"
SKIP = "#2a78d6"
NUDGE = "#d9602e"
REBUILD = "#15986a"
GUARD = "#c98a00"
ARROW = "#3c4650"


def box(ax, x, y, w, h, title=None, lines=(), fill=PANEL, edge=LINE, lw=1.4,
        title_color=INK, body_color=MUTED, title_size=11.5, body_size=9.5, radius=0.09):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    n = (1 if title else 0) + len(lines)
    step = 0.30 if body_size <= 9.5 else 0.34
    top = y + h / 2 + (n - 1) * step / 2
    if title:
        ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=title_size,
                fontweight="600", color=title_color, zorder=3)
        top -= step
    for ln in lines:
        ax.text(x + w / 2, top, ln, ha="center", va="center", fontsize=body_size,
                color=body_color, zorder=3)
        top -= step


def band(ax, x, y, w, h, label, edge=LINE, lw=1.4, fill="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))
    ax.text(x + 0.22, y + h - 0.26, label, ha="left", va="center", fontsize=10,
            fontweight="600", color=ACCENT, zorder=3)


def arrow(ax, path, color=ARROW, lw=1.8, style="-|>", dashed=False):
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        last = (x1, y1) == path[-1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style if last else "-",
                                     mutation_scale=14, color=color, linewidth=lw,
                                     linestyle="--" if dashed else "-",
                                     shrinkA=0, shrinkB=0, zorder=4))


def canvas(w=16, h=10):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, facecolor=PAPER, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {OUT / (name + '.png')}")


# --------------------------------------------------------------------------- architecture
def architecture() -> None:
    fig, ax = canvas()
    ax.text(0.4, 9.62, "MechanismSelector — runtime architecture", fontsize=19,
            fontweight="600", color=INK)
    ax.text(0.4, 9.28, "One policy, one model family, one stream. Everything runs per row, in original time order.",
            fontsize=11, color=MUTED)

    # ---- band 1: the stream loop
    band(ax, 0.4, 7.80, 15.2, 1.20, "1 · STREAM LOOP   runner.py")
    w, y, h = 3.4, 7.95, 0.62
    xs = [0.75, 4.55, 8.35, 12.15]
    box(ax, xs[0], y, w, h, "Stream", ["rows in time order, never shuffled"], fill=BAND)
    box(ax, xs[1], y, w, h, "Predict the row", ["before its label is used"])
    box(ax, xs[2], y, w, h, "Buffer", ["last 1,000 rows · min 100 to act"])
    box(ax, xs[3], y, w, h, "Drift detector", ["ADWIN δ=0.002, or DDM"])
    for a, b in zip(xs, xs[1:]):
        arrow(ax, [(a + w, y + h / 2), (b - 0.02, y + h / 2)], lw=1.6)

    # ---- band 2: the decision
    band(ax, 1.05, 2.95, 8.75, 4.25, "2 · DECISION   selector.py — adapt()")
    bx, bw = 1.35, 4.05
    rows = [
        (6.20, 0.62, "Split the window by time", ["older 80% train · newest 20% holdout"], PANEL, LINE),
        (5.30, 0.62, "Window under ~250 rows?", ["holdout < 50 or train < 100"], "#fbf1dc", GUARD),
        (4.40, 0.62, "Model above the floor?", ["floor = reference − 0.02"], "#e8f0fb", SKIP),
        (3.20, 0.90, "Nudge a copy, test on the holdout", ["trained on the older 80% only",
                                                           "kept nudge never sees the newest 20%"], "#fbeee7", NUDGE),
    ]
    for y0, hh, t, ls, f, e in rows:
        box(ax, bx, y0, bw, hh, t, ls, fill=f, edge=e)
    for (y0, _, *_), (y1, h1, *_) in zip(rows, rows[1:]):
        arrow(ax, [(bx + bw / 2, y0), (bx + bw / 2, y1 + h1 + 0.02)], lw=1.5)

    ox, ow = 5.95, 3.55
    box(ax, ox, 5.30, ow, 0.62, "REBUILD  (guard)", ["half of all its rebuilds"],
        fill=GUARD, edge=GUARD, title_color="#241900", body_color="#3c2c00")
    box(ax, ox, 4.40, ow, 0.62, "SKIP", ["zero cost"], fill=SKIP, edge=SKIP,
        title_color="white", body_color="#e4eefb")
    box(ax, ox, 3.72, ow, 0.42, "NUDGE — keep it", [], fill=NUDGE, edge=NUDGE, title_color="white")
    box(ax, ox, 3.20, ow, 0.42, "REBUILD — pay for both", [], fill=REBUILD, edge=REBUILD, title_color="white")
    for y0, lbl in [(5.61, "yes"), (4.71, "yes"), (3.93, "clears"), (3.41, "fails")]:
        arrow(ax, [(bx + bw, y0), (ox - 0.02, y0)], lw=1.5)
        ax.text((bx + bw + ox) / 2, y0 + 0.15, lbl, ha="center", fontsize=8.5, color=MUTED)

    # ---- band 3: actions and measurement
    band(ax, 10.15, 2.95, 5.45, 3.35, "3 · ACTION & MEASUREMENT")
    box(ax, 10.45, 4.58, 4.85, 1.27, "mechanisms.py — nudge per family",
        ["XGBoost +5 rounds · RF +5 trees, retire 5", "SGD / GaussianNB: one partial_fit",
         "4% of a rebuild (trees) · 0.6–1% SGD · 80% GNB"], fill="#fbeee7", edge=NUDGE)
    box(ax, 10.45, 3.82, 4.85, 0.64, "Rebuild — fresh model on the full window",
        ["handed no old model to inherit from"], fill="#e6f4ee", edge=REBUILD, title_size=10.5)
    box(ax, 10.45, 3.00, 2.35, 0.78, "accounting.py", ["work_units =", "passes × rows"], fill=BAND, body_size=9)
    box(ax, 12.95, 3.00, 2.35, 0.78, "footprint.py", ["size · node visits", "per row"], fill=BAND, body_size=9)
    arrow(ax, [(ox + ow, 3.93), (10.43, 4.85)], color=ACCENT, lw=1.5, dashed=True)

    # ---- band 4: outputs
    band(ax, 0.4, 0.70, 15.2, 1.70, "4 · WHAT IS WRITTEN DOWN")
    ow4, y4, h4 = 3.55, 0.90, 1.10
    box(ax, 0.75, y4, ow4, h4, "AdaptResult", ["model to use next + exact cost",
                                               "buffer cleared, reference re-measured"], fill="#eaf0f5", edge="#9fbdd2")
    box(ax, 4.70, y4, ow4, h4, "grid.csv", ["one row per run", "500 runs per detector"])
    box(ax, 8.65, y4, ow4, h4, "events/", ["one row per alarm", "why each decision was made"])
    box(ax, 12.05, y4, 3.55, h4, "analysis.py", ["Friedman · Nemenyi · Wilcoxon+Holm",
                                                 "n = 60 blocks, n = 20 check"])

    # ---- corridors
    arrow(ax, [(13.85, 7.95), (13.85, 7.50), (9.90, 7.50), (9.90, 7.22)], lw=1.8)
    ax.text(11.9, 7.62, "alarm + buffer ≥ 100 rows", ha="center", fontsize=9, color=ARROW)
    arrow(ax, [(7.72, 2.95), (7.72, 2.62), (2.53, 2.62), (2.53, 2.02)], lw=1.5)
    ax.text(5.1, 2.74, "outcome + exact cost", ha="center", fontsize=9, color=ARROW)
    arrow(ax, [(0.75, 1.45), (0.55, 1.45), (0.55, 7.50), (6.25, 7.50), (6.25, 7.93)],
          color=ACCENT, lw=1.8)
    ax.text(3.6, 7.62, "new model returns to the loop", ha="center", fontsize=9, color=ACCENT)

    save(fig, "architecture")


# --------------------------------------------------------------------------- workflow
def workflow() -> None:
    fig, ax = canvas(16, 10.5)
    ax.text(0.4, 10.1, "How the project was built — what we did, in order", fontsize=19,
            fontweight="600", color=INK)
    ax.text(0.4, 9.76, "Each phase had a gate that had to pass before the next one started. Nothing was tuned after results were seen.",
            fontsize=11, color=MUTED)

    phases = [
        ("1", "Data layer", "Load 5 drift streams in original time order, never shuffled.",
         "GATE · all five loaders return correctly shaped arrays", "datasets.py · elec2, insects ×3, covtype", False),
        ("2", "Cost accounting", "work_units = passes × rows, read from the fitted model.",
         "GATE · a 300-tree fit costs exactly 30× a 10-tree fit", "accounting.py · energy logged, never decisive", False),
        ("3", "Repair mechanisms", "Skip, nudge and rebuild for four model families.",
         "GATE · nudge cheaper than rebuild; forest size constant", "mechanisms.py · 4% · 0.6–1% · 80%", False),
        ("4", "The selector", "Try the cheap option, verify on recent rows, escalate if it fails.",
         "GATE · holdout rows never appear in training data", "selector.py · the contribution", False),
        ("5", "Baseline policies", "Four competitors with identical actions.",
         "GATE · all five policies run end to end", "policies.py · FixedSchedule(k=3) is the one to beat", False),
        ("6", "Experiment runner", "Stream, detect, adapt, log every run and every alarm.",
         "GATE · full grid completes — 500 runs, 0 failures", "runner.py · 5 × 4 × 5 × 5", False),
        ("7", "Analysis", "Friedman, Nemenyi, Wilcoxon + Holm, Pareto, 11 figures.",
         "GATE · determinism verified before collapsing to n = 60", "analysis.py · n = 20 kept as a stricter check", False),
        ("7b", "Second detector — DDM", "Repeat all 500 runs with a one-sided detector.",
         "RESULT · it falsified our own explanation, and we kept it", "ddm_contrast.py · replay reproduced exactly", True),
        ("8", "Packaging", "Ship the selector; README states the negative results first.",
         "GATE · clean install; 251 tests pass against the package", "package/ · 0.1.0 · defects documented", False),
    ]

    top, row_h, gap = 9.25, 0.74, 0.20
    spine = 0.95
    ax.plot([spine, spine], [top - len(phases) * (row_h + gap) + gap + 0.1, top - 0.1],
            color=LINE, lw=2.5, zorder=1)

    for i, (num, title, body, gate, tool, hot) in enumerate(phases):
        y = top - (i + 1) * (row_h + gap) + gap
        colour = GUARD if hot else ACCENT
        ax.add_patch(plt.Circle((spine, y + row_h / 2), 0.20, color=colour, zorder=3))
        ax.text(spine, y + row_h / 2, num, ha="center", va="center", fontsize=9.5,
                fontweight="700", color="white" if not hot else "#241900", zorder=4)

        ax.add_patch(FancyBboxPatch((1.45, y), 7.0, row_h, boxstyle="round,pad=0,rounding_size=0.09",
                                    facecolor="#fbf7ec" if hot else PANEL,
                                    edgecolor=colour if hot else LINE,
                                    linewidth=2.0 if hot or num == "4" else 1.4, zorder=2))
        ax.text(1.70, y + row_h * 0.66, title, fontsize=12.5, fontweight="600", color=INK, zorder=3)
        ax.text(1.70, y + row_h * 0.28, body, fontsize=9.5, color=MUTED, zorder=3)

        ax.add_patch(FancyBboxPatch((8.70, y), 6.9, row_h, boxstyle="round,pad=0,rounding_size=0.09",
                                    facecolor="#fbf1dc" if hot else BAND,
                                    edgecolor=GUARD if hot else LINE, linewidth=1.4, zorder=2))
        ax.text(8.95, y + row_h * 0.66, gate, fontsize=9.5, fontweight="600",
                color="#6d4a00" if hot else ACCENT, zorder=3)
        ax.text(8.95, y + row_h * 0.28, tool, fontsize=9.5, color="#6d4a00" if hot else MUTED, zorder=3)

    y_last = top - len(phases) * (row_h + gap) + gap
    ax.add_patch(FancyBboxPatch((0.4, y_last - 1.20), 15.2, 0.98,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=PANEL, edgecolor=LINE, linewidth=1.4, zorder=2))
    ax.text(0.65, y_last - 0.40, "RUNNING ALONGSIDE EVERY PHASE", fontsize=10,
            fontweight="600", color=ACCENT, zorder=3)
    ax.text(0.65, y_last - 0.72,
            "251 tests — invariants · exact cost · chunked prediction equals row-by-row   ·   verify_*.py phase demos",
            fontsize=9.5, color=MUTED, zorder=3)
    ax.text(0.65, y_last - 1.00,
            "diagnostics that answered what the grid could not: nudge_window_diagnostic.py · nudge_overwrite_demo.py · placeholder_sanity.py",
            fontsize=9.5, color=MUTED, zorder=3)

    save(fig, "workflow")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    workflow()
