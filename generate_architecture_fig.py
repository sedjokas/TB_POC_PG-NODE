"""Generate the PG-NODE architecture figure using matplotlib only."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib.patheffects as pe

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.2)
ax.axis("off")

# ── colour palette ──────────────────────────────────────────
C_MECH  = "#ddeeff"   # light blue  — mechanistic boxes
C_NN    = "#fff0dd"   # light amber — neural-net boxes
C_EDGE  = "#222222"   # near-black  — box borders
C_NARR  = "#1a4a8a"   # dark blue   — NN / dashed arrows
C_TXT   = "#111111"

def box(ax, cx, cy, w, h, label, color, edgecolor=C_EDGE, fontsize=9.5, bold=False):
    """Draw a rounded rectangle centred at (cx, cy)."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=edgecolor, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color=C_TXT, fontweight=weight, zorder=4,
            multialignment="center")
    return (cx, cy, w, h)

def circle(ax, cx, cy, r, label):
    c = Circle((cx, cy), r, facecolor="white", edgecolor=C_EDGE,
               linewidth=1.5, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_TXT, zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#333333", dashed=False, lw=1.6,
          arrowstyle="->", shrink=0.18):
    style = dict(arrowstyle=arrowstyle, color=color,
                 lw=lw, mutation_scale=14,
                 connectionstyle="arc3,rad=0.0")
    if dashed:
        style["linestyle"] = "dashed"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=style, zorder=2)

def bent_arrow(ax, pts, color="#333333", dashed=False, lw=1.6):
    """Multi-segment arrow: list of (x,y) waypoints, last is arrowhead."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ls = "--" if dashed else "-"
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, linestyle=ls,
            solid_capstyle="round", zorder=2)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=14), zorder=2)

# ── layout constants ─────────────────────────────────────────
BW, BH = 1.85, 0.72   # box width / height
NW, NH = 1.85, 0.72   # NN box
RW = 0.40             # circle radius

ROW_TOP = 3.5    # y for top row (mechanistic, solver, pred, obs, loss)
ROW_MID = 2.0    # y for neural net box
ROW_BOT = 0.8    # y for backprop box

X = {
    "state":  1.2,
    "mech":   3.3,
    "nn":     3.3,
    "sum":    5.1,
    "solver": 6.9,
    "pred":   8.7,
    "obs":    8.7,
    "loss":  10.5,
    "bp":     6.9,
}

# ── draw boxes ───────────────────────────────────────────────
box(ax, X["state"], ROW_TOP, BW, BH,
    "State $\\mathbf{x}(t)$\n$S, L, I, T, \\ldots$", C_MECH)

box(ax, X["mech"],  ROW_TOP, BW, BH,
    "Mechanistic ODE\n$f_{\\mathrm{mech}}(\\mathbf{x},\\boldsymbol{\\theta})$",
    C_MECH, bold=False)

box(ax, X["nn"],    ROW_MID, NW, NH,
    "Neural Network\n$f_{\\mathrm{NN}}(\\mathbf{x},t;\\mathbf{W})$",
    C_NN, edgecolor=C_NARR, fontsize=9.5)

circle(ax, X["sum"], (ROW_TOP + ROW_MID)/2, RW, "$+$")

box(ax, X["solver"], ROW_TOP, BW, BH,
    "ODE Solver\n(RK4 / adjoint)", C_MECH)

box(ax, X["pred"],  ROW_TOP, BW, BH,
    "Predictions\n$\\hat{\\mathbf{x}}(t_1,\\ldots,t_T)$", C_MECH)

box(ax, X["obs"],   ROW_MID, BW, NH,
    "Observations\n$\\mathbf{y}_{1:T}$", "#f0f8f0", edgecolor="#448844")

box(ax, X["loss"],  ROW_TOP, BW, BH,
    "Loss  $\\mathcal{L}$\n+ physics penalty", "#fff0f0", edgecolor="#aa3333")

box(ax, X["bp"],    ROW_BOT, BW, BH,
    "Backprop\n$\\nabla_{\\boldsymbol{\\theta},\\mathbf{W}}\\mathcal{L}$",
    C_NN, edgecolor=C_NARR)

# ── arrows (forward pass) ────────────────────────────────────
# State → Mech
arrow(ax, X["state"]+BW/2, ROW_TOP,
          X["mech"]-BW/2,  ROW_TOP)
# State → NN  (diagonal down)
bent_arrow(ax, [(X["state"]+BW/2, ROW_TOP),
                (X["state"]+BW/2, ROW_MID),
                (X["nn"]-NW/2,    ROW_MID)])
# Mech → Sum
arrow(ax, X["mech"]+BW/2, ROW_TOP,
          X["sum"]-RW,     ROW_TOP)
# NN → Sum
arrow(ax, X["nn"]+NW/2, ROW_MID,
          X["sum"]+RW*0.05, ROW_MID)
# Sum → Solver
mid_y = (ROW_TOP + ROW_MID)/2
arrow(ax, X["sum"]+RW, mid_y,
          X["solver"]-BW/2, ROW_TOP)
# Solver → Pred
arrow(ax, X["solver"]+BW/2, ROW_TOP,
          X["pred"]-BW/2,   ROW_TOP)
# Obs → Pred  (vertical)
arrow(ax, X["obs"], ROW_MID+NH/2,
          X["pred"], ROW_TOP-BH/2)
# Pred → Loss
arrow(ax, X["pred"]+BW/2, ROW_TOP,
          X["loss"]-BW/2,  ROW_TOP)

# ── dashed backprop arrows ───────────────────────────────────
# Loss → Backprop (down-left L-shape)
bent_arrow(ax,
    [(X["loss"], ROW_TOP-BH/2),
     (X["loss"], ROW_BOT),
     (X["bp"]+BW/2, ROW_BOT)],
    color=C_NARR, dashed=True)
# Backprop → Mech (up-left)
bent_arrow(ax,
    [(X["bp"]-BW/2, ROW_BOT),
     (X["mech"],    ROW_BOT),
     (X["mech"],    ROW_TOP-BH/2)],
    color=C_NARR, dashed=True)

# ── section labels ───────────────────────────────────────────
ax.text(X["mech"], 4.62, "Mechanistic (known physics)",
        ha="center", fontsize=8.5, color="#1a4a8a",
        fontstyle="italic")
ax.text(X["nn"],   ROW_MID-0.68, "Neural residual (unknown dynamics)",
        ha="center", fontsize=8.5, color="#8a4a00",
        fontstyle="italic")
ax.text((X["loss"]+X["bp"])/2 + 0.55, (ROW_TOP+ROW_BOT)/2,
        "gradient\nflow", ha="left", va="center",
        fontsize=8, color=C_NARR, fontstyle="italic")

# ── legend ───────────────────────────────────────────────────
leg = [
    mpatches.Patch(facecolor=C_MECH, edgecolor=C_EDGE, label="Mechanistic component"),
    mpatches.Patch(facecolor=C_NN,   edgecolor=C_NARR, label="Neural-network component"),
    mpatches.Patch(facecolor="#f0f8f0", edgecolor="#448844", label="Observations"),
    mpatches.Patch(facecolor="#fff0f0", edgecolor="#aa3333", label="Loss"),
    plt.Line2D([0],[0], color="#333", lw=1.6, label="Forward pass"),
    plt.Line2D([0],[0], color=C_NARR, lw=1.6, ls="--", label="Backpropagation"),
]
ax.legend(handles=leg, loc="lower left", fontsize=8.5,
          bbox_to_anchor=(0.0, 0.0), ncol=3, framealpha=0.9)

ax.set_title("PG-NODE Architecture: Mechanistic SLIT + Neural Residual $f_{\\mathrm{NN}}$",
             fontsize=11, pad=10)

plt.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(OUTDIR, f"fig0_pgnode_architecture.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
plt.close(fig)
