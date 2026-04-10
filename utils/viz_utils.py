"""Consistent visualization helpers for matplotlib and plotly."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "website", "figures"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style defaults
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CATEGORY_COLORS = {
    "breaking_news": "#E74C3C",
    "controversy":   "#F39C12",
    "product_launch": "#3498DB",
    "disaster":      "#8E44AD",
    "meme_viral":    "#2ECC71",
}


def save_fig(fig, name: str, tight=True):
    """Save a matplotlib figure to the website figures directory."""
    path = os.path.join(FIGURES_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def save_plotly(fig, name: str):
    """Save a plotly figure as HTML for embedding."""
    path = os.path.join(FIGURES_DIR, name)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Saved {path}")
