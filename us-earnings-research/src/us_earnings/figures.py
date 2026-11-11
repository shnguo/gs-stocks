"""Portable figures for the executed notebook; no browser chart dependencies."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .research import ARM_LABELS

COLORS = {"spy": "#282b30", "universe": "#9a8a62", "price": "#929aa5",
          "earnings": "#285b9a", "guidance": "#b66c26", "ml": "#a94777"}
STYLES = {"spy": "--", "universe": ":", "price": "-.",
          "earnings": "-", "guidance": "--", "ml": "-"}


def style():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#30353b", "text.color": "#30353b",
                         "axes.titleweight": "bold", "axes.grid": True,
                         "grid.color": "#e5e7eb", "grid.linewidth": 0.6,
                         "legend.frameon": False, "font.family": "DejaVu Sans"})


def equity_figure(result: dict, out: Path):
    style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for arm in ["spy", "universe", "price", "earnings", "guidance", "ml"]:
        curve = result["curves"][arm]
        sliced = curve.loc["2023-01-01":]
        prior = curve.loc[:"2022-12-31", "nav"]
        rebased = sliced.nav / (prior.iloc[-1] if len(prior) else 1.0)
        axes[0].plot(sliced.index, rebased, color=COLORS[arm], ls=STYLES[arm],
                     lw=1.7, label=ARM_LABELS[arm])
        peak = np.maximum.accumulate(np.r_[1.0, rebased.to_numpy()])[1:]
        axes[1].plot(sliced.index, rebased / peak - 1, color=COLORS[arm], ls=STYLES[arm], lw=1.5)
    axes[0].set_title("Portfolio comparison · 2023–2025", loc="left", pad=16)
    axes[0].set_ylabel("Wealth per $1 at the start of 2023")
    axes[0].legend(ncols=3, loc="upper left", fontsize=9)
    axes[1].set_ylabel("Drawdown from peak")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(sliced.index[0], sliced.index[-1])
    axes[1].set_xticks([sliced.index[0], np.datetime64("2024-01-02"),
                       np.datetime64("2025-01-02"), sliced.index[-1]])
    axes[1].set_xticklabels(["Jan 2023", "Jan 2024", "Jan 2025", "Dec 2025"])
    fig.savefig(out / "portfolio-comparison.png", bbox_inches="tight")
    return fig


def event_figure(result: dict, out: Path):
    style()
    rows = result["intervals"].loc[result["intervals"].period.str.contains("later evaluation")]
    fig, ax = plt.subplots(figsize=(11, 4.3), layout="constrained")
    labels = []
    for i, row in enumerate(rows.to_dict("records")):
        arm = row["filter"]
        labels.append(f"{ARM_LABELS[arm]}\n{row['events']} events · {row['months']} signal months")
        if row["mean_excess"] is not None and np.isfinite(row["mean_excess"]):
            available = row["lower_95"] is not None and np.isfinite(row["lower_95"])
            if available:
                ax.plot([row["lower_95"], row["upper_95"]], [i, i], color=COLORS[arm], lw=2)
            ax.scatter([row["mean_excess"]], [i], color=COLORS[arm], s=45, zorder=3)
            label = f"{row['mean_excess']:+.2%}" + (" · interval unavailable" if not available else "")
            ax.annotate(label, (row["mean_excess"], i),
                        xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10)
    ax.axvline(0, color="#555b63", lw=1, ls="--")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - 0.6, -0.6)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Mean stock return minus same-date SPY return · 20 sessions")
    ax.set_title("Event comparison · 2024–2025", loc="left", pad=20)
    fig.text(0.01, -0.025, "95% percentile bootstrap intervals, resampling signal months. Equal event weights; "
             "no stops. Overlapping observations remain dependent.", fontsize=9, color="#616872")
    fig.savefig(out / "event-comparison.png", bbox_inches="tight")
    return fig


def exposure_figure(result: dict, out: Path):
    style()
    rows = result["summary"].loc[result["summary"].period.str.contains("later evaluation")]
    rows = rows.set_index("arm").loc[["spy", "universe", "price", "earnings", "guidance", "ml"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    y = np.arange(len(rows))
    axes[0].barh(y, rows.average_exposure, color=[COLORS[a] for a in rows.index], height=0.55)
    axes[0].set_yticks(y, rows.label)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.12)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Average invested exposure · 2024–2025", loc="left", pad=15)
    axes[0].set_xlabel("Share of portfolio value")
    for i, value in enumerate(rows.average_exposure):
        axes[0].text(value + 0.015, i, f"{value:.1%}", va="center", fontsize=9)
    ledger = result["ledgers"]["guidance"]
    counts = ledger.loc[ledger.entry_date >= "2024-01-01"].groupby("ticker").size().sort_values()
    if not counts.empty:
        axes[1].barh(counts.index, counts, color=COLORS["guidance"], height=0.55)
        axes[1].set_xlim(0, counts.max() + 1)
        axes[1].xaxis.get_major_locator().set_params(integer=True)
    axes[1].set_title("Guidance trades by issuer · 2024–2025", loc="left", pad=15)
    axes[1].set_xlabel("Number of entries")
    fig.savefig(out / "exposure-and-concentration.png", bbox_inches="tight")
    return fig
