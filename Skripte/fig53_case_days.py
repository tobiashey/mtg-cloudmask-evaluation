"""Abbildung 5.3c: Fallbeispiel-Tage 19. und 21.07. (fig53_case_days).

Zwei gestapelte Tagesverlaeufe (Kamera / MTG CLM / Bright Sky) mit der
Entscheidungsschwelle 40 % als einzigem Rot-Akzent der Abbildung. Zeigt beide
Fehlertypen: die CLM-Overcast-Episode am 19.07. mittags, die DWD-Traegheit am
19.07. nachmittags und die unnoetige Beschattung der Baseline am 21.07. bei
bedecktem Himmel. Daten: out/paired_slots.csv (Tagfenster-Slots).

Aufruf:  python fig53_case_days.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

DAYS = ["2026-07-19", "2026-07-21"]
SERIES = [
    ("camera", "sky camera (ground truth)", "#009e73", "-."),
    ("satellite_clm", "MTG FCI L2 CLM", "#0072b2", "-"),
    ("weather_brightsky", "Bright Sky (DWD)", "#e69f00", "--"),
]
THRESHOLD = 40.0

CM = 1 / 2.54


def main() -> None:
    df = pd.read_csv(OUT_DIR / "paired_slots.csv", parse_dates=["ts"])
    df = df[df["daylight"] == 1]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.6,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(12.2 * CM, 8.0 * CM), sharey=True,
                             constrained_layout=True)
    for ax, day in zip(axes, DAYS):
        d = df[df["date"] == day]
        for col, label, color, ls in SERIES:
            ax.plot(d["ts"], d[col], color=color, linestyle=ls, linewidth=1.0,
                    label=label if day == DAYS[0] else None)
        # Entscheidungsschwelle: der eine Rot-Akzent der Abbildung
        ax.axhline(THRESHOLD, color="#c1272d", linewidth=0.8, linestyle=":")
        ax.text(1.0, THRESHOLD + 2, "decision threshold 40 %",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6.5, style="italic", color="#c1272d")
        ax.text(0.01, 0.96, pd.Timestamp(day).strftime("%d Jul %Y"),
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.85, "pad": 0.6})
        ax.set_ylim(0, 103)
        ax.set_ylabel("cloud cover [%]")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.3)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[1].set_xlabel("time [UTC]")
    axes[0].legend(frameon=False, fontsize=7.0, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, 1.22))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig53_case_days.{suffix}", dpi=300)
    print(f"OK -> {FIGURES_DIR / 'fig53_case_days.{png,pdf}'}")


if __name__ == "__main__":
    main()
