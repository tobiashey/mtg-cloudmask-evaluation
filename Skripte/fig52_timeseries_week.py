"""Abbildung 5.2b: Beispielwoche als Zeitreihen-Overlay (fig52_timeseries_week).

Slot-aufgeloestes Overlay Kamera vs. MTG CLM vs. Bright Sky fuer die
Beispielwoche 06.–12.07.2026 (wie d2_plots.py: enthaelt klare, durchbrochene
und bedeckte Lagen). Nachtluecken brechen die Linien (Reindex auf das volle
10-min-Raster). Linienstile gemaess Abbildungsrichtlinie:
Kamera gruen strichpunktiert, Satellit blau durchgezogen, DWD orange
gestrichelt — in Graustufen ueber die Linienstile unterscheidbar.

Aufruf:  python fig52_timeseries_week.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

WEEK_START = "2026-07-06"
WEEK_END = "2026-07-13"  # exklusiv

SERIES = [
    ("camera", "sky camera (ground truth)", "#009e73", "-."),
    ("satellite_clm", "MTG FCI L2 CLM", "#0072b2", "-"),
    ("weather_brightsky", "Bright Sky (DWD)", "#e69f00", "--"),
]

CM = 1 / 2.54


def main() -> None:
    df = pd.read_csv(OUT_DIR / "paired_slots.csv", parse_dates=["ts"])
    df = df[(df["ts"] >= WEEK_START) & (df["ts"] < WEEK_END)].set_index("ts")
    # Volles 10-min-Raster: fehlende Slots werden NaN -> Linienbruch (Nacht).
    grid = pd.date_range(df.index.min().floor("D"),
                         df.index.max().ceil("D"), freq="10min", tz="UTC")
    df = df.reindex(grid)

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

    fig, ax = plt.subplots(figsize=(12.2 * CM, 5.2 * CM), constrained_layout=True)
    for col, label, color, ls in SERIES:
        ax.plot(df.index, df[col], color=color, linestyle=ls, linewidth=1.0,
                label=label)

    ax.set_ylim(0, 100)
    ax.set_ylabel("cloud cover [%]")
    ax.set_xlabel("time [UTC]")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.3)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d Jul"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=7.0, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig52_timeseries_week.{suffix}", dpi=300)
    print(f"Woche {WEEK_START}..{WEEK_END} -> "
          f"{FIGURES_DIR / 'fig52_timeseries_week.{png,pdf}'}")


if __name__ == "__main__":
    main()
