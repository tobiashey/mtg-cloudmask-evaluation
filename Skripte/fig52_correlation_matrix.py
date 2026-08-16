"""Abbildung 5.2d: Inter-Quellen-Korrelationsmatrix (fig52_correlation_matrix).

Paarweise Pearson-Korrelation aller sieben Quellen (D5) als annotierte
Heatmap — untere Dreiecksmatrix inkl. Diagonale, Werte direkt in den Zellen
(kein Colorbar noetig, keine Punktwolken). Werte werden NICHT neu berechnet,
sondern aus out/d5_pearson_matrix.csv uebernommen (Reproduzierbarkeit G1).
Styling gemaess Abbildungsrichtlinie: sequenzielle Blau-Skala von sehr
hellem Blaugrau (#e4ecf3) zum Quellen-Blau (#0072b2) — helligkeitsmonoton,
damit in Graustufen unterscheidbar; Textfarbe kippt bei dunklen Zellen auf
Weiss.

Aufruf:  python fig52_correlation_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

# Reihenfolge + englische Anzeige-Labels konsistent zu den uebrigen Abbildungen
ORDER = [
    ("Camera", "sky camera"),
    ("MTG CLM", "MTG FCI L2 CLM"),
    ("Bright Sky (DWD)", "Bright Sky (DWD)"),
    ("Open-Meteo", "Open-Meteo"),
    ("OpenWeatherMap", "OpenWeatherMap"),
    ("Tomorrow.io", "Tomorrow.io"),
    ("WeatherAPI.com", "WeatherAPI.com"),
]

CM = 1 / 2.54


def main() -> None:
    m = pd.read_csv(OUT_DIR / "d5_pearson_matrix.csv", index_col=0)
    keys = [k for k, _ in ORDER]
    labels = [lbl for _, lbl in ORDER]
    m = m.loc[keys, keys]
    n = len(keys)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
        }
    )

    cmap = LinearSegmentedColormap.from_list("blues_lni", ["#e4ecf3", "#0072b2"])

    fig, ax = plt.subplots(figsize=(11.0 * CM, 8.2 * CM), constrained_layout=True)
    vmin, vmax = 0.15, 1.0
    for i in range(n):
        for j in range(n):
            if j > i:
                continue  # obere Dreieckshaelfte weglassen (symmetrisch)
            val = m.iloc[i, j]
            frac = (val - vmin) / (vmax - vmin)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=cmap(frac),
                                       edgecolor="white", linewidth=0.8))
            ax.text(j, i, f"{val:.2f}".replace("0.", "."),
                    ha="center", va="center", fontsize=7.5,
                    color="white" if frac > 0.62 else "#3a3a3a")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # Zeile 0 oben
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig52_correlation_matrix.{suffix}", dpi=300)
    print(f"Pearson-r-Matrix ({n}x{n}) -> "
          f"{FIGURES_DIR / 'fig52_correlation_matrix.{png,pdf}'}")


if __name__ == "__main__":
    main()
