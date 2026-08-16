"""Abbildung 5.2a: Scatter-Panel Quelle vs. Bodenwahrheit (fig52_scatter_sources).

2x3-Panel: je Quelle Streudiagramm gegen die Kamera-Bodenwahrheit (v106) auf
dem Complete-Case-Datensatz, mit Identitaetslinie und MAE/r-Annotation (Werte
aus out/d1_ff1_haupttabelle.csv — keine Neuberechnung, Reproduzierbarkeit G1).
Styling gemaess Abbildungsrichtlinie (12,2 cm, Times, englisch, Quellfarben;
Punktwolke rasterisiert mit 300 dpi, Achsen/Text bleiben Vektor).

Aufruf:  python fig52_scatter_sources.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

SOURCES = [
    ("satellite_clm", "MTG FCI L2 CLM", "#0072b2"),
    ("weather_brightsky", "Bright Sky (DWD)", "#e69f00"),
    ("weather_openmeteo", "Open-Meteo", "#999999"),
    ("weather_openweathermap", "OpenWeatherMap", "#999999"),
    ("weather_tomorrowio", "Tomorrow.io", "#999999"),
    ("weather_weatherapi", "WeatherAPI.com", "#999999"),
]

CM = 1 / 2.54


def main() -> None:
    df = pd.read_csv(OUT_DIR / "paired_slots.csv")
    df = df[df["complete_case"] == 1]
    d1 = pd.read_csv(OUT_DIR / "d1_ff1_haupttabelle.csv").set_index("source")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )

    fig, axes = plt.subplots(
        2, 3, figsize=(12.2 * CM, 8.6 * CM), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, (col, label, color) in zip(axes, SOURCES):
        ax.plot([0, 100], [0, 100], color="#5a5a5a", linewidth=0.7,
                linestyle="--", zorder=1)
        ax.scatter(df["camera"], df[col], s=2.5, color=color, alpha=0.20,
                   linewidths=0, rasterized=True, zorder=2)
        mae = d1.loc[label, "MAE"]
        r = d1.loc[label, "Pearson r"]
        ax.text(0.04, 0.97, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.0,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.85, "pad": 0.6})
        ax.text(0.96, 0.04, f"MAE {mae:.1f} pp\nr = {r:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.85, "pad": 0.6})
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[3:]:
        ax.set_xlabel("camera cloud cover [%]")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("source cloud cover [%]")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig52_scatter_sources.{suffix}", dpi=300)
    print(f"N = {len(df)} -> {FIGURES_DIR / 'fig52_scatter_sources.{png,pdf}'}")


if __name__ == "__main__":
    main()
