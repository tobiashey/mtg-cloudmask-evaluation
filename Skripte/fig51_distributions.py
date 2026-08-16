"""Abbildung 5.1: Verteilungs-Panel des Bewoelkungsgrads je Quelle (fig51).

Erzeugt das Histogramm-Panel aller sieben Quellen auf dem Complete-Case-
Datensatz (N = 2 646, [E5]) gemaess der Abbildungsrichtlinie der Arbeit:
- Endbreite 12,2 cm (LNI-Satzspiegel), Times New Roman, Labels englisch,
  kein Abbildungstitel (Information gehoert in die Caption).
- Quellenfarben fest: Kamera gruen #009e73, Satellit blau #0072b2,
  DWD/Bright Sky orange #e69f00, uebrige Quellen grau #999999.
- Export als PDF (Vektor, Einbindung) + PNG (300 dpi, Vorschau) nach
  Satelitten/figures/.

Deterministisch; liest ausschliesslich out/paired_slots.csv
(Ausgabe von c2_paired_slots.py gegen die eingefrorene Analyse-DB).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
PAIRED_SLOTS = AUSWERTUNG_DIR / "out" / "paired_slots.csv"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

# Reihenfolge = Leselogik von 5.1/5.2: Bodenwahrheit, Artefakt, Baseline,
# dann sekundaere APIs. Farben gemaess Richtlinie Abschnitt 4.
SOURCES: list[tuple[str, str, str]] = [
    ("camera", "Sky camera\n(ground truth)", "#009e73"),
    ("satellite_clm", "MTG FCI L2 CLM", "#0072b2"),
    ("weather_brightsky", "Bright Sky (DWD)", "#e69f00"),
    ("weather_openmeteo", "Open-Meteo", "#999999"),
    ("weather_openweathermap", "OpenWeatherMap", "#999999"),
    ("weather_tomorrowio", "Tomorrow.io", "#999999"),
    ("weather_weatherapi", "WeatherAPI.com", "#999999"),
]

BINS = np.arange(0, 101, 10)  # 10-pp-Bins wie in der bedingten Analyse (D3)

CM = 1 / 2.54  # matplotlib rechnet in Zoll


def main() -> None:
    df = pd.read_csv(PAIRED_SLOTS)
    df = df[df["complete_case"] == 1]
    n = len(df)

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
        2,
        4,
        figsize=(12.2 * CM, 7.6 * CM),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    # Leeres Panel oben rechts statt unten rechts: So traegt in jeder Spalte
    # genau das unterste Panel die x-Achsenbeschriftung — einheitlich obere
    # Reihe ohne, untere Reihe mit Beschriftung.
    panel_idx = [0, 1, 2, 4, 5, 6, 7]
    axes[3].set_visible(False)

    for idx, (col, label, color) in zip(panel_idx, SOURCES):
        ax = axes[idx]
        # Absolute Anzahl je Bin: durch den Complete-Case-Ansatz haben alle
        # Panels identisches N -> direkt vergleichbar und intuitiver als Anteile.
        ax.hist(
            df[col].to_numpy(),
            bins=BINS,
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.text(
            0.04,
            0.94,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            # weisser Hintergrund, damit das Label auch ueber hohen Balken
            # lesbar bleibt (WeatherAPI: 0-%-Bin > 60 %)
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.6},
        )
        ax.set_ylim(0, 2000)
        ax.set_yticks([0, 500, 1000, 1500])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.3)
        ax.set_axisbelow(True)
        ax.set_xticks([0, 50, 100])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[4:8]:
        ax.set_xlabel("cloud cover [%]")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("number of measurements")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES_DIR / f"fig51_distributions.{suffix}",
            dpi=300,
        )
    print(f"N = {n} Complete-Case-Slots -> {FIGURES_DIR / 'fig51_distributions.{png,pdf}'}")


if __name__ == "__main__":
    main()
