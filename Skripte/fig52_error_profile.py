"""Abbildung 5.2c: Fehlerprofil ueber den Bewoelkungsgrad (fig52_error_profile).

MAE und Bias je 10-pp-Bin der Bodenwahrheit (Bin-Logik identisch zu
d3_bedingt.py; Werte gegen out/d3_fehlerprofil_bins.csv abgleichbar — beim
Lauf wird der 0-10-Bin des CLM als Anker geprueft). Zwei gestapelte Panels
mit gemeinsamer x-Achse. Styling gemaess Abbildungsrichtlinie; die vier
sekundaeren APIs in Grau mit unterschiedlichen Markern (s/w-tauglich).

Aufruf:  python fig52_error_profile.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

SOURCES = [
    # (Spalte, Label, Farbe, Linienstil, Marker)
    ("satellite_clm", "MTG FCI L2 CLM", "#0072b2", "-", "o"),
    ("weather_brightsky", "Bright Sky (DWD)", "#e69f00", "--", "s"),
    ("weather_openmeteo", "Open-Meteo", "#999999", ":", "^"),
    ("weather_openweathermap", "OpenWeatherMap", "#999999", ":", "D"),
    ("weather_tomorrowio", "Tomorrow.io", "#999999", ":", "v"),
    ("weather_weatherapi", "WeatherAPI.com", "#999999", ":", "x"),
]

BIN_EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0001])
BIN_CENTERS = np.arange(5, 100, 10)

CM = 1 / 2.54


def main() -> None:
    df = pd.read_csv(OUT_DIR / "paired_slots.csv")
    df = df[df["complete_case"] == 1].copy()
    df["bin"] = pd.cut(df["camera"], bins=BIN_EDGES, right=False,
                       labels=BIN_CENTERS)

    mae = {}
    bias = {}
    for col, *_ in SOURCES:
        err = df[col] - df["camera"]
        grp = err.groupby(df["bin"], observed=True)
        mae[col] = grp.apply(lambda e: e.abs().mean())
        bias[col] = grp.mean()

    # Anker gegen das d3-Artefakt (CLM 0-10-Bin MAE 6,87 pp)
    anker = mae["satellite_clm"].iloc[0]
    assert abs(anker - 6.87) < 0.05, f"Bin-Logik weicht von d3 ab: {anker:.2f}"

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

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.2 * CM, 8.8 * CM), sharex=True,
        constrained_layout=True,
    )
    for col, label, color, ls, marker in SOURCES:
        ax1.plot(BIN_CENTERS, mae[col], color=color, linestyle=ls,
                 marker=marker, markersize=3, linewidth=1.0, label=label)
        ax2.plot(BIN_CENTERS, bias[col], color=color, linestyle=ls,
                 marker=marker, markersize=3, linewidth=1.0)

    ax2.axhline(0, color="#5a5a5a", linewidth=0.6)
    # Bewusst beschnittene Skalen (Autor-Entscheid 10.08.): WeatherAPI.com
    # laeuft aus dem Bild, dafuer sind die Unterschiede der uebrigen Quellen
    # besser aufgeloest; Randnotizen nennen die abgeschnittenen Maximalwerte.
    ax1.set_ylim(0, 50)
    ax1.set_yticks(np.arange(0, 51, 10))
    ax2.set_ylim(-40, 30)
    ax2.set_yticks(np.arange(-40, 31, 10))
    ax1.text(0.99, 0.97, "WeatherAPI.com continues to 78 pp",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6.5,
             style="italic", color="#5a5a5a")
    ax2.text(0.99, 0.03, "WeatherAPI.com continues to −78 pp",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=6.5,
             style="italic", color="#5a5a5a")
    ax1.set_ylabel("MAE [pp]")
    ax2.set_ylabel("bias [pp]")
    ax2.set_xlabel("camera cloud cover [%]")
    ax2.set_xticks(np.arange(0, 101, 20))
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    ax1.legend(frameon=False, fontsize=7.0, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.28))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig52_error_profile.{suffix}", dpi=300)
    print(f"Anker CLM 0-10: {anker:.2f} pp -> "
          f"{FIGURES_DIR / 'fig52_error_profile.{png,pdf}'}")


if __name__ == "__main__":
    main()
