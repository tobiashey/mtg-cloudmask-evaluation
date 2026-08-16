"""Abbildung 5.4a: Effektive Wert-Aenderungsintervalle je Quelle (fig54_update_intervals).

Horizontale Balken der empirischen Aenderungsintervalle (F2,
out/f2_update_intervalle.csv) mit dem 10-Minuten-Entscheidungstakt der
Automation als einzigem Rot-Akzent. Quellenfarben gemaess Richtlinie.

In die Kennzahl gehen nur Slot-Paare mit gueltiger Messung auf beiden Seiten
ein (siehe f2_update_intervalle.py); Ausfaelle und ungueltige Kamerabilder
erzeugen also keine kuenstlichen Wertwechsel.

Aufruf:  python fig54_update_intervals.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

LABELS = {
    "camera": ("sky camera (ground truth)", "#009e73"),
    "satellite_clm": ("MTG FCI L2 CLM", "#0072b2"),
    "weather_brightsky": ("Bright Sky (DWD)", "#e69f00"),
    "weather_openmeteo": ("Open-Meteo", "#999999"),
    "weather_openweathermap": ("OpenWeatherMap", "#999999"),
    "weather_tomorrowio": ("Tomorrow.io", "#999999"),
    "weather_weatherapi": ("WeatherAPI.com", "#999999"),
}

CM = 1 / 2.54


def main() -> None:
    f2 = pd.read_csv(OUT_DIR / "f2_update_intervalle.csv")
    f2 = f2.sort_values("eff_interval_min", ascending=False)
    y = np.arange(len(f2))

    plt.rcParams.update(
        {
            "font.family": "serif",
            # Liberation Serif ist metrisch kompatibel und dient als Fallback,
            # falls Times New Roman auf dem Rechner nicht installiert ist.
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
        }
    )

    fig, ax = plt.subplots(figsize=(12.2 * CM, 5.2 * CM), constrained_layout=True)
    colors = [LABELS[s][1] for s in f2["source"]]
    ax.barh(y, f2["eff_interval_min"], color=colors, height=0.62)
    for yi, val in zip(y, f2["eff_interval_min"]):
        ax.text(val + 1.2, yi, f"{val:.1f} min", ha="left", va="center",
                fontsize=7.0, color="#3a3a3a")

    # Entscheidungstakt der Automation: der eine Rot-Akzent der Abbildung
    ax.axvline(10, color="#c1272d", linewidth=0.9, linestyle=":")
    ax.text(10.8, len(f2) - 0.42, "automation cycle (10 min)", fontsize=6.5,
            style="italic", color="#c1272d", ha="left", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[s][0] for s in f2["source"]])
    ax.set_xlim(0, 85)
    ax.set_xlabel("mean interval between new values [min]")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig54_update_intervals.{suffix}", dpi=300)
    print(f"OK -> {FIGURES_DIR / 'fig54_update_intervals.{png,pdf}'}")


if __name__ == "__main__":
    main()
