"""Abbildung 5.4b: Ausfall-Zeitleiste je Quelle (fig54_gap_timeline).

Markiert je Quelle alle 10-min-Raster-Slots des Analysezeitraums
(02.07.–04.08.), an denen KEIN Wert vorliegt (out/paired_slots.csv).
Kamera nur im Tagfenster bewertet (nachts wird konstruktionsbedingt nicht
aufgenommen); Satellit zeigt den Stand NACH Backfill — der zehntaegige
Live-Ausfall ist geschlossen und erscheint hier nicht (Fallstudie im Text).
Anzahl fehlender Slots am rechten Rand.

Aufruf:  python fig54_gap_timeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

ROWS = [
    ("camera", "sky camera (day window)", "#009e73", True),
    ("satellite_clm", "MTG FCI L2 CLM (post-backfill)", "#0072b2", False),
    ("weather_brightsky", "Bright Sky (DWD)", "#e69f00", False),
    ("weather_openmeteo", "Open-Meteo", "#999999", False),
    ("weather_openweathermap", "OpenWeatherMap", "#999999", False),
    ("weather_tomorrowio", "Tomorrow.io", "#999999", False),
    ("weather_weatherapi", "WeatherAPI.com", "#999999", False),
]

CM = 1 / 2.54


CAMPAIGN_END = "2026-08-04T14:00:00+00:00"  # nominelles Kampagnenende [E4];
# spaetere Raster-Slots stammen aus dem geplanten Abschalt-Fenster und sind
# keine Ausfaelle (vgl. datenlage_final.txt, Hinweis zu 04.08. nach 14:00Z)


def main() -> None:
    df = pd.read_csv(OUT_DIR / "paired_slots.csv", parse_dates=["ts"])
    df = df[df["ts"] <= pd.Timestamp(CAMPAIGN_END)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
        }
    )

    fig, ax = plt.subplots(figsize=(12.2 * CM, 5.6 * CM), constrained_layout=True)
    for i, (col, label, color, day_only) in enumerate(ROWS):
        base = df[df["daylight"] == 1] if day_only else df
        missing = base.loc[base[col].isna(), "ts"]
        y = len(ROWS) - 1 - i
        ax.eventplot(mdates.date2num(missing), lineoffsets=y, linelengths=0.55,
                     linewidths=0.7, color=color)
        ax.text(1.005, y, str(len(missing)), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.0, color="#3a3a3a")

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([label for _, label, _, _ in reversed(ROWS)])
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlim(mdates.date2num(pd.Timestamp("2026-07-02")),
                mdates.date2num(pd.Timestamp("2026-08-05")))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("time [UTC] — marks = 10-min slots without a value")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig54_gap_timeline.{suffix}", dpi=300)
    print(f"OK -> {FIGURES_DIR / 'fig54_gap_timeline.{png,pdf}'}")


if __name__ == "__main__":
    main()
