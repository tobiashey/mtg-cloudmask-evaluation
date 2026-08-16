"""D2 — FF1-Abbildungen: Streudiagramme und Zeitreihen-Overlay (Phase D).

Grundlage: Complete-Case-Datensatz aus C2 (`out/paired_slots.csv`).
  (1) Streudiagramm-Panel je Quelle vs. Kamera-Bodenwahrheit (Identitätslinie,
      MAE/r im Titel) → Kap. 5.2.
  (2) Slot-aufgelöstes Zeitreihen-Overlay Kamera vs. MTG CLM vs. Bright Sky
      für eine kontrastreiche Beispielwoche → Kap. 5.2.

Plot-Sprache: Englisch. Ausgaben: out/d2_scatter_sources.{png,pdf},
out/d2_timeseries_week.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "out"
PAIRED_CSV = OUT_DIR / "paired_slots.csv"

SOURCES = {
    "satellite_clm": "MTG FCI L2 CLM",
    "weather_brightsky": "Bright Sky (DWD)",
    "weather_openmeteo": "Open-Meteo",
    "weather_openweathermap": "OpenWeatherMap",
    "weather_tomorrowio": "Tomorrow.io",
    "weather_weatherapi": "WeatherAPI.com",
}
# Beispielwoche mit klaren, durchbrochenen und bedeckten Tagen (aus C3-Zeitreihe)
WEEK_START, WEEK_END = "2026-07-06", "2026-07-12"


def main() -> None:
    df = pd.read_csv(PAIRED_CSV, parse_dates=["ts"])
    cc = df[df["complete_case"] == 1].copy()
    y = cc["camera"].astype(float)

    # --- (1) Streudiagramm-Panel ---------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (col, label) in zip(axes.flat, SOURCES.items()):
        x = cc[col].astype(float)
        mae = float(np.mean(np.abs(x - y)))
        r = float(np.corrcoef(x, y)[0, 1])
        ax.plot([0, 100], [0, 100], color="grey", lw=1, ls="--", zorder=1)
        ax.scatter(y, x, s=4, alpha=0.15, color="tab:blue", zorder=2)
        ax.set_title(f"{label}\nMAE = {mae:.1f} pp, r = {r:.2f}", fontsize=10)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.supxlabel("Camera ground truth cloud cover [%]")
    fig.supylabel("Source cloud cover [%]")
    fig.suptitle(f"Source vs. ground truth, complete-case 10-min slots (N = {len(cc)})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d2_scatter_sources.{ext}", dpi=200)
    plt.close(fig)

    # --- (2) Zeitreihen-Overlay Beispielwoche --------------------------------
    wk = cc[(cc["date"] >= WEEK_START) & (cc["date"] <= WEEK_END)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wk["ts"], wk["camera"], color="black", lw=1.6, label="Camera (ground truth)")
    ax.plot(wk["ts"], wk["satellite_clm"], color="tab:red", lw=1.1, alpha=0.9,
            label="MTG FCI L2 CLM")
    ax.plot(wk["ts"], wk["weather_brightsky"], color="tab:blue", lw=1.1, alpha=0.9,
            label="Bright Sky (DWD)")
    ax.set_ylabel("Cloud cover [%]")
    ax.set_xlabel("Date (2026, UTC)")
    ax.set_ylim(-2, 102)
    ax.set_title(f"10-min slot comparison, example week {WEEK_START} – {WEEK_END} "
                 "(gaps = night-time)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d2_timeseries_week.{ext}", dpi=200)
    plt.close(fig)

    print(f"Plots geschrieben nach {OUT_DIR}: d2_scatter_sources, d2_timeseries_week (je png+pdf)")


if __name__ == "__main__":
    main()
