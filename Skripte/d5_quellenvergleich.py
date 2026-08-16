"""D5 — Inter-Quellen-Vergleich mit Satellit als Bezugspunkt (Amendment 06.08.2026).

Motivation (Autor): Die FF1-Analysen vergleichen alle Quellen gegen die Kamera.
Ergänzend interessiert die Übereinstimmung der Quellen **untereinander**,
insbesondere MTG CLM vs. Bright Sky/DWD. Rechercheergebnis (06.08., Quellen in
Kap. 2.1 der Arbeit): Der DWD-Stationswert an automatisierten Stationen stammt
**ausschließlich vom Ceilometer** (zeitliche Integration der letzten Stunde,
rein vertikaler Blick); Meteosat-/CM-SAF-Daten fließen nur in DWD-Raster-
produkte ein, nicht in die Stationsbeobachtung. CLM und DWD-Wert sind damit
methodisch unabhängig — hohe Übereinstimmung validiert unsere eigene
Pixel-Extraktion (ROI-Aggregation) als eigenständigen Befund.

Berechnungen (Complete-Case-Datensatz aus C2, N identisch zu D1):
  (1) Paarweise Pearson-r-Matrix aller 7 Quellen (+ MAE-Matrix als CSV),
  (2) Scatter-Panel: MTG CLM (x-Achse) vs. die 5 APIs + Kamera,
      mit MAE/Bias/r je Panel.

Nur deskriptiv/Punktschätzer — die Inferenz (CIs) bleibt bei D1; Rollen [E6]
unverändert (Kamera bleibt Bodenwahrheit, dieser Vergleich ist Zusatzbefund).

Ausgaben: out/d5_pearson_matrix.csv, out/d5_mae_matrix.csv,
          out/d5_correlation_matrix.{png,pdf}, out/d5_scatter_vs_satellite.{png,pdf}
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
    "camera": "Camera",
    "satellite_clm": "MTG CLM",
    "weather_brightsky": "Bright Sky (DWD)",
    "weather_openmeteo": "Open-Meteo",
    "weather_openweathermap": "OpenWeatherMap",
    "weather_tomorrowio": "Tomorrow.io",
    "weather_weatherapi": "WeatherAPI.com",
}


def main() -> None:
    df = pd.read_csv(PAIRED_CSV)
    cc = df[df["complete_case"] == 1]
    data = cc[list(SOURCES)].astype(float).rename(columns=SOURCES)

    # --- (1) Paarweise Matrizen ----------------------------------------------
    r_mat = data.corr(method="pearson").round(3)
    labels = list(SOURCES.values())
    mae_mat = pd.DataFrame(
        {b: [float(np.mean(np.abs(data[a] - data[b]))) for a in labels] for b in labels},
        index=labels,
    ).round(2)
    r_mat.to_csv(OUT_DIR / "d5_pearson_matrix.csv")
    mae_mat.to_csv(OUT_DIR / "d5_mae_matrix.csv")

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(r_mat.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = r_mat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Pairwise correlation of cloud cover sources\n"
                 f"(complete-case 10-min slots, N = {len(data)})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d5_correlation_matrix.{ext}", dpi=200)
    plt.close(fig)

    # --- (2) Scatter: Satellit als Bezugspunkt -------------------------------
    others = [l for l in labels if l != "MTG CLM"]
    x = data["MTG CLM"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, others):
        y = data[label]
        mae = float(np.mean(np.abs(y - x)))
        bias = float(np.mean(y - x))
        r = float(np.corrcoef(x, y)[0, 1])
        ax.plot([0, 100], [0, 100], color="grey", lw=1, ls="--", zorder=1)
        ax.scatter(x, y, s=4, alpha=0.15,
                   color="tab:red" if label == "Bright Sky (DWD)" else "tab:blue", zorder=2)
        ax.set_title(f"{label}\nMAE {mae:.1f} pp, bias {bias:+.1f} pp, r = {r:.2f}",
                     fontsize=9.5)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.supxlabel("MTG FCI L2 CLM cloud cover [%]")
    fig.supylabel("Other source cloud cover [%]")
    fig.suptitle(f"Agreement with the satellite estimate (complete-case slots, N = {len(data)};\n"
                 "descriptive — camera remains the ground truth [E6])")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d5_scatter_vs_satellite.{ext}", dpi=200)
    plt.close(fig)

    print("Pearson-r-Matrix:")
    print(r_mat.to_string())
    print("\nMAE-Matrix [pp]:")
    print(mae_mat.to_string())
    print(f"\nPlots: {OUT_DIR}\\d5_correlation_matrix.*, d5_scatter_vs_satellite.*")


if __name__ == "__main__":
    main()
