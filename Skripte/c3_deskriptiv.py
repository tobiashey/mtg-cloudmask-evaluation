"""C3 — Deskriptive Statistik und Kampagnen-Zeitreihe (Phase C).

Grundlage: `out/paired_slots.csv` aus C2. Zwei Sichten:
  (1) Complete-Case-Datensatz (FF1/FF2-Hauptdatensatz, identisches N) —
      Verteilungskennzahlen je Quelle → Tabelle Kap. 5.1.
  (2) Wetterlagen-Charakterisierung aus der Kamera-Bodenwahrheit
      (Repräsentativität; Tagesmittel nur als Kennzahl — der Tagesmittel-
      Zeitreihenplot entfällt, Entscheid des Autors 05.08.2026).

Wetterlagen-Bins (deskriptiv, keine Auswertungsgröße; Okta-Äquivalente nur
zur Anschauung): klar < 12,5 % (0–1 Okta), durchbrochen 12,5–87,5 %,
bedeckt > 87,5 % (7–8 Okta).

Plot-Sprache: Englisch (Thesis-Artefakt). Ausgaben nach `out/`.
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
    "camera": "Camera (ground truth)",
    "satellite_clm": "MTG FCI L2 CLM",
    "weather_brightsky": "Bright Sky (DWD)",
    "weather_openmeteo": "Open-Meteo",
    "weather_openweathermap": "OpenWeatherMap",
    "weather_tomorrowio": "Tomorrow.io",
    "weather_weatherapi": "WeatherAPI.com",
}
CLEAR_MAX = 12.5    # klar        (< 12,5 %  ~ 0–1 Okta)
OVERCAST_MIN = 87.5  # bedeckt    (> 87,5 %  ~ 7–8 Okta)


def main() -> None:
    df = pd.read_csv(PAIRED_CSV, parse_dates=["ts"])
    cc = df[df["complete_case"] == 1].copy()

    # --- (1) Verteilungskennzahlen je Quelle (Complete-Case) ------------------
    stats = []
    for col, label in SOURCES.items():
        v = cc[col].astype(float)
        stats.append(
            {
                "source": label,
                "n": int(v.notna().sum()),
                "mean": v.mean(),
                "sd": v.std(ddof=1),
                "median": v.median(),
                "p25": v.quantile(0.25),
                "p75": v.quantile(0.75),
                "share_0_12.5": (v < CLEAR_MAX).mean() * 100,
                "share_87.5_100": (v > OVERCAST_MIN).mean() * 100,
                "n_distinct": int(v.nunique()),
            }
        )
    stats_df = pd.DataFrame(stats).round(1)
    stats_df.to_csv(OUT_DIR / "c3_deskriptiv_quellen.csv", index=False)

    # --- (2) Wetterlagen-Charakterisierung aus der Kamera ---------------------
    cam = cc["camera"].astype(float)
    lage = pd.cut(
        cam,
        bins=[-0.1, CLEAR_MAX, OVERCAST_MIN, 100.1],
        labels=["clear (<12.5%)", "broken (12.5-87.5%)", "overcast (>87.5%)"],
    )
    lage_share = lage.value_counts(normalize=True).sort_index() * 100

    daily = cc.groupby(cc["ts"].dt.date)[list(SOURCES)].mean()
    daily_cam = daily["camera"]
    day_lage = pd.cut(
        daily_cam,
        bins=[-0.1, 25, 75, 100.1],
        labels=["mostly clear day (<25%)", "mixed day (25-75%)", "mostly overcast day (>75%)"],
    ).value_counts().sort_index()

    report = [
        "C3 — Deskriptive Statistik (Complete-Case, N = %d Slots, %d Tage)" % (len(cc), daily.shape[0]),
        "Skript: c3_deskriptiv.py | Grundlage: out/paired_slots.csv (C2)",
        "",
        stats_df.to_string(index=False),
        "",
        "Wetterlagen-Anteile (Slot-Ebene, Kamera-Bodenwahrheit):",
        lage_share.round(1).to_string(),
        "",
        "Tagescharakter (Tagesmittel Kamera):",
        day_lage.to_string(),
        "",
        "Hinweis Quantisierung: n_distinct zeigt die Wertevielfalt je Quelle",
        "(Bright Sky liefert okta-quantisierte Prozentwerte, vgl. [E2]).",
    ]
    (OUT_DIR / "c3_deskriptiv_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    # --- Plot (englisch) ------------------------------------------------------
    # Verteilungen: Histogramm je Quelle (identische Bins)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharex=True, sharey=True)
    bins = np.arange(0, 105, 5)
    for ax, (col, label) in zip(axes.flat, SOURCES.items()):
        ax.hist(cc[col].astype(float), bins=bins, color="tab:blue", edgecolor="white")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    fig.supxlabel("Cloud cover [%]")
    fig.supylabel("Number of 10-min slots")
    fig.suptitle("Cloud cover distributions, complete-case slots (N = %d)" % len(cc))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_distributions.{ext}", dpi=200)
    plt.close(fig)

    print(f"\nPlot: {OUT_DIR}\\c3_distributions.png/.pdf")


if __name__ == "__main__":
    main()
