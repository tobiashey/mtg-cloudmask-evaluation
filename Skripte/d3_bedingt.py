"""D3 — Bedingte Fehleranalyse (Phase D).

MAE (primär) und Bias je Quelle vs. Kamera-Bodenwahrheit, aufgeschlüsselt nach
a priori definierten Bedingungen:

  (1) `sun_in_frame` ja/nein (aus camera_classification v106, via C2-CSV),
  (2) Tageszeit (UTC-Stundenbänder: morning < 09, midday 09–13, afternoon ≥ 13),
  (3) Bewölkungscharakter aus der Bodenwahrheit:
      homogen-klar (< 12,5 %), durchbrochen (12,5–87,5 %), homogen-bedeckt
      (> 87,5 %) — Proxy „0/100-Nähe" laut Plan,
  (4) Satellit: Live- vs. Backfill-Slots (Robustheitscheck, Erwartung: kein
      Unterschied). Zuordnung über `created_at − ts` in `measurement`
      (> 24 h = Backfill; der Backfill lief am 04.08., vgl. A1),
  (5) Fehlerprofil über den Bewölkungsgrad (Ergänzung des Autors 05.08.):
      MAE und Bias je Quelle in 10-pp-Bins der Kamera-Bodenwahrheit —
      beantwortet „welche Quelle ist bei welcher Wetterlage genau?"
      (Tabelle + Abbildung, englische Beschriftung),
  (6) Sonnenelevations-Bänder (Ergänzung 05.08., Low-Light-Frage): 5–15°,
      15–30°, ≥ 30° — prüft, ob die Kamera-Bodenwahrheit bei tiefer Sonne
      systematisch anders abweicht (Input für die Diskussion, ob Slots
      mit geringer Solar-Relevanz gesondert behandelt werden müssen).

Punktschätzer + N je Zelle (keine CIs — Haupttabelle D1 trägt die Inferenz).
Ausgabe: out/d3_bedingte_analyse.txt (+ CSV je Bedingung),
         out/d3_error_vs_cloudcover.{png,pdf}.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "out"
DB_PATH = Path(__file__).parent / "cloudhub_analysis.db"
PAIRED_CSV = OUT_DIR / "paired_slots.csv"

SOURCES = {
    "satellite_clm": "MTG FCI L2 CLM",
    "weather_brightsky": "Bright Sky (DWD)",
    "weather_openmeteo": "Open-Meteo",
    "weather_openweathermap": "OpenWeatherMap",
    "weather_tomorrowio": "Tomorrow.io",
    "weather_weatherapi": "WeatherAPI.com",
}


def cond_table(cc: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """MAE/Bias je Quelle und Bedingungs-Ausprägung (Spalten-Multiindex)."""
    y = cc["camera"].astype(float)
    out = {}
    for col, label in SOURCES.items():
        e = cc[col].astype(float) - y
        g = pd.DataFrame({"grp": group, "abs_e": e.abs(), "e": e}).groupby("grp", observed=True)
        agg = g.agg(N=("e", "size"), MAE=("abs_e", "mean"), Bias=("e", "mean")).round(2)
        out[label] = agg
    tbl = pd.concat(out, axis=1)
    return tbl


def main() -> None:
    cc = pd.read_csv(PAIRED_CSV, parse_dates=["ts"])
    cc = cc[cc["complete_case"] == 1].copy()

    sections: list[str] = [
        "D3 — Bedingte Fehleranalyse (MAE/Bias in pp vs. Kamera, Complete-Case N = %d)"
        % len(cc),
        "Skript: d3_bedingt.py | Grundlage: out/paired_slots.csv (C2)",
    ]

    # (1) sun_in_frame
    sun = cc["sun_in_frame"].map({1.0: "sun in frame", 0.0: "sun not in frame"})
    t1 = cond_table(cc, sun)
    t1.to_csv(OUT_DIR / "d3_sun_in_frame.csv")
    sections += ["", "== (1) Sonne im Bild ==", t1.to_string()]

    # (2) Tageszeit (UTC)
    hour = cc["ts"].dt.hour
    tod = pd.cut(hour, bins=[0, 9, 13, 24], right=False,
                 labels=["morning (<09Z)", "midday (09-13Z)", "afternoon (>=13Z)"])
    t2 = cond_table(cc, tod)
    t2.to_csv(OUT_DIR / "d3_tageszeit.csv")
    sections += ["", "== (2) Tageszeit ==", t2.to_string()]

    # (3) Bewölkungscharakter (Bodenwahrheit)
    charakter = pd.cut(cc["camera"].astype(float), bins=[-0.1, 12.5, 87.5, 100.1],
                       labels=["clear (<12.5%)", "broken (12.5-87.5%)", "overcast (>87.5%)"])
    t3 = cond_table(cc, charakter)
    t3.to_csv(OUT_DIR / "d3_bewoelkungscharakter.csv")
    sections += ["", "== (3) Bewölkungscharakter (Kamera) ==", t3.to_string()]

    # (4) Satellit live vs. backfill
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    sat = pd.read_sql_query(
        "SELECT ts, created_at FROM measurement WHERE source = 'satellite_clm'", con)
    con.close()
    delay_h = (pd.to_datetime(sat["created_at"], utc=True, format="ISO8601")
               - pd.to_datetime(sat["ts"], utc=True)).dt.total_seconds() / 3600
    sat["ingest"] = np.where(delay_h > 24, "backfill (>24h)", "live (<=24h)")
    cc4 = cc.merge(sat[["ts", "ingest"]], left_on=cc["ts"].apply(lambda t: t.isoformat()),
                   right_on="ts", how="left", suffixes=("", "_sat"))
    e = cc4["satellite_clm"].astype(float) - cc4["camera"].astype(float)
    g = pd.DataFrame({"grp": cc4["ingest"], "abs_e": e.abs(), "e": e}).groupby("grp")
    t4 = g.agg(N=("e", "size"), MAE=("abs_e", "mean"), Bias=("e", "mean")).round(2)
    t4.to_csv(OUT_DIR / "d3_satellit_live_backfill.csv")
    sections += ["", "== (4) Satellit: Live vs. Backfill (Robustheitscheck [E4]) ==",
                 t4.to_string()]

    # (5) Fehlerprofil über den Bewölkungsgrad (10-pp-Bins der Bodenwahrheit)
    y = cc["camera"].astype(float)
    edges = np.arange(0, 110, 10)
    labels = [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])]
    bins = pd.cut(y, bins=edges, labels=labels, include_lowest=True)
    t5 = cond_table(cc, bins)
    t5.to_csv(OUT_DIR / "d3_fehlerprofil_bins.csv")
    sections += ["", "== (5) Fehlerprofil: MAE/Bias je 10-pp-Bin der Kamera-Bodenwahrheit ==",
                 t5.to_string()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    centers = edges[:-1] + 5
    colors = {
        "MTG FCI L2 CLM": "tab:red",
        "Bright Sky (DWD)": "tab:blue",
        "Open-Meteo": "tab:green",
        "OpenWeatherMap": "tab:orange",
        "Tomorrow.io": "tab:purple",
        "WeatherAPI.com": "tab:brown",
    }
    for label, color in colors.items():
        lw = 2 if label in ("MTG FCI L2 CLM", "Bright Sky (DWD)") else 1
        ax1.plot(centers, t5[(label, "MAE")], marker="o", ms=3, lw=lw, color=color, label=label)
        ax2.plot(centers, t5[(label, "Bias")], marker="o", ms=3, lw=lw, color=color, label=label)
    ax1.set_ylabel("MAE [pp]")
    ax1.set_title("Error magnitude by sky condition")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("Bias (source − camera) [pp]")
    ax2.set_title("Systematic deviation by sky condition")
    for ax in (ax1, ax2):
        ax.set_xlabel("Camera ground truth cloud cover bin [%]")
        ax.set_xticks(edges)
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    fig.suptitle("Conditional error profile per source (complete-case slots, bin width 10 pp)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d3_error_vs_cloudcover.{ext}", dpi=200)
    plt.close(fig)

    # (6) Sonnenelevations-Bänder (Low-Light-Sensitivität der Bodenwahrheit)
    elev = pd.cut(cc["sun_elev_deg"].astype(float), bins=[5, 15, 30, 90],
                  labels=["low sun (5-15°)", "mid sun (15-30°)", "high sun (>=30°)"],
                  include_lowest=True)
    t6 = cond_table(cc, elev)
    t6.to_csv(OUT_DIR / "d3_sonnenelevation.csv")
    sections += ["", "== (6) Sonnenelevation (Low-Light-Check) ==", t6.to_string()]

    text = "\n".join(sections)
    (OUT_DIR / "d3_bedingte_analyse.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
