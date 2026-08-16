"""D1 — FF1-Haupttabelle: Genauigkeit der Bewölkungsschätzung (Phase D).

Metriken je Quelle vs. Kamera-Bodenwahrheit (v106, [E1]/[E6]) auf dem
Complete-Case-Datensatz aus C2 (`out/paired_slots.csv`):

  MAE (primär), RMSE (sekundär)  — Willmott & Matsuura 2005; Chai & Draxler 2014
  Bias (ME = Quelle − Kamera)    — Jolliffe & Stephenson 2012
  Pearson-r                      — Jolliffe & Stephenson 2012

Konfidenzintervalle **[E3]**: Block-Bootstrap auf Tagesebene (Resampling-
Einheit = Kalendertag, 10 000 Resamples, 95-%-Perzentil-CIs; Künsch 1989,
Efron & Tibshirani 1993). Fester Seed → deterministisch reproduzierbar.

Sensitivität **D4(a)**: identische Metriken gegen die unterlegene
Bodenwahrheits-Variante v086m (R/B ≥ 0.86 + Sonnenmaske r 25°, [E1]) aus
`camera_classification`.

Ausgaben: out/d1_ff1_haupttabelle.csv, out/d1_ff1_sensitivitaet_v086m.csv,
          out/d1_ff1_report.txt
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "out"
DB_PATH = Path(__file__).parent / "cloudhub_analysis.db"
PAIRED_CSV = OUT_DIR / "paired_slots.csv"

N_BOOT = 10_000
SEED = 858157  # a priori fixiert, deterministisch
CI = (2.5, 97.5)

SOURCES = {
    "satellite_clm": "MTG FCI L2 CLM",
    "weather_brightsky": "Bright Sky (DWD)",
    "weather_openmeteo": "Open-Meteo",
    "weather_openweathermap": "OpenWeatherMap",
    "weather_tomorrowio": "Tomorrow.io",
    "weather_weatherapi": "WeatherAPI.com",
}


def day_aggregates(x: np.ndarray, y: np.ndarray, day_idx: np.ndarray, n_days: int) -> np.ndarray:
    """Tagesweise Summen für die Blockbootstrap-Rekombination.

    Spalten: n, Σ|e|, Σe, Σe², Σx, Σy, Σx², Σy², Σxy  (e = x − y; x = Quelle,
    y = Bodenwahrheit). Aus diesen Summen sind MAE/RMSE/ME/Pearson-r für jede
    Tages-Rekombination exakt berechenbar.
    """
    e = x - y
    cols = np.stack([np.ones_like(e), np.abs(e), e, e**2, x, y, x**2, y**2, x * y], axis=1)
    agg = np.zeros((n_days, cols.shape[1]))
    np.add.at(agg, day_idx, cols)
    return agg


def metrics_from_sums(s: np.ndarray) -> dict[str, np.ndarray]:
    """Metriken aus (aufsummierten) Tagesaggregaten; s hat Shape (..., 9)."""
    n, sae, se, sse, sx, sy, sxx, syy, sxy = (s[..., i] for i in range(9))
    mae = sae / n
    rmse = np.sqrt(sse / n)
    me = se / n
    cov = sxy / n - (sx / n) * (sy / n)
    r = cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
    return {"MAE": mae, "RMSE": rmse, "Bias (ME)": me, "Pearson r": r}


def evaluate(df: pd.DataFrame, truth_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Haupttabelle: Punktschätzer + 95-%-Block-Bootstrap-CIs je Quelle."""
    days = pd.factorize(df["date"])[0]
    n_days = days.max() + 1
    boot_idx = rng.integers(0, n_days, size=(N_BOOT, n_days))

    rows = []
    y = df[truth_col].to_numpy(float)
    for col, label in SOURCES.items():
        x = df[col].to_numpy(float)
        agg = day_aggregates(x, y, days, n_days)
        point = metrics_from_sums(agg.sum(axis=0))
        boot = metrics_from_sums(agg[boot_idx].sum(axis=1))
        row: dict[str, object] = {"source": label, "n": len(df), "n_days": n_days}
        for m, val in point.items():
            lo, hi = np.percentile(boot[m], CI)
            row[m] = round(float(val), 2)
            row[f"{m} 95% CI"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(PAIRED_CSV)
    cc = df[df["complete_case"] == 1].copy()
    rng = np.random.default_rng(SEED)

    # --- Haupttabelle (Bodenwahrheit v106) ------------------------------------
    main_tbl = evaluate(cc, "camera", rng)
    main_tbl.to_csv(OUT_DIR / "d1_ff1_haupttabelle.csv", index=False)

    # --- Sensitivität D4(a): Bodenwahrheit v086m ------------------------------
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    v086 = pd.read_sql_query(
        "SELECT ts, cloud_pct AS camera_v086m FROM camera_classification "
        "WHERE variant = 'v086m' AND invalid = 0 AND cloud_pct IS NOT NULL",
        con,
    )
    con.close()
    cc86 = cc.merge(v086, on="ts", how="inner")
    rng86 = np.random.default_rng(SEED)  # gleicher Seed, eigener Strom
    sens_tbl = evaluate(cc86, "camera_v086m", rng86)
    sens_tbl.to_csv(OUT_DIR / "d1_ff1_sensitivitaet_v086m.csv", index=False)

    report = [
        "D1 — FF1-Haupttabelle: Quelle vs. Kamera-Bodenwahrheit (cloud_pct, 0-100 %)",
        f"Complete-Case [E5], N = {len(cc)} Slots / {cc['date'].nunique()} Tage.",
        f"CIs: Tagesblock-Bootstrap [E3], B = {N_BOOT}, Perzentil 95 %, Seed = {SEED}.",
        "Bias = Quelle - Kamera (positiv = Quelle überschätzt Bewölkung).",
        "",
        main_tbl.to_string(index=False),
        "",
        f"Sensitivität D4(a) — Bodenwahrheit v086m (0.86 + Sonnenmaske), N = {len(cc86)}:",
        sens_tbl.to_string(index=False),
        "",
        "Skript: d1_ff1_metriken.py (deterministisch).",
    ]
    (OUT_DIR / "d1_ff1_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
