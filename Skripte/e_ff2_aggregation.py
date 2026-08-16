"""Phase E — Ergänzung (Amendment 10.08.): FF2 auf 30-/60-min-Mitteln.

Motivation (Autor): In der Praxis sollen Rollläden nicht alle 10 Minuten
fahren, nur weil eine einzelne Wolke durchzieht. Diese Ergänzung wiederholt
die FF2-Entscheidungsanalyse (identische Regel < 40 %, identisches
Entscheidungsfenster wie `e_ff2_entscheidungen.py`) auf zeitlich gemittelten
Werten und beziffert zusätzlich die **Schalthäufigkeit** (Entscheidungs-
wechsel pro Tag) je Quelle und Aggregationsebene.

Aggregation analog D4(d): nicht überlappende 30-/60-min-Fenster über den
Complete-Case-Slots im Entscheidungsfenster; ein Fenster zählt nur bei
> 50 % Füllung (30 min: >= 2 von 3 Slots, 60 min: >= 4 von 6). Entscheidung
= Fenstermittel < 40 %, Referenz = Kamera-Fenstermittel < 40 %.

Ausgaben:
  - out/e_ff2_aggregation.csv        — Konfusionszellen + Kennzahlen je Ebene/Quelle
  - out/e_ff2_schaltwechsel.csv      — Entscheidungswechsel pro Tag je Ebene/Quelle
  - out/e_ff2_aggregation_report.txt — Report

Aufruf:  python e_ff2_aggregation.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pysolar.solar import get_altitude, get_azimuth

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"

# Fenster-Logik identisch zu e_ff2_entscheidungen.py (E1)
SITE_LAT = 48.685  # Mitte der Region of Interest
SITE_LON = 9.011
SHADE_THRESHOLD_PCT = 40.0
AZIMUTH_MIN_DEG = 200.0
LOCAL_NOON_UTC = 10
SUNSET_OFFSET_MIN = 105

SOURCES = [
    ("satellite_clm", "MTG FCI L2 CLM"),
    ("weather_brightsky", "Bright Sky (DWD)"),
    ("weather_openmeteo", "Open-Meteo"),
    ("weather_openweathermap", "OpenWeatherMap"),
    ("weather_tomorrowio", "Tomorrow.io"),
    ("weather_weatherapi", "WeatherAPI.com"),
]

LEVELS = [("10 min", "10min", 1), ("30 min", "30min", 2), ("60 min", "60min", 4)]


def in_window(ts: datetime) -> bool:
    if ts.hour < LOCAL_NOON_UTC:
        return False
    if get_azimuth(SITE_LAT, SITE_LON, ts) <= AZIMUTH_MIN_DEG:
        return False
    return get_altitude(SITE_LAT, SITE_LON, ts + timedelta(minutes=SUNSET_OFFSET_MIN)) > 0.0


def confusion(pred: pd.Series, ref: pd.Series) -> dict[str, float]:
    tp = int((pred & ref).sum())
    tn = int((~pred & ~ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    n = tp + tn + fp + fn
    return {
        "N": n,
        "korrekt_beschattet_TP": tp,
        "korrekt_nicht_TN": tn,
        "unnoetig_beschattet_FP": fp,
        "verpasst_FN": fn,
        "accuracy": round((tp + tn) / n, 4) if n else float("nan"),
        "precision": round(tp / (tp + fp), 4) if tp + fp else float("nan"),
        "recall": round(tp / (tp + fn), 4) if tp + fn else float("nan"),
    }


def toggles_per_day(dec: pd.Series, dates: pd.Series) -> float:
    """Mittlere Zahl der Entscheidungswechsel pro Tag (aufeinanderfolgende Bins)."""
    flips_total = 0
    n_days = 0
    for _, grp in dec.groupby(dates):
        if len(grp) < 2:
            continue
        flips_total += int((grp != grp.shift()).iloc[1:].sum())
        n_days += 1
    return round(flips_total / n_days, 2) if n_days else float("nan")


def main() -> None:
    df = pd.read_csv(OUT_DIR / "paired_slots.csv", parse_dates=["ts"])
    df = df[df["complete_case"] == 1].copy()
    df["in_window"] = [in_window(t.to_pydatetime()) for t in df["ts"]]
    win = df[df["in_window"]].copy()

    rows, trows = [], []
    for level_name, freq, min_slots in LEVELS:
        g = win.copy()
        g["bin"] = g["ts"].dt.floor(freq)
        agg = g.groupby("bin").agg(
            n_slots=("ts", "count"),
            **{col: (col, "mean") for col, _ in SOURCES},
            camera=("camera", "mean"),
        ).reset_index()
        agg = agg[agg["n_slots"] >= min_slots]
        agg["date"] = agg["bin"].dt.date
        ref = agg["camera"] < SHADE_THRESHOLD_PCT

        trows.append({"aggregation": level_name, "source": "Kamera-Referenz",
                      "wechsel_pro_tag": toggles_per_day(ref, agg["date"])})
        for col, label in SOURCES:
            pred = agg[col] < SHADE_THRESHOLD_PCT
            rows.append({"aggregation": level_name, "source": label,
                         **confusion(pred, ref)})
            trows.append({"aggregation": level_name, "source": label,
                          "wechsel_pro_tag": toggles_per_day(pred, agg["date"])})

    conf = pd.DataFrame(rows)
    conf.to_csv(OUT_DIR / "e_ff2_aggregation.csv", index=False)
    tog = pd.DataFrame(trows)
    tog.to_csv(OUT_DIR / "e_ff2_schaltwechsel.csv", index=False)

    lines = [
        "Phase E — Ergänzung: FF2 auf Aggregationsebenen (e_ff2_aggregation.py)",
        "Fenster-Logik identisch zu e_ff2_entscheidungen.py; Fenster > 50 % gefüllt.",
        "",
        "Konfusions-Kennzahlen je Ebene:",
        conf.to_string(index=False),
        "",
        "Entscheidungswechsel pro Tag (Schalthäufigkeit der Rollläden):",
        tog.pivot(index="source", columns="aggregation",
                  values="wechsel_pro_tag").to_string(),
    ]
    report = "\n".join(lines)
    (OUT_DIR / "e_ff2_aggregation_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
