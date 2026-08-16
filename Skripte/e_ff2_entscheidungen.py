"""Phase E — FF2: Beschattungsentscheidungen im Vergleich.

Wendet die Entscheidungsregel der produktiven Automation **identisch auf alle
Quellen** an und vergleicht die abgeleiteten Entscheidungen gegen die
Kamera-Referenzentscheidung (Konfusionsmatrizen, Accuracy/Precision/Recall)
sowie ergaenzend gegen die [E7]-Referenzen (Lux-Flag `sun_unobscured`,
dichte Bewoelkung `cloud_dense_pct`).

Entscheidungsregel (E1, a priori [E2]):
  beschatten :<=> cloud_pct < 40 %  — innerhalb des Entscheidungsfensters der
  Automation. Fenster (aus `HA-BEschattung.yaml`, identisch fuer alle Quellen):
    (a) Ortszeit nach 12:00 (Kampagne durchgehend MESZ -> ab 10:00 UTC),
    (b) Sonnenazimut > 200 Grad (pysolar, Standort wie c2),
    (c) mehr als 1:45 h vor Sonnenuntergang (operationalisiert: Sonne steht
        105 min nach dem Slot noch ueber dem Horizont),
    (d) Complete-Case-Slot [E5] (identisches N fuer alle Quellen).
  Die Elevationsbaender der Automation (< 50 Grad rechts / > 10 Grad links)
  steuern nur die Zuordnung zu den Rollladengruppen, nicht OB beschattet wird —
  fuer die Entscheidungsanalyse daher nicht filterwirksam.

Konfusions-Konvention (positive Klasse = "beschatten"):
  TP korrekt beschattet / TN korrekt nicht beschattet /
  FP unnoetig beschattet / FN verpasst zu beschatten.

Ausgaben (deterministisch; liest nur out/paired_slots.csv + camera_e7.csv):
  - out/e2_konfusionsmatrizen.csv   — Zellen + Kennzahlen je Quelle vs. Kamera
  - out/e7_ff2_referenzen.csv       — Kennzahlen vs. Lux- und Dicht-Referenz
  - out/e_ff2_report.txt            — Report inkl. Fallbeispiel-Kandidaten (E4)

Aufruf:  python e_ff2_entscheidungen.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pysolar.solar import get_altitude, get_azimuth

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
PAIRED_SLOTS = OUT_DIR / "paired_slots.csv"
CAMERA_E7 = AUSWERTUNG_DIR / "camera_e7.csv"

SITE_LAT = 48.685  # Mitte der Region of Interest (wie c2)
SITE_LON = 9.011

SHADE_THRESHOLD_PCT = 40.0    # [E2]: "< 3 Okta" -> 37,5 % -> operationalisiert < 40 %
AZIMUTH_MIN_DEG = 200.0       # HA-Automation: Azimut > 200
LOCAL_NOON_UTC = 10           # 12:00 MESZ = 10:00 UTC (Kampagne komplett in MESZ)
SUNSET_OFFSET_MIN = 105       # "Sonnenuntergang - 1:45 h"

SOURCES = [
    ("satellite_clm", "MTG FCI L2 CLM"),
    ("weather_brightsky", "Bright Sky (DWD)"),
    ("weather_openmeteo", "Open-Meteo"),
    ("weather_openweathermap", "OpenWeatherMap"),
    ("weather_tomorrowio", "Tomorrow.io"),
    ("weather_weatherapi", "WeatherAPI.com"),
]


def confusion(pred: pd.Series, ref: pd.Series) -> dict[str, float]:
    """Konfusionszellen + Kennzahlen; positive Klasse = beschatten (True)."""
    tp = int(((pred) & (ref)).sum())
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


def main() -> None:
    df = pd.read_csv(PAIRED_SLOTS, parse_dates=["ts"])
    df = df[df["complete_case"] == 1].copy()

    # --- E1: Entscheidungsfenster -------------------------------------------
    def in_window(ts: datetime) -> bool:
        if ts.hour < LOCAL_NOON_UTC:
            return False
        if get_azimuth(SITE_LAT, SITE_LON, ts) <= AZIMUTH_MIN_DEG:
            return False
        later = ts + timedelta(minutes=SUNSET_OFFSET_MIN)
        return get_altitude(SITE_LAT, SITE_LON, later) > 0.0

    df["in_window"] = [in_window(ts.to_pydatetime()) for ts in df["ts"]]
    win = df[df["in_window"]].copy()

    # --- E2/E3: Entscheidungen + Konfusion vs. Kamera-Referenz ---------------
    ref_cam = win["camera"] < SHADE_THRESHOLD_PCT
    rows = []
    for col, label in SOURCES:
        pred = win[col] < SHADE_THRESHOLD_PCT
        rows.append({"source": label, "reference": "Kamera (v106)", **confusion(pred, ref_cam)})
    conf = pd.DataFrame(rows)
    conf.to_csv(OUT_DIR / "e2_konfusionsmatrizen.csv", index=False)

    # --- E7: Zusatzreferenzen Lux + dichte Bewoelkung ------------------------
    e7 = pd.read_csv(CAMERA_E7, parse_dates=["ts"])
    e7 = e7[e7["invalid"] == 0][["ts", "cloud_dense_pct", "sun_unobscured"]]
    wine7 = win.merge(e7, on="ts", how="left")

    rows7 = []
    # Referenz A: dichte Bewoelkung — beschatten, wenn dichte Wolkenflaeche < 40 %
    sub = wine7[wine7["cloud_dense_pct"].notna()]
    ref_dense = sub["cloud_dense_pct"] < SHADE_THRESHOLD_PCT
    for col, label in SOURCES:
        rows7.append({"source": label, "reference": "cloud_dense_pct < 40",
                      **confusion(sub[col] < SHADE_THRESHOLD_PCT, ref_dense)})
    # Referenz B: Lux-Flag — beschatten, wenn die Sonne unverdeckt ist
    sub = wine7[wine7["sun_unobscured"].isin([0, 1])]
    ref_lux = sub["sun_unobscured"] == 1
    for col, label in SOURCES:
        rows7.append({"source": label, "reference": "sun_unobscured (Lux)",
                      **confusion(sub[col] < SHADE_THRESHOLD_PCT, ref_lux)})
    # Zum Einordnen: wie gut trifft die Kamera-Gesamtbewoelkung die Lux-Referenz?
    rows7.append({"source": "Kamera (v106)", "reference": "sun_unobscured (Lux)",
                  **confusion(sub["camera"] < SHADE_THRESHOLD_PCT, ref_lux)})
    conf7 = pd.DataFrame(rows7)
    conf7.to_csv(OUT_DIR / "e7_ff2_referenzen.csv", index=False)

    # --- E4: Fallbeispiel-Kandidaten (CLM- vs. DWD-Entscheidung divergiert) --
    win["dec_clm"] = win["satellite_clm"] < SHADE_THRESHOLD_PCT
    win["dec_dwd"] = win["weather_brightsky"] < SHADE_THRESHOLD_PCT
    win["dec_cam"] = ref_cam
    div = win[win["dec_clm"] != win["dec_dwd"]].copy()
    div["clm_richtig"] = div["dec_clm"] == div["dec_cam"]
    per_day = div.groupby("date").agg(
        n_divergent=("ts", "count"), clm_richtig=("clm_richtig", "sum")
    ).sort_values("n_divergent", ascending=False)

    # --- Report --------------------------------------------------------------
    lines = []
    lines.append("Phase E — FF2 Beschattungsentscheidungen (e_ff2_entscheidungen.py)")
    lines.append(f"Complete-Case-Slots gesamt: {len(df)}")
    lines.append(f"davon im Entscheidungsfenster (nach 12:00 MESZ, Azimut > 200°, "
                 f"> 1:45 h vor Sonnenuntergang): {len(win)}")
    lines.append(f"Kamera-Referenz: beschatten in {int(ref_cam.sum())} Slots "
                 f"({100 * ref_cam.mean():.1f} %)")
    lines.append("")
    lines.append("Konfusionsmatrizen vs. Kamera-Referenz (Zellen absolut):")
    lines.append(conf.to_string(index=False))
    lines.append("")
    lines.append("[E7] Zusatzreferenzen (Teilmengen mit gueltiger Referenz):")
    lines.append(conf7.to_string(index=False))
    lines.append("")
    lines.append("E4 — Divergenz CLM vs. DWD je Tag (Top 8; clm_richtig = Kamera "
                 "gibt dem Satelliten recht):")
    lines.append(per_day.head(8).to_string())
    lines.append("")
    beispiele = div.sort_values("ts").head(0)  # Platzhalter, s. Auswahl unten
    fokus = div[div["date"].isin(["2026-07-19", "2026-07-21"])]
    lines.append(f"Divergente Slots am 19./21.07. (Kandidaten aus dem Leitfaden): "
                 f"{len(fokus)}")
    cols = ["ts", "camera", "satellite_clm", "weather_brightsky", "dec_cam",
            "dec_clm", "dec_dwd"]
    if len(fokus):
        lines.append(fokus[cols].to_string(index=False))
    report = "\n".join(lines)
    (OUT_DIR / "e_ff2_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
