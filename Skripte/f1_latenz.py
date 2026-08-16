"""Phase F — F1: Latenzkette Satellit (nur Live-Betrieb [E4]).

Berechnet je Live-Szene die End-to-End-Latenz der Satellitenkette aus der
eingefrorenen Analyse-DB (read-only):

  latenz_sensing = created_at − sensing_end   (Verfuegbarkeit nach Scan-Ende)
  latenz_slot    = created_at − ts            (Wertalter bezogen auf Slot-Beginn;
                                               sensing_end = ts + 10 min)

Live-Abgrenzung [E4]: nur Szenen bis zum Stream-Abbruch (ts <= 25.07. 20:00Z)
mit Latenz < 6 h (schliesst theoretische Nachzuegler aus); alles danach ist
Backfill (job satellite_backfill) und geht NICHT in die Latenzstatistik ein.

Ausgaben:
  - out/f1_latenz.csv         — je Live-Szene ts, latenz_sensing_min, latenz_slot_min
  - out/f1_latenz_report.txt  — Kennzahlen (N, Median, Mittel, P5/P95, Max)

Aufruf:  python f1_latenz.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
DB_PATH = AUSWERTUNG_DIR / "cloudhub_analysis.db"
OUT_DIR = AUSWERTUNG_DIR / "out"

LIVE_END = "2026-07-25T20:00:00+00:00"  # letzter Live-Slot vor dem Stream-Abbruch
MAX_LIVE_LATENCY_H = 6.0                # Schutz gegen theoretische Nachzuegler


def main() -> None:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT ts, sensing_end, created_at FROM satellite_scene "
        "WHERE source = 'satellite_clm' AND roi = 'boeblingen' ORDER BY ts",
        con,
    )
    con.close()

    for col in ("ts", "sensing_end", "created_at"):
        df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)

    df["latenz_sensing_min"] = (
        (df["created_at"] - df["sensing_end"]).dt.total_seconds() / 60.0
    )
    df["latenz_slot_min"] = (
        (df["created_at"] - df["ts"]).dt.total_seconds() / 60.0
    )

    live = df[
        (df["ts"] <= pd.Timestamp(LIVE_END))
        & (df["latenz_sensing_min"] < MAX_LIVE_LATENCY_H * 60)
    ].copy()

    out = live[["ts", "latenz_sensing_min", "latenz_slot_min"]].copy()
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "f1_latenz.csv", index=False)

    def stats(s: pd.Series) -> str:
        return (f"N={len(s)}  Median={s.median():.1f}  Mittel={s.mean():.1f}  "
                f"P5={s.quantile(0.05):.1f}  P95={s.quantile(0.95):.1f}  "
                f"Max={s.max():.1f}  (Minuten)")

    lines = [
        "F1 — Latenz Satellitenkette, NUR Live-Betrieb [E4] (f1_latenz.py)",
        f"Szenen gesamt in DB: {len(df)}; davon Live (ts <= {LIVE_END}, "
        f"Latenz < {MAX_LIVE_LATENCY_H:.0f} h): {len(live)}",
        "",
        "latenz_sensing = created_at - sensing_end (Verfuegbarkeit nach Scan-Ende):",
        "  " + stats(live["latenz_sensing_min"]),
        "",
        "latenz_slot = created_at - ts (Wertalter ab Slot-Beginn; Scan dauert 10 min):",
        "  " + stats(live["latenz_slot_min"]),
        "",
        f"Anteil Szenen mit latenz_sensing <= 10 min: "
        f"{100 * (live['latenz_sensing_min'] <= 10).mean():.1f} %",
        f"Anteil Szenen mit latenz_sensing <= 15 min: "
        f"{100 * (live['latenz_sensing_min'] <= 15).mean():.1f} %",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "f1_latenz_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
