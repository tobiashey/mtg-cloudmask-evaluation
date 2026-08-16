"""C2 — Paired-Slot-Datensatz für FF1/FF2 (Phase C).

Baut aus der eingefrorenen Analyse-DB (`cloudhub_analysis.db`, read-only) die
Paired-Slot-Tabelle ts × Quelle → cloud_pct und wendet die Inklusionskriterien
**[E5]** an:

  (a) Sonnenelevation ≥ 5° am Slot-Zeitpunkt (pysolar, Standort SITE_LAT/LON),
  (b) gültige Werte **aller** 7 Quellen im selben 10-min-UTC-Slot
      (Complete-Case-Prinzip → identisches N für alle Quellenvergleiche),
  (c) Kamerabild nicht als ungültig markiert
      (`camera_classification.invalid = 0`, Variante v106).

Zeitraum **[E4]**: volle Kampagne ab Inbetriebnahme Cam 3
(2026-07-02T13:50Z) bis Kampagnenende (2026-08-04). FF1 nutzt auch
Backfill-Satelliten-Slots (fachlich identisches Produkt).

Ausgaben (deterministisch, read-only gegenüber der DB):
  - out/paired_slots.csv        — alle Kamera-Tagfenster-Slots mit Flags,
                                  Spalte `complete_case` markiert den
                                  FF1/FF2-Hauptdatensatz
  - out/c2_ausschluss_report.txt — N-Bilanz und Ausschlussgründe (→ Kap. 5.1)

Aufruf:  python c2_paired_slots.py
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pysolar.solar import get_altitude

# --- Konfiguration (zentral, keine Magic Numbers in der Logik) ---------------
DB_PATH = Path(__file__).parent / "cloudhub_analysis.db"
OUT_DIR = Path(__file__).parent / "out"
SITE_LAT = 48.685  # Mitte der Region of Interest
SITE_LON = 9.011
SUN_ELEV_MIN_DEG = 5.0          # [E5](a)
CAMPAIGN_START = "2026-07-02T13:50:00+00:00"  # [E4] Inbetriebnahme Cam 3
CAMPAIGN_END = "2026-08-04T23:59:59+00:00"
SLOT_MINUTES = 10

SOURCES = [
    "camera",
    "satellite_clm",
    "weather_brightsky",
    "weather_openmeteo",
    "weather_openweathermap",
    "weather_tomorrowio",
    "weather_weatherapi",
]


def iter_slots(start: datetime, end: datetime) -> list[datetime]:
    """Alle 10-min-Slots des Kampagnenfensters (UTC)."""
    slots = []
    t = start
    while t <= end:
        slots.append(t)
        t += timedelta(minutes=SLOT_MINUTES)
    return slots


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = con.cursor()

    # cloud_pct je (ts, source) — ROI ist durchgehend 'boeblingen' (QS A4)
    values: dict[str, dict[str, float]] = {s: {} for s in SOURCES}
    for ts, source, pct in cur.execute(
        "SELECT ts, source, cloud_pct FROM measurement "
        "WHERE ts BETWEEN ? AND ? AND cloud_pct IS NOT NULL",
        (CAMPAIGN_START, CAMPAIGN_END),
    ):
        if source in values:
            values[source][ts] = pct

    # Kamera-Gültigkeit und sun_in_frame aus der v106-Klassifikation [E5](c)
    cam_invalid: dict[str, str] = {}
    cam_sun: dict[str, int] = {}
    for ts, invalid, reason, sun in cur.execute(
        "SELECT ts, invalid, invalid_reason, sun_in_frame "
        "FROM camera_classification WHERE variant = 'v106' AND ts IS NOT NULL"
    ):
        if invalid:
            cam_invalid[ts] = reason or "unspezifiziert"
        if sun is not None:
            cam_sun[ts] = int(sun)
    con.close()

    start = datetime.fromisoformat(CAMPAIGN_START)
    end = datetime.fromisoformat(CAMPAIGN_END)
    slots = iter_slots(start, end)

    # Bilanz-Zähler (Reihenfolge = Prüfreihenfolge; jeder Slot zählt nur beim
    # ersten verletzten Kriterium → additive Ausschlusstabelle für Kap. 5.1)
    n_grid = len(slots)
    n_day = 0
    excl_camera_missing = 0     # kein Kamerabild / keine Klassifikation im Slot
    excl_camera_invalid = 0     # [E5](c)
    excl_source_missing: dict[str, int] = {s: 0 for s in SOURCES if s != "camera"}
    n_complete = 0

    rows = []
    for slot in slots:
        ts = slot.isoformat()
        elev = get_altitude(SITE_LAT, SITE_LON, slot)
        is_day = elev >= SUN_ELEV_MIN_DEG
        if is_day:
            n_day += 1

        row: dict[str, object] = {
            "ts": ts,
            "date": ts[:10],
            "sun_elev_deg": round(elev, 2),
            "daylight": int(is_day),
            "camera_invalid": int(ts in cam_invalid),
            "sun_in_frame": cam_sun.get(ts, ""),
        }
        for s in SOURCES:
            row[s] = values[s].get(ts, "")

        complete = False
        if is_day:
            if ts in cam_invalid:
                excl_camera_invalid += 1
            elif ts not in values["camera"]:
                excl_camera_missing += 1
            else:
                missing = [s for s in SOURCES[1:] if ts not in values[s]]
                if missing:
                    excl_source_missing[missing[0]] += 1
                else:
                    complete = True
                    n_complete += 1
        row["complete_case"] = int(complete)
        rows.append(row)

    # --- CSV ------------------------------------------------------------------
    csv_path = OUT_DIR / "paired_slots.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # --- Report ---------------------------------------------------------------
    days_complete = sorted({r["date"] for r in rows if r["complete_case"]})
    lines = [
        "C2 — Paired-Slot-Datensatz: N-Bilanz und Ausschlüsse [E5]",
        f"Erstellt: 2026-08-05 | DB: {DB_PATH.name} (read-only) | Skript: c2_paired_slots.py",
        "",
        f"Kampagnenfenster [E4]:      {CAMPAIGN_START} .. letzter Slot {rows[-1]['ts']}",
        f"Slot-Raster gesamt:         {n_grid}",
        f"davon Tagfenster (Sonnenelevation >= {SUN_ELEV_MIN_DEG}°) [E5a]: {n_day}",
        "",
        "Ausschluss-Kaskade im Tagfenster (jeder Slot zählt beim ersten verletzten Kriterium):",
        f"  Kamerabild ungültig [E5c]:            {excl_camera_invalid}",
        f"  Kamerawert fehlt (kein Bild/NULL):    {excl_camera_missing}",
    ]
    for s, n in excl_source_missing.items():
        lines.append(f"  {s} fehlt:{' ' * max(1, 26 - len(s))}{n}")
    lines += [
        "",
        f"=> Complete-Case-Slots (FF1/FF2-Hauptdatensatz) [E5b]: {n_complete}",
        f"   auf {len(days_complete)} Kalendertagen ({days_complete[0]} .. {days_complete[-1]})",
        "",
        "Hinweis: Complete-Case erzwingt identisches N für alle Quellenvergleiche;",
        "Slots mit nur teilweiser Quellenabdeckung stehen mit Flags in paired_slots.csv",
        "und gehen in die Ausfallanalyse F4 ein, nicht in FF1/FF2.",
    ]
    report_path = OUT_DIR / "c2_ausschluss_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGeschrieben: {csv_path}\n             {report_path}")


if __name__ == "__main__":
    main()
