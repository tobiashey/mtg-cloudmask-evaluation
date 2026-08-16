"""B3-Nachtrag — Scan aller Kamerabilder auf Magenta-/Farbkanal-Fehlbilder.

Befund 04.08. (Autor + Stichproben): einzelne Aufnahmen zeigen den bekannten
IMX708-Pipeline-Fehler (magenta Frames mit totem Gruenkanal, teils nur in
Bildteilen — vgl. Kommentar zu STREAM_ENABLED in skycam/config.py). Bei totem
Gruen ist die R/B-Klassifikation unbrauchbar (R/B ueberall ~2 -> 100 % Wolke).

Detektor (kalibriert an 3 bekannten Fehlbildern + Gutbildern, alle Gutbilder
exakt 0.0 %): Anteil der Pixel mit G < 5 bei R+B > 40 ("dead green") sowie
Anteil der Bildzeilen, deren Median-Gruen tot ist. Fehlbild, wenn einer der
beiden Anteile > 2 % (normale Bilder liegen bei 0.0 %).

Ergebnis: Liste + Update der Analyse-DB:
  - camera_classification: Spalte invalid_reason (neu), invalid = 1 fuer
    Magenta-Frames (Werte bleiben erhalten - nichts wird verworfen)
  - measurement: cloud_pct/n_* der betroffenen Slots auf NULL (keine gueltige
    Bodenwahrheit an diesem Slot; Rohwert bleibt in camera_classification)
  - Export camera_classification.csv aktualisieren

Aufruf:  python b3_fehlbilder_scan.py
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Kamerabilder und Analyse-DB sind nicht Teil dieses Repositories und werden
# unter daten/ bzw. im Skriptordner erwartet (siehe README).
BASE_DIR = Path(__file__).resolve().parent
CAMERA_DIR = BASE_DIR / "daten" / "camera"
ANALYSIS_DB = BASE_DIR / "cloudhub_analysis.db"
CSV_PATH = BASE_DIR / "camera_classification.csv"

DEAD_FRACTION_LIMIT = 0.02  # > 2 % tote Pixel/Zeilen -> Fehlbild

# Bereits bekannte invalid-Gruende (werden mit benannt, nicht neu bewertet).
APRIORI = {
    "skycam_20260702T135000Z.jpg": "inbetriebnahme",
    "skycam_20260702T140000Z.jpg": "inbetriebnahme",
    "skycam_20260730T180609Z.jpg": "datei_defekt",
    "skycam_20260730T181541Z.jpg": "datei_defekt",
    "skycam_20260730T182412Z.jpg": "datei_defekt",
}


def dead_green_fractions(path: Path) -> tuple[float, float]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    px = float(((g < 5) & (r + b > 40)).mean())
    rows = float(((np.median(g, axis=1) < 5) & (np.median(r + b, axis=1) > 40)).mean())
    return px, rows


def main() -> int:
    images = sorted(CAMERA_DIR.glob("skycam_*.jpg"))
    print(f"Scanne {len(images)} Bilder auf tote Gruenkanaele ...")
    magenta: list[tuple[str, float, float]] = []
    for i, p in enumerate(images, 1):
        if p.name in APRIORI:
            continue
        try:
            px, rows = dead_green_fractions(p)
        except OSError:
            continue
        if px > DEAD_FRACTION_LIMIT or rows > DEAD_FRACTION_LIMIT:
            magenta.append((p.name, px, rows))
            print(f"  FEHLBILD {p.name}: dead_green {px*100:.1f} % Pixel / {rows*100:.1f} % Zeilen")
        if i % 500 == 0:
            print(f"  {i}/{len(images)} ...", flush=True)

    print(f"\n{len(magenta)} Magenta-Fehlbilder gefunden.")

    con = sqlite3.connect(ANALYSIS_DB)
    cols = [r[1] for r in con.execute("PRAGMA table_info(camera_classification)")]
    if "invalid_reason" not in cols:
        con.execute("ALTER TABLE camera_classification ADD COLUMN invalid_reason TEXT")

    for name, reason in APRIORI.items():
        con.execute(
            "UPDATE camera_classification SET invalid=1, invalid_reason=? WHERE image_file=?",
            (reason, name),
        )
    for name, px, rows in magenta:
        con.execute(
            "UPDATE camera_classification SET invalid=1, invalid_reason=? WHERE image_file=?",
            (f"magenta_frame(px={px*100:.1f}%,rows={rows*100:.1f}%)", name),
        )
        # Kein gueltiger Bodenwahrheitswert an diesem Slot: measurement leeren,
        # der Rohwert bleibt in camera_classification nachvollziehbar.
        ts = con.execute(
            "SELECT ts FROM camera_classification WHERE image_file=? AND ts IS NOT NULL LIMIT 1",
            (name,),
        ).fetchone()
        if ts:
            con.execute(
                "UPDATE measurement SET cloud_pct=NULL, oktas=NULL, n_cloud=NULL, "
                "n_clear=NULL, n_valid=NULL WHERE source='camera' AND ts=?",
                (ts[0],),
            )
    con.commit()

    rows_csv = con.execute(
        "SELECT image_file, ts, variant, cloud_pct, n_valid, n_cloud, sun_in_frame, invalid, invalid_reason "
        "FROM camera_classification ORDER BY image_file, variant"
    ).fetchall()
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_file", "ts", "variant", "cloud_pct", "n_valid", "n_cloud",
                    "sun_in_frame", "invalid", "invalid_reason"])
        w.writerows(rows_csv)

    n_inv = con.execute(
        "SELECT COUNT(DISTINCT image_file) FROM camera_classification WHERE invalid=1"
    ).fetchone()[0]
    n_meas = con.execute(
        "SELECT COUNT(*) FROM measurement WHERE source='camera' AND cloud_pct IS NOT NULL"
    ).fetchone()[0]
    print(f"invalid gesamt (Bilder): {n_inv}; measurement mit cloud_pct: {n_meas}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
