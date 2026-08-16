"""B3 — Batch-Klassifikation aller Kamerabilder des Analyse-Freeze.

Klassifiziert jedes Kamera-JPG des Freeze-Ordners mit der finalen Konfiguration
(Entscheid [E1]) und der Sensitivitaetsvariante:

  v106  = R/B >= 1.06, OHNE zirkumsolare Sonnenmaske   (Hauptvariante)
  v086m = R/B >= 0.86, MIT Sonnenmaske r = 25 Grad     (Sensitivitaet)

Beide Varianten nutzen die identische Gueltigkeitslogik aus ``skycam.rbratio``:
Saettigung < 250, Blau > 1, Helligkeitssumme >= 110 (dynamische Maske) und die
statische Hauskanten-Maske (config.STATIC_SKY_MASK_PATH). ``sun_in_frame`` wird
als Diagnose-Flag mitgefuehrt (kein Ausschluss).

Ablauf (idempotent, deterministisch):
  1. Arbeitskopie der Freeze-DB anlegen (Freeze bleibt read-only).
  2. Sanity-Anker: die 15 Gegenpruefungs-Bilder muessen die v106-Referenzwerte
     aus ergebnis_rows.json auf 0,5 pp reproduzieren, sonst Abbruch.
  3. Alle skycam_*.jpg klassifizieren -> Tabelle camera_classification
     (beide Varianten, mit config_json); bereits berechnete Bilder werden
     uebersprungen (resumefaehig).
  4. measurement (source='camera') per image_path mit den v106-Werten fuellen.
  5. Export camera_classification.csv + Kurzstatistik.

Aufruf:  python b3_batch_klassifikation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

# Kamerabilder, Kampagnen-DB und das Kameramodul `skycam` sind nicht Teil
# dieses Repositories und werden unter daten/ erwartet (siehe README).
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "daten"
sys.path.insert(0, str(DATA_ROOT))

from skycam import config, rbratio  # noqa: E402

FREEZE_DB = DATA_ROOT / "cloudhub_final_20260804.db"
CAMERA_DIR = DATA_ROOT / "camera"
OUT_DIR = BASE_DIR
ANALYSIS_DB = OUT_DIR / "cloudhub_analysis.db"
CSV_PATH = OUT_DIR / "camera_classification.csv"
GEGENPRUEFUNG = DATA_ROOT / "gegenpruefung"

# A priori ungueltige Inbetriebnahme-Bilder [E5].
INVALID_FILES = {"skycam_20260702T135000Z.jpg", "skycam_20260702T140000Z.jpg"}

# Toleranz des Sanity-Ankers gegen ergebnis_rows.json (in Prozentpunkten).
ANCHOR_TOL_PP = 0.5

VARIANTS = {
    "v106": {"threshold": 1.06, "sun_mask": False},
    "v086m": {"threshold": 0.86, "sun_mask": True},
}


def _config_json(variant: str) -> str:
    """Konfigurationsstand der Variante fuer die Nachvollziehbarkeit."""
    v = VARIANTS[variant]
    mask_sha = hashlib.sha256(config.STATIC_SKY_MASK_PATH.read_bytes()).hexdigest()
    return json.dumps(
        {
            "variant": variant,
            "rb_threshold": v["threshold"],
            "sun_mask_enabled": v["sun_mask"],
            "sun_mask_radius_deg": config.SUN_MASK_RADIUS_DEG if v["sun_mask"] else None,
            "saturation_cutoff": rbratio.SATURATION_CUTOFF,
            "blue_min": 1.0,
            "sky_mask_min_brightness_sum": config.SKY_MASK_MIN_BRIGHTNESS_SUM,
            "static_sky_mask": config.STATIC_SKY_MASK_PATH.name,
            "static_sky_mask_sha256": mask_sha,
            "camera_tilt_deg": config.CAMERA_TILT_DEG,
            "camera_azimuth_deg": config.CAMERA_AZIMUTH_DEG,
            "hfov_deg": config.CAMERA_HFOV_DEG,
            "vfov_deg": config.CAMERA_VFOV_DEG,
            "source": "skycam.rbratio.classify_image",
            "date": "2026-08-04",
        },
        sort_keys=True,
    )


def sanity_check() -> None:
    """Repliziert die 15 Gegenpruefungs-Bilder und vergleicht mit ergebnis_rows.json."""
    rows = json.loads((GEGENPRUEFUNG / "ergebnis_rows.json").read_text(encoding="utf-8"))
    print(f"Sanity-Anker: {len(rows)} Gegenpruefungs-Bilder ...")
    worst = 0.0
    for row in rows:
        img = CAMERA_DIR / row["orig"]
        r106 = rbratio.classify_image(img, threshold=1.06, apply_sun_mask=False)
        r086m = rbratio.classify_image(img, threshold=0.86, apply_sun_mask=True)
        d106 = abs(r106.cloud_pct - row["rb106"])
        d086m = abs(r086m.cloud_pct - row["rb086m"])
        worst = max(worst, d106, d086m)
        status = "ok" if max(d106, d086m) <= ANCHOR_TOL_PP else "FEHLER"
        print(
            f"  {row['orig']}: v106 {r106.cloud_pct:6.2f} (ref {row['rb106']:6.2f}, "
            f"d={d106:.3f}) | v086m {r086m.cloud_pct:6.2f} (ref {row['rb086m']:6.2f}, "
            f"d={d086m:.3f}) [{status}]"
        )
        if max(d106, d086m) > ANCHOR_TOL_PP:
            raise SystemExit(
                f"ABBRUCH: {row['orig']} weicht > {ANCHOR_TOL_PP} pp von der Referenz ab - "
                "Maskenlogik pruefen (statische Maske? Sonnenmaske in v106 aktiv?)."
            )
    print(f"Sanity-Anker bestanden (max. Abweichung {worst:.3f} pp).\n")


def ensure_analysis_db() -> sqlite3.Connection:
    if not ANALYSIS_DB.exists():
        print(f"Kopiere Freeze-DB -> {ANALYSIS_DB.name} ...")
        shutil.copyfile(FREEZE_DB, ANALYSIS_DB)
    con = sqlite3.connect(ANALYSIS_DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS camera_classification (
            image_file   TEXT NOT NULL,
            ts           TEXT,
            variant      TEXT NOT NULL,
            cloud_pct    REAL,
            n_valid      INTEGER,
            n_cloud      INTEGER,
            sun_in_frame INTEGER,
            invalid      INTEGER NOT NULL DEFAULT 0,
            config_json  TEXT,
            PRIMARY KEY (image_file, variant)
        )"""
    )
    con.commit()
    return con


def main() -> int:
    assert config.STATIC_SKY_MASK_ENABLED, "Statische Maske muss aktiv sein (B3)."
    assert not config.SUN_MASK_ENABLED, "config.SUN_MASK_ENABLED muss False sein ([E1])."
    assert rbratio.RB_CLOUD_THRESHOLD == 1.06, "RB_CLOUD_THRESHOLD != 1.06 ([E1])."

    sanity_check()

    con = ensure_analysis_db()

    # ts-Zuordnung ueber image_path (Dateinamens-Zeitstempel != Slot-ts, z.B.
    # skycam_...141001Z.jpg -> Slot 14:10:00).
    path_to_ts = {
        Path(p).name: ts
        for ts, p in con.execute(
            "SELECT ts, image_path FROM measurement WHERE source='camera' AND image_path IS NOT NULL"
        )
    }

    images = sorted(CAMERA_DIR.glob("skycam_*.jpg"))
    print(f"{len(images)} Kamerabilder, {len(path_to_ts)} measurement-Zeilen.")

    cfg = {v: _config_json(v) for v in VARIANTS}
    done = {
        (f, v)
        for f, v in con.execute("SELECT image_file, variant FROM camera_classification")
    }

    n_new = 0
    for i, img in enumerate(images, 1):
        name = img.name
        if all((name, v) in done for v in VARIANTS):
            continue
        ts = path_to_ts.get(name)
        invalid = 1 if name in INVALID_FILES else 0
        for variant, v in VARIANTS.items():
            try:
                r = rbratio.classify_image(
                    img, threshold=v["threshold"], apply_sun_mask=v["sun_mask"]
                )
            except OSError as exc:
                # Defekte Datei (z.B. abgebrochene Uebertragung): als ungueltig
                # protokollieren, nicht klassifizierbar. Betrifft die drei
                # Off-Raster-Testaufnahmen vom 30.07. (keine measurement-Zeile).
                print(f"  DEFEKT {name}: {exc} -> invalid=1, cloud_pct=NULL")
                con.execute(
                    "INSERT OR REPLACE INTO camera_classification "
                    "(image_file, ts, variant, cloud_pct, n_valid, n_cloud, sun_in_frame, invalid, config_json) "
                    "VALUES (?,?,?,NULL,NULL,NULL,NULL,1,?)",
                    (name, ts, variant, cfg[variant]),
                )
                continue
            con.execute(
                "INSERT OR REPLACE INTO camera_classification "
                "(image_file, ts, variant, cloud_pct, n_valid, n_cloud, sun_in_frame, invalid, config_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    name, ts, variant,
                    None if r.n_valid == 0 else round(r.cloud_pct, 4),
                    r.n_valid, r.n_cloud, int(r.sun_in_frame), invalid, cfg[variant],
                ),
            )
            if variant == "v106" and ts is not None:
                n_total = 2304 * 1296
                con.execute(
                    "UPDATE measurement SET cloud_pct=?, oktas=?, n_cloud=?, n_clear=?, "
                    "n_valid=?, n_total=? WHERE source='camera' AND ts=?",
                    (
                        None if r.n_valid == 0 else round(r.cloud_pct, 4),
                        r.oktas, r.n_cloud, r.n_clear, r.n_valid, n_total, ts,
                    ),
                )
        n_new += 1
        if i % 100 == 0 or i == len(images):
            con.commit()
            print(f"  {i}/{len(images)} verarbeitet ({n_new} neu) ...", flush=True)
    con.commit()

    # CSV-Export (deterministisch sortiert).
    rows = con.execute(
        "SELECT image_file, ts, variant, cloud_pct, n_valid, n_cloud, sun_in_frame, invalid "
        "FROM camera_classification ORDER BY image_file, variant"
    ).fetchall()
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_file", "ts", "variant", "cloud_pct", "n_valid", "n_cloud",
                    "sun_in_frame", "invalid"])
        w.writerows(rows)
    print(f"\nCSV: {CSV_PATH} ({len(rows)} Zeilen)")

    # Kurzstatistik.
    print("\n=== Kurzstatistik ===")
    for variant in VARIANTS:
        q = con.execute(
            "SELECT COUNT(*), AVG(cloud_pct), MIN(cloud_pct), MAX(cloud_pct), "
            "SUM(sun_in_frame), SUM(invalid), SUM(cloud_pct IS NULL) "
            "FROM camera_classification WHERE variant=?",
            (variant,),
        ).fetchone()
        print(
            f"{variant}: n={q[0]}, mean={q[1]:.1f} %, min={q[2]:.1f}, max={q[3]:.1f}, "
            f"sun_in_frame={q[4]} ({100*q[4]/q[0]:.0f} %), invalid={q[5]}, cloud_pct NULL={q[6]}"
        )
        hist = con.execute(
            "SELECT CAST(cloud_pct/10 AS INT)*10 AS b, COUNT(*) FROM camera_classification "
            "WHERE variant=? AND cloud_pct IS NOT NULL GROUP BY b ORDER BY b",
            (variant,),
        ).fetchall()
        print("   Verteilung: " + ", ".join(f"{b}-{min(b + 10, 100)}: {c}" for b, c in hist))

    n_filled = con.execute(
        "SELECT COUNT(*) FROM measurement WHERE source='camera' AND cloud_pct IS NOT NULL"
    ).fetchone()[0]
    n_cam = con.execute(
        "SELECT COUNT(*) FROM measurement WHERE source='camera'"
    ).fetchone()[0]
    print(f"\nmeasurement: {n_filled}/{n_cam} Kamera-Zeilen mit cloud_pct gefuellt.")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
