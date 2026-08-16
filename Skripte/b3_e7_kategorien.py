"""[E7] — Zwei Wolkenkategorien + Lux-Referenz (Entscheid des Autors, 04.08.2026).

Warum: Die binaere R/B-Klassifikation (v106) beantwortet "Wolke ja/nein", die
Beschattungsautomation braucht aber "blockiert die Bewoelkung die Sonne?".
Halo und duenne Cirren zaehlen als Wolke, lassen aber den Grossteil der
Einstrahlung durch. Deshalb werden zwei Kategorien unterschieden und je Slot
eine energetische Referenz aus dem BH1750-Lux-Sensor abgeleitet
(Begruendung/Kalibrierung: Kap. 4.3 der Arbeit).

Kategorien (identische Gueltigkeitsmaske wie v106, [E1]).
AMENDMENT 05.08.2026 (Stufe-1-Validierung an den Blind-Masken,
e7_stufe1_ergebnis.txt): die urspruenglichen Grenzen (0.86-1.06)
massen ueberwiegend Glow statt duenner Bewoelkung (41 % der freien
Himmelspixel im Band; menschlich-duenne Pixel median bei R/B 1.108).
Neue Grenzen laut Youden-Analyse (Dicht-Optimum 1.14):
  duenn  = 1.06 <= R/B < 1.14     -> schwaecht ab, blendet nicht aus
  dicht  = R/B >= 1.14            -> blockiert die Sonne ueberwiegend
  duenn + dicht = exakt die v106-Wolkenflaeche ([E1]) - die Kategorien
  partitionieren die Wolkenklasse (Konsistenz wird geprueft).

Lux-Flag sun_unobscured (nur im Automations-Fenster definiert):
  sun_in_frame UND Sonnenelevation >= 15 Grad UND Lux vorhanden:
    lux >= 20000 -> 1 (Sonne unverdeckt), sonst 0. Ausserhalb: NULL
  (Sensor ist gerichtet montiert; klare Vormittage bleiben am Sensor dunkel).

Ausgaben:
  - Tabelle camera_e7 + Export camera_e7.csv
  - Zwei-Farben-Overlays: overlays/e7_zwei_kategorien/
    (dicht = Gruen, duenn = Gelb, ungueltig abgedunkelt; Fusszeile mit beiden
    Anteilen, Lux und Sonnen-Status)

Aufruf:
  python b3_e7_kategorien.py --samples   # 4 Beispielbilder
  python b3_e7_kategorien.py             # alle Bilder + DB + CSV
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Kamerabilder, Analyse-DB und das Kameramodul `skycam` sind nicht Teil dieses
# Repositories und werden unter daten/ erwartet (siehe README).
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "daten"
sys.path.insert(0, str(DATA_ROOT))

from skycam import config, daylight, rbratio  # noqa: E402

FREEZE_CAMERA = DATA_ROOT / "camera"
ANALYSIS_DB = BASE_DIR / "cloudhub_analysis.db"
OUT_DIR = BASE_DIR / "overlays" / "e7_zwei_kategorien"
SAMPLE_DIR = BASE_DIR / "overlays_beispiele"
CSV_PATH = BASE_DIR / "camera_e7.csv"

THIN_LO = 1.06       # Untergrenze duenn = v106-Hauptschwelle ([E1]); Amendment 05.08.
DENSE = 1.14         # Dicht-Grenze (Youden-Optimum der Blind-Masken-Validierung)
LUX_THRESHOLD = 20000.0  # Kalibrierung siehe [E7] (trennt klar/bedeckt 94 %/87 %)
ELEV_MIN_DEG = 15.0

DENSE_RGB = (0, 210, 60)     # Gruen (Stil-Freigabe des Autors)
THIN_RGB = (255, 210, 0)     # Gelb - zweite Kategorie, kommt in den Szenen nicht vor
ALPHA = 0.5
INVALID_BRIGHTNESS = 0.35
INVALID_DESAT = 0.5
BAR_HEIGHT = 48
JPEG_QUALITY = 82

SAMPLES = [
    "skycam_20260709T065001Z.jpg",   # klar
    "skycam_20260703T103001Z.jpg",   # durchbrochen
    "skycam_20260703T095000Z.jpg",   # Overcast
    "skycam_20260708T133001Z.jpg",   # Sonne + Flare (Halo -> duenn)
]

CONFIG_JSON = json.dumps(
    {
        "decision": "E7 (Amendment 2026-08-05)",
        "dense_threshold": DENSE,
        "thin_range": [THIN_LO, DENSE],
        "lux_threshold": LUX_THRESHOLD,
        "lux_window": f"sun_in_frame AND elevation >= {ELEV_MIN_DEG}",
        "validity": "wie v106 ([E1]): rot<250, blau>1, R+G+B>=110, statische Maske",
        "amendment_basis": "Stufe-1-Blind-Masken-Validierung (e7_stufe1_ergebnis.txt)",
        "date": "2026-08-05",
    },
    sort_keys=True,
)


def _font() -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, 28)
        except OSError:
            continue
    return ImageFont.load_default()


def compute(arr: np.ndarray):
    """(valid, dense, thin) fuer die [E7]-Kategorien."""
    height, width = arr.shape[:2]
    red, blue = arr[:, :, 0], arr[:, :, 2]
    valid = (red < rbratio.SATURATION_CUTOFF) & (blue > 1.0)
    if config.SKY_MASK_ENABLED:
        valid &= arr.sum(axis=2) >= config.SKY_MASK_MIN_BRIGHTNESS_SUM
    if config.STATIC_SKY_MASK_ENABLED:
        valid &= rbratio._static_sky_mask(width, height)
    rb = np.zeros_like(red)
    np.divide(red, blue, out=rb, where=valid)
    dense = valid & (rb >= DENSE)
    thin = valid & (rb >= THIN_LO) & (rb < DENSE)
    return valid, dense, thin


def render(img_path: Path, meta: dict) -> tuple[Image.Image, float, float, int]:
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    valid, dense, thin = compute(arr)
    n_valid = int(valid.sum())
    dense_pct = 100.0 * dense.sum() / n_valid if n_valid else float("nan")
    thin_pct = 100.0 * thin.sum() / n_valid if n_valid else float("nan")

    out = arr.copy()
    invalid = ~valid
    gray = arr.mean(axis=2, keepdims=True)
    desat = (1.0 - INVALID_DESAT) * arr + INVALID_DESAT * gray
    out[invalid] = desat[invalid] * INVALID_BRIGHTNESS
    out[dense] = (1.0 - ALPHA) * arr[dense] + ALPHA * np.array(DENSE_RGB, dtype=np.float64)
    out[thin] = (1.0 - ALPHA) * arr[thin] + ALPHA * np.array(THIN_RGB, dtype=np.float64)
    overlay = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    w, h = overlay.size
    canvas = Image.new("RGB", (w, h + BAR_HEIGHT), (0, 0, 0))
    canvas.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    cy = h + BAR_HEIGHT // 2

    when = rbratio._parse_capture_time_from_path(img_path)
    ts_txt = when.strftime("%d.%m. %H:%M UTC") if when else "Zeit unbekannt"
    text = (
        f"{img_path.name} · {ts_txt} · dicht: "
        f"{dense_pct:.1f} %" if n_valid else f"{img_path.name} · {ts_txt} · dicht: n/a"
    )
    if n_valid:
        text += f" · dünn: {thin_pct:.1f} %"
    if meta.get("lux") is not None:
        text += f" · {meta['lux']:.0f} lx"
    if meta.get("sun_unobscured") is not None:
        text += " · Sonne: " + ("durch" if meta["sun_unobscured"] else "verdeckt")
    draw.text((16, cy), text, fill=(255, 255, 255), font=font, anchor="lm")
    if meta.get("invalid_reason"):
        x_end = 16 + draw.textlength(text, font=font)
        draw.text((x_end + 24, cy),
                  f"UNGUELTIG ({meta['invalid_reason'].split('(')[0]})",
                  fill=(255, 70, 70), font=font, anchor="lm")

    # Legende rechts: dicht / duenn / nicht gewertet.
    sw = 22
    entries = [
        (tuple(int((1 - ALPHA) * 200 + ALPHA * c) for c in DENSE_RGB), "dicht"),
        (tuple(int((1 - ALPHA) * 200 + ALPHA * c) for c in THIN_RGB), "dünn"),
        ((64, 64, 64), "nicht gewertet"),
    ]
    x = w - 16
    for color, label in reversed(entries):
        tw = draw.textlength(label, font=font)
        x -= tw
        draw.text((x, cy), label, fill=(255, 255, 255), font=font, anchor="lm")
        x -= sw + 8
        draw.rectangle([x, cy - sw // 2, x + sw, cy + sw // 2], fill=color,
                       outline=(255, 255, 255))
        x -= 28
    return canvas, dense_pct, thin_pct, n_valid


def load_meta(con: sqlite3.Connection) -> dict[str, dict]:
    """Slot-ts, sun_in_frame, invalid, Lux je Bilddatei aus der Analyse-DB."""
    meta: dict[str, dict] = {}
    for f, ts, sif, inv, reason, pct in con.execute(
        "SELECT image_file, ts, sun_in_frame, invalid, invalid_reason, cloud_pct "
        "FROM camera_classification WHERE variant='v106'"
    ):
        meta[f] = {
            "ts": ts, "sun_in_frame": sif, "invalid": inv,
            "invalid_reason": reason, "v106_pct": pct,
            "lux": None, "sun_elevation_deg": None, "sun_unobscured": None,
        }
    lux_by_ts = dict(con.execute(
        "SELECT ts, lux FROM sensor_reading WHERE lux IS NOT NULL"
    ))
    from datetime import datetime
    for m in meta.values():
        if not m["ts"]:
            continue
        m["lux"] = lux_by_ts.get(m["ts"])
        elev = daylight.sun_elevation_deg(datetime.fromisoformat(m["ts"]))
        m["sun_elevation_deg"] = round(elev, 2)
        if m["sun_in_frame"] and elev >= ELEV_MIN_DEG and m["lux"] is not None:
            m["sun_unobscured"] = int(m["lux"] >= LUX_THRESHOLD)
    return meta


def main() -> int:
    samples_only = "--samples" in sys.argv
    out_dir = SAMPLE_DIR if samples_only else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(ANALYSIS_DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS camera_e7 (
            image_file        TEXT PRIMARY KEY,
            ts                TEXT,
            cloud_dense_pct   REAL,
            cloud_thin_pct    REAL,
            n_valid           INTEGER,
            lux               REAL,
            sun_elevation_deg REAL,
            sun_in_frame      INTEGER,
            sun_unobscured    INTEGER,
            invalid           INTEGER,
            invalid_reason    TEXT,
            config_json       TEXT
        )"""
    )
    meta = load_meta(con)

    if samples_only:
        images = [FREEZE_CAMERA / n for n in SAMPLES]
    else:
        images = sorted(FREEZE_CAMERA.glob("skycam_*.jpg"))

    for i, img_path in enumerate(images, 1):
        m = meta.get(img_path.name, {})
        suffix = "_e7.jpg" if samples_only else "_overlay.jpg"
        dest = out_dir / f"{img_path.stem}{suffix}"
        have_row = con.execute(
            "SELECT 1 FROM camera_e7 WHERE image_file=?", (img_path.name,)
        ).fetchone()
        if dest.exists() and have_row and not samples_only:
            continue
        try:
            canvas, dense_pct, thin_pct, n_valid = render(img_path, m)
        except OSError as exc:
            print(f"  UEBERSPRUNGEN {img_path.name}: {exc}")
            if not samples_only:
                con.execute(
                    "INSERT OR REPLACE INTO camera_e7 VALUES (?,?,NULL,NULL,NULL,?,?,?,?,1,?,?)",
                    (img_path.name, m.get("ts"), m.get("lux"), m.get("sun_elevation_deg"),
                     m.get("sun_in_frame"), m.get("sun_unobscured"),
                     m.get("invalid_reason"), CONFIG_JSON),
                )
            continue
        # Konsistenz: duenn + dicht == v106 (Partition der [E1]-Wolkenklasse).
        if m.get("v106_pct") is not None and n_valid and abs(
            (dense_pct + thin_pct) - m["v106_pct"]
        ) > 0.05:
            raise SystemExit(
                f"ABBRUCH: {img_path.name} duenn+dicht {dense_pct + thin_pct:.3f} "
                f"!= v106 {m['v106_pct']:.3f}"
            )
        canvas.save(dest, quality=JPEG_QUALITY)
        if not samples_only:
            con.execute(
                "INSERT OR REPLACE INTO camera_e7 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (img_path.name, m.get("ts"),
                 None if not n_valid else round(dense_pct, 4),
                 None if not n_valid else round(thin_pct, 4),
                 n_valid, m.get("lux"), m.get("sun_elevation_deg"),
                 m.get("sun_in_frame"), m.get("sun_unobscured"),
                 m.get("invalid", 0), m.get("invalid_reason"), CONFIG_JSON),
            )
        if samples_only:
            print(f"{dest.name}: dicht {dense_pct:.1f} % · dünn {thin_pct:.1f} % · "
                  f"lux {m.get('lux')} · sun_unobscured {m.get('sun_unobscured')}")
        elif i % 200 == 0 or i == len(images):
            con.commit()
            print(f"  {i}/{len(images)} ...", flush=True)
    con.commit()

    if not samples_only:
        rows = con.execute(
            "SELECT image_file, ts, cloud_dense_pct, cloud_thin_pct, n_valid, lux, "
            "sun_elevation_deg, sun_in_frame, sun_unobscured, invalid, invalid_reason "
            "FROM camera_e7 ORDER BY image_file"
        ).fetchall()
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_file", "ts", "cloud_dense_pct", "cloud_thin_pct", "n_valid",
                        "lux", "sun_elevation_deg", "sun_in_frame", "sun_unobscured",
                        "invalid", "invalid_reason"])
            w.writerows(rows)
        print(f"CSV: {CSV_PATH} ({len(rows)} Zeilen)")

        print("\n=== Kurzstatistik [E7] (gueltige Bilder) ===")
        q = con.execute(
            "SELECT COUNT(*), AVG(cloud_dense_pct), AVG(cloud_thin_pct) FROM camera_e7 "
            "WHERE invalid=0 AND cloud_dense_pct IS NOT NULL"
        ).fetchone()
        print(f"n={q[0]}, dicht mean={q[1]:.1f} %, duenn mean={q[2]:.1f} %")
        for v, label in [(1, "Sonne durch"), (0, "Sonne verdeckt")]:
            q = con.execute(
                "SELECT COUNT(*), AVG(cloud_dense_pct), AVG(cloud_thin_pct) FROM camera_e7 "
                "WHERE invalid=0 AND sun_unobscured=?", (v,)
            ).fetchone()
            print(f"{label}: n={q[0]}, dicht mean={q[1]:.1f} %, duenn mean={q[2]:.1f} %")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
