"""B3 — Visualisierungs-Overlays der Kamera-Klassifikation (finale Konfiguration v106).

Erzeugt je Kamera-JPG des Freeze ein Overlay-Bild, das die R/B-Klassifikation
visuell nachvollziehbar macht (Abbildungen fuer die Arbeit + Sichtpruefung
jedes Datenpunkts):

  - Wolkenpixel (R/B >= 1.06, gueltig): halbtransparentes Orange (255,140,26),
    45 % Deckkraft.
  - Ungueltige Bereiche (statische Hauskanten-Maske, Saettigung, dunkler
    Vordergrund): auf 35 % Helligkeit abgedunkelt, leicht entsaettigt.
  - Klarer Himmel: unveraendert.
  - Unten angesetzte schwarze Leiste (48 px) mit weissem Monospace-Text:
    Dateiname, Zeitstempel, Wolkenanteil, ggf. "Sonne im Bild".

Die Pixellogik repliziert exakt ``skycam.rbratio.classify_image`` (gleiche
Konstanten aus skycam.config); der resultierende cloud_pct wird gegen die
Werte der Batch-Klassifikation (camera_classification, v106) verifiziert.

Aufruf:
  python b3_overlays.py --samples     # 4 Beispielbilder -> overlays_beispiele/
  python b3_overlays.py               # alle Bilder      -> overlays/
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Kamerabilder, Analyse-DB und das Kameramodul `skycam` sind nicht Teil dieses
# Repositories und werden unter daten/ erwartet (siehe README).
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "daten"
sys.path.insert(0, str(DATA_ROOT))

from skycam import config, rbratio  # noqa: E402

FREEZE_CAMERA = DATA_ROOT / "camera"
# Seit [E7] liegen die v106-Overlays im Unterordner v106/ (daneben:
# e7_zwei_kategorien/ aus b3_e7_kategorien.py).
OUT_DIR = BASE_DIR / "overlays" / "v106"
SAMPLE_DIR = BASE_DIR / "overlays_beispiele"
ANALYSIS_DB = BASE_DIR / "cloudhub_analysis.db"

THRESHOLD = 1.06          # Hauptvariante v106 ([E1]); Sonnenmaske aus.
# Auffaelliges Gruen (Autor-Feedback 04.08.): kommt in den Szenen nicht vor
# (Himmel/Wolken/dunkle Daecher), dadurch eindeutig als Markierung erkennbar.
CLOUD_RGB = (0, 210, 60)
CLOUD_ALPHA = 0.5
INVALID_BRIGHTNESS = 0.35  # Faktor fuer ungueltige Bereiche
INVALID_DESAT = 0.5        # Anteil Graumischung vor dem Abdunkeln
BAR_HEIGHT = 48
JPEG_QUALITY = 82

SAMPLES = [
    "skycam_20260709T065001Z.jpg",   # klar (gp_02)
    "skycam_20260703T103001Z.jpg",   # durchbrochen (gp_03)
    "skycam_20260703T095000Z.jpg",   # Overcast (gp_05)
    "skycam_20260708T133001Z.jpg",   # Sonne + Flare (gp_06)
]


def _font() -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, 28)
        except OSError:
            continue
    return ImageFont.load_default()


def compute_masks(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(valid, cloud) fuer die v106-Konfiguration - identisch zu rbratio."""
    height, width = arr.shape[:2]
    red, blue = arr[:, :, 0], arr[:, :, 2]
    valid = (red < rbratio.SATURATION_CUTOFF) & (blue > 1.0)
    if config.SKY_MASK_ENABLED:
        valid &= arr.sum(axis=2) >= config.SKY_MASK_MIN_BRIGHTNESS_SUM
    if config.STATIC_SKY_MASK_ENABLED:
        valid &= rbratio._static_sky_mask(width, height)
    rb = np.zeros_like(red)
    np.divide(red, blue, out=rb, where=valid)
    cloud = valid & (rb >= THRESHOLD)
    return valid, cloud


def render_overlay(
    img_path: Path, sun_in_frame: bool, invalid_reason: str | None = None
) -> tuple[Image.Image, float]:
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    valid, cloud = compute_masks(arr)

    n_valid = int(valid.sum())
    cloud_pct = 100.0 * cloud.sum() / n_valid if n_valid else float("nan")

    out = arr.copy()
    # Ungueltig: leicht entsaettigen, dann abdunkeln.
    invalid = ~valid
    gray = arr.mean(axis=2, keepdims=True)
    desat = (1.0 - INVALID_DESAT) * arr + INVALID_DESAT * gray
    out[invalid] = desat[invalid] * INVALID_BRIGHTNESS
    # Wolke: halbtransparentes Orange.
    orange = np.array(CLOUD_RGB, dtype=np.float64)
    out[cloud] = (1.0 - CLOUD_ALPHA) * arr[cloud] + CLOUD_ALPHA * orange

    overlay = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    # Textleiste unten ansetzen (Bildinhalt bleibt vollstaendig sichtbar).
    w, h = overlay.size
    canvas = Image.new("RGB", (w, h + BAR_HEIGHT), (0, 0, 0))
    canvas.paste(overlay, (0, 0))
    when = rbratio._parse_capture_time_from_path(img_path)
    ts_txt = when.strftime("%d.%m. %H:%M UTC") if when else "Zeit unbekannt"
    pct_txt = f"{cloud_pct:.1f} %" if n_valid else "n/a"
    text = f"{img_path.name} · {ts_txt} · Wolken: {pct_txt}"
    if sun_in_frame:
        text += " · Sonne im Bild"
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((16, h + BAR_HEIGHT // 2), text, fill=(255, 255, 255),
              font=font, anchor="lm")
    if invalid_reason:
        # Ungueltig-Markierung in Rot direkt hinter dem Text.
        x_end = 16 + draw.textlength(text, font=font)
        draw.text((x_end + 24, h + BAR_HEIGHT // 2),
                  f"UNGUELTIG ({invalid_reason.split('(')[0]})",
                  fill=(255, 70, 70), font=font, anchor="lm")

    # Mini-Legende rechts in der Fusszeile: Farbfeld + Label.
    sw = 22  # Kantenlaenge der Farbfelder
    cy = h + BAR_HEIGHT // 2
    entries = [
        (tuple(int((1 - CLOUD_ALPHA) * 200 + CLOUD_ALPHA * c) for c in CLOUD_RGB), "Wolke"),
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
    return canvas, cloud_pct


def main() -> int:
    samples_only = "--samples" in sys.argv
    out_dir = SAMPLE_DIR if samples_only else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Referenzwerte + sun_in_frame + invalid_reason aus der Batch-Klassifikation (v106).
    ref: dict[str, tuple[float | None, int]] = {}
    reasons: dict[str, str] = {}
    if ANALYSIS_DB.exists():
        con = sqlite3.connect(f"file:{ANALYSIS_DB}?mode=ro", uri=True)
        try:
            for f, p, s, reason in con.execute(
                "SELECT image_file, cloud_pct, sun_in_frame, invalid_reason "
                "FROM camera_classification WHERE variant='v106'"
            ):
                ref[f] = (p, s)
                if reason:
                    reasons[f] = reason
        except sqlite3.OperationalError:
            try:  # aeltere DB ohne invalid_reason-Spalte
                for f, p, s in con.execute(
                    "SELECT image_file, cloud_pct, sun_in_frame "
                    "FROM camera_classification WHERE variant='v106'"
                ):
                    ref[f] = (p, s)
            except sqlite3.OperationalError:
                pass  # Tabelle existiert noch nicht (Batch laeuft) -> Flag selbst rechnen
        con.close()

    if samples_only:
        images = [FREEZE_CAMERA / n for n in SAMPLES]
    else:
        images = sorted(FREEZE_CAMERA.glob("skycam_*.jpg"))

    max_dev = 0.0
    for i, img_path in enumerate(images, 1):
        dest = out_dir / f"{img_path.stem}_overlay.jpg"
        if dest.exists() and not samples_only:
            continue
        try:
            ref_pct, ref_sun = ref.get(img_path.name, (None, None))
            if ref_sun is None:
                r = rbratio.classify_image(img_path, threshold=THRESHOLD, apply_sun_mask=False)
                ref_sun = int(r.sun_in_frame)
                ref_pct = r.cloud_pct if r.n_valid else None
            canvas, cloud_pct = render_overlay(
                img_path, bool(ref_sun), reasons.get(img_path.name)
            )
        except OSError as exc:
            # Defekte Datei (abgebrochene Uebertragung) -> kein Overlay moeglich.
            print(f"  UEBERSPRUNGEN {img_path.name}: {exc}")
            continue
        if ref_pct is not None and not np.isnan(cloud_pct):
            dev = abs(cloud_pct - ref_pct)
            max_dev = max(max_dev, dev)
            if dev > 0.05:
                raise SystemExit(
                    f"ABBRUCH: {img_path.name} Overlay-cloud_pct {cloud_pct:.3f} != "
                    f"Batch-Wert {ref_pct:.3f} - Logik-Drift pruefen."
                )
        canvas.save(dest, quality=JPEG_QUALITY)
        if samples_only:
            print(f"{dest.name}: Wolken {cloud_pct:.1f} % (Referenz {ref_pct})")
        elif i % 200 == 0 or i == len(images):
            print(f"  {i}/{len(images)} Overlays ...", flush=True)

    print(f"Fertig: {len(images)} Bilder, max. Abweichung zur Batch-Referenz {max_dev:.4f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
