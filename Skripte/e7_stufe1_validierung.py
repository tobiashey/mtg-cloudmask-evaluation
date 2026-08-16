"""[E7] Stufe-1-Validierung: Duenn/Dicht-Kategorien gegen die Blind-Masken (n = 15).

Nutzt die vorhandenen, VOR dem [E7]-Entscheid blind eingezeichneten Pixelmasken
(drei Staerken: s1 duenn / s2 normal / s3 opak) als menschliche Referenz fuer
die [E7]-Pixelkategorien (duenn = 0.86 <= R/B < 1.06, dicht = R/B >= 1.06).
Kein neues Labeln - echte Blind-Validierung (Masken datieren vor [E7]).

Fragen:
  1. R/B-Verteilung je menschlicher Klasse (kein / s1 / s2 / s3) - Separierbarkeit.
  2. Trefferquoten der [E7]-Kategorien je Klasse (Pixel-Ebene, gepoolt + je Bild).
  3. Optimale Grenzen laut Mensch (Youden-Index) fuer
       (a) kein  vs. s1+s2+s3  ("Wolke ja/nein", Referenz [E1]: Minimum 1.04)
       (b) kein+s1 vs. s2+s3   ("Dicht-Grenze", [E7] nutzt 1.06)
       (c) kein  vs. s1        ("Duenn-Untergrenze", [E7] nutzt 0.86)
     -> reine Diagnose; [E1] bleibt fix, [E7]-Anpassung nur per Autor-Entscheid.

Aufloesung 1152x648 (Maskenaufloesung, wie ERGEBNIS.md); Gueltigkeit wie
skycam.rbratio inkl. statischer Maske; ohne Sonnenmaske (v106-Konstrukt).

Aufruf:  python e7_stufe1_validierung.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Kamerabilder, Blind-Masken und das Kameramodul `skycam` sind nicht Teil
# dieses Repositories und werden unter daten/ erwartet (siehe README).
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "daten"
sys.path.insert(0, str(DATA_ROOT))

from skycam import config, rbratio  # noqa: E402

GP_DIR = DATA_ROOT / "gegenpruefung"
CAMERA_DIR = DATA_ROOT / "camera"
OUT_TXT = BASE_DIR / "e7_stufe1_ergebnis.txt"

W, H = 1152, 648
THIN_LO, DENSE = 0.86, 1.06
STRENGTH_RGB = {1: (0xFF, 0xD6, 0x33), 2: (0xFF, 0x8C, 0x1A), 3: (0xE6, 0x19, 0x4B)}


def decode_mask(data_url: str) -> np.ndarray:
    """PNG-Data-URL -> Klassenbild (0 = nicht bemalt, 1/2/3 = Staerke)."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if img.size != (W, H):
        img = img.resize((W, H), Image.NEAREST)
    arr = np.asarray(img)
    rgb, alpha = arr[:, :, :3].astype(np.int32), arr[:, :, 3]
    classes = np.zeros((H, W), dtype=np.uint8)
    painted = alpha > 128
    # Naechstgelegene Staerken-Farbe (Pinselraender koennen leicht abweichen).
    dists = np.stack(
        [np.abs(rgb - np.array(c)).sum(axis=2) for c in STRENGTH_RGB.values()], axis=0
    )
    nearest = np.argmin(dists, axis=0) + 1
    classes[painted] = nearest[painted]
    return classes


def image_rb_and_valid(orig_name: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(CAMERA_DIR / orig_name).convert("RGB").resize((W, H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float64)
    red, blue = arr[:, :, 0], arr[:, :, 2]
    valid = (red < rbratio.SATURATION_CUTOFF) & (blue > 1.0)
    valid &= arr.sum(axis=2) >= config.SKY_MASK_MIN_BRIGHTNESS_SUM
    valid &= rbratio._static_sky_mask(W, H)
    rb = np.zeros_like(red)
    np.divide(red, blue, out=rb, where=valid)
    return rb, valid


def youden_sweep(rb_neg: np.ndarray, rb_pos: np.ndarray, lo=0.6, hi=1.4, step=0.01):
    """Bester Trenn-Threshold (max. Sensitivitaet + Spezifitaet - 1)."""
    best_t, best_j = None, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        sens = (rb_pos >= t).mean()
        spec = (rb_neg < t).mean()
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def main() -> int:
    data = json.loads((GP_DIR / "gegenpruefung_masken.json").read_text(encoding="utf-8"))
    mapping = {r["gp"]: r["orig"] for r in json.loads(
        (GP_DIR / "ergebnis_rows.json").read_text(encoding="utf-8"))}

    pooled = {0: [], 1: [], 2: [], 3: []}
    lines: list[str] = []
    lines.append("[E7] Stufe-1-Validierung gegen Blind-Masken (n = 15) - " )
    lines.append("Pixel-Ebene, 1152x648, Gueltigkeit wie v106 inkl. statischer Maske\n")
    lines.append(f"{'Bild':34s} {'dicht@s2+s3':>11s} {'duenn@s1':>9s} {'kein@frei':>9s}")

    for entry in data["images"]:
        orig = mapping[entry["image"]]
        classes = decode_mask(entry["mask_png"])
        rb, valid = image_rb_and_valid(orig)
        for c in (0, 1, 2, 3):
            sel = valid & (classes == c)
            if sel.any():
                pooled[c].append(rb[sel])
        dense_hit = (rb[valid & (classes >= 2)] >= DENSE).mean() if (valid & (classes >= 2)).any() else float("nan")
        thin_hit = ((rb[valid & (classes == 1)] >= THIN_LO) & (rb[valid & (classes == 1)] < DENSE)).mean() if (valid & (classes == 1)).any() else float("nan")
        clear_ok = (rb[valid & (classes == 0)] < THIN_LO).mean() if (valid & (classes == 0)).any() else float("nan")
        lines.append(f"{orig:34s} {100*dense_hit:10.1f}% {100*thin_hit:8.1f}% {100*clear_ok:8.1f}%")

    pool = {c: (np.concatenate(v) if v else np.array([])) for c, v in pooled.items()}
    lines.append("\nR/B-Verteilung je menschlicher Klasse (gepoolt ueber 15 Bilder):")
    lines.append(f"{'Klasse':22s} {'n Pixel':>10s} {'p5':>7s} {'p25':>7s} {'p50':>7s} {'p75':>7s} {'p95':>7s}")
    for c, label in [(0, "kein (frei)"), (1, "s1 duenn"), (2, "s2 normal"), (3, "s3 opak")]:
        v = pool[c]
        lines.append(
            f"{label:22s} {len(v):10d} " + " ".join(f"{np.percentile(v, p):7.3f}" for p in (5, 25, 50, 75, 95))
        )

    lines.append("\nTrefferquoten der [E7]-Kategorien (gepoolt):")
    for c, label in [(0, "kein (frei)"), (1, "s1 duenn"), (2, "s2 normal"), (3, "s3 opak")]:
        v = pool[c]
        in_thin = ((v >= THIN_LO) & (v < DENSE)).mean()
        in_dense = (v >= DENSE).mean()
        below = (v < THIN_LO).mean()
        lines.append(f"{label:22s} -> klar {100*below:5.1f} % | duenn {100*in_thin:5.1f} % | dicht {100*in_dense:5.1f} %")

    lines.append("\nOptimale Grenzen laut Mensch (Youden-Index, Schrittweite 0.01):")
    any_cloud = np.concatenate([pool[1], pool[2], pool[3]])
    not_dense = np.concatenate([pool[0], pool[1]])
    dense_h = np.concatenate([pool[2], pool[3]])
    for label, neg, pos, current in [
        ("(a) Wolke ja/nein     ", pool[0], any_cloud, "[E1]-Sweep-Min. 1.04, prod. 1.06"),
        ("(b) Dicht-Grenze      ", not_dense, dense_h, "[E7] nutzt 1.06"),
        ("(c) Duenn-Untergrenze ", pool[0], pool[1], "[E7] nutzt 0.86"),
    ]:
        t, j = youden_sweep(neg, pos)
        lines.append(f"{label} optimal {t:.2f} (Youden {j:.3f})  | aktuell: {current}")

    text = "\n".join(lines)
    print(text)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    print(f"\n-> {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
