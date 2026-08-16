"""Abbildung 5.3a: Konfusionsmatrizen CLM und Bright Sky (fig53_confusion_matrices).

Zwei annotierte 2x2-Matrizen (Satellit und Baseline) gegen die
Kamera-Referenzentscheidung; Werte aus out/e2_konfusionsmatrizen.csv
(keine Neuberechnung). Zellenfarbe = Anteil an N (gleiche Blau-Skala wie
fig52_correlation_matrix, helligkeitsmonoton -> s/w-tauglich).

Aufruf:  python fig53_confusion_matrices.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

PANELS = ["MTG FCI L2 CLM", "Bright Sky (DWD)"]
CM = 1 / 2.54


def main() -> None:
    conf = pd.read_csv(OUT_DIR / "e2_konfusionsmatrizen.csv").set_index("source")
    cmap = LinearSegmentedColormap.from_list("blues_lni", ["#e4ecf3", "#0072b2"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2 * CM, 6.0 * CM),
                             constrained_layout=True)
    for ax, source in zip(axes, PANELS):
        row = conf.loc[source]
        n = row["N"]
        # Zellen: Zeile = Entscheidung der Quelle, Spalte = Kamera-Referenz
        cells = [
            [row["korrekt_beschattet_TP"], row["unnoetig_beschattet_FP"]],
            [row["verpasst_FN"], row["korrekt_nicht_TN"]],
        ]
        for i in range(2):
            for j in range(2):
                val = int(cells[i][j])
                frac = val / n
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=cmap(frac / 0.55),
                                           edgecolor="white", linewidth=1.2))
                ax.text(j, i, f"{val}\n{100 * frac:.1f} %",
                        ha="center", va="center", fontsize=8,
                        color="white" if frac / 0.55 > 0.62 else "#3a3a3a")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["shade", "no shade"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["shade", "no shade"])
        ax.set_xlabel("camera reference")
        ax.set_ylabel("source decision" if source == PANELS[0] else "")
        ax.set_title(source, fontsize=8)
        ax.tick_params(length=0)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig53_confusion_matrices.{suffix}", dpi=300)
    print(f"OK -> {FIGURES_DIR / 'fig53_confusion_matrices.{png,pdf}'}")


if __name__ == "__main__":
    main()
