"""Abbildung 5.3b: Fehlentscheidungen je Quelle (fig53_decision_errors).

Divergierende Balken: unnoetige Beschattungen (FP, nach links) vs. verpasste
Beschattungen (FN, nach rechts) je Quelle, sortiert nach Accuracy; Accuracy
am rechten Rand annotiert. Werte aus out/e2_konfusionsmatrizen.csv.
Farben aus der Richtlinien-Palette (Technik-Blau hell/dunkel — ueber
Helligkeit auch in Graustufen unterscheidbar).

Aufruf:  python fig53_decision_errors.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUSWERTUNG_DIR = Path(__file__).resolve().parent
OUT_DIR = AUSWERTUNG_DIR / "out"
FIGURES_DIR = AUSWERTUNG_DIR.parent / "Satelitten" / "figures"

CM = 1 / 2.54


def main() -> None:
    conf = pd.read_csv(OUT_DIR / "e2_konfusionsmatrizen.csv")
    conf = conf.sort_values("accuracy")  # beste Quelle oben (barh von unten)
    n = int(conf["N"].iloc[0])
    y = np.arange(len(conf))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
        }
    )

    fig, ax = plt.subplots(figsize=(12.2 * CM, 5.6 * CM), constrained_layout=True)
    ax.barh(y, -conf["unnoetig_beschattet_FP"], color="#8fb3d9", height=0.62,
            label="unnecessary shading (FP)")
    ax.barh(y, conf["verpasst_FN"], color="#4a6fa5", height=0.62,
            label="missed shading (FN)")
    ax.axvline(0, color="#3a3a3a", linewidth=0.8)

    for yi, (_, row) in zip(y, conf.iterrows()):
        ax.text(-row["unnoetig_beschattet_FP"] - 6, yi,
                str(int(row["unnoetig_beschattet_FP"])),
                ha="right", va="center", fontsize=7.0, color="#3a3a3a")
        ax.text(row["verpasst_FN"] + 6, yi, str(int(row["verpasst_FN"])),
                ha="left", va="center", fontsize=7.0, color="#3a3a3a")
        ax.text(200, yi, f"accuracy {row['accuracy']:.2f}".replace("0.", "."),
                ha="left", va="center", fontsize=7.0, style="italic",
                color="#5a5a5a")

    ax.set_yticks(y)
    ax.set_yticklabels(conf["source"])
    ax.set_xlim(-440, 265)
    ticks = np.arange(-400, 201, 100)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(t)) for t in ticks])
    ax.set_xlabel(f"wrong decisions out of N = {n} paired measurements")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"fig53_decision_errors.{suffix}", dpi=300)
    print(f"OK -> {FIGURES_DIR / 'fig53_decision_errors.{png,pdf}'}")


if __name__ == "__main__":
    main()
