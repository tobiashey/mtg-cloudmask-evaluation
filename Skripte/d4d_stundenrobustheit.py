"""D4(d) — Robustheitscheck Aggregationsebenen: 30-min- und 60-min-Mittel
(Amendment 05.08.2026, erweitert um 30 min auf Wunsch des Autors).

Motivation (zweigleisig):
  1. Drei APIs (Bright Sky, OpenWeatherMap, WeatherAPI.com) liefern faktisch
     Stundenwerte (empirisch ~70 min, siehe `f2_update_intervalle.py`) — der
     10-min-Slot-Vergleich enthält damit einen Staleness-Anteil.
  2. Die Kamera sieht nicht den ganzen Himmel (directional sky-view) und ist
     eine Momentaufnahme; zeitliche Mittelung wirkt wie eine räumliche
     Glättung (Wolkenfelder ziehen durch das Sichtfeld) und nähert das
     Kamera-Konstrukt dem Flächenmittel der übrigen Quellen an.

Vorgehen: Complete-Case-Slots [E5] werden je Quelle und Kamera über Fenster
von 30 bzw. 60 min gemittelt (nur Fenster, die mehr als zur Hälfte gefüllt
sind: ≥ 2/3 bzw. ≥ 3/6 Slots). Metriken und CIs identisch zu D1
(Tagesblock-Bootstrap [E3], Seed 858157; Import aus `d1_ff1_metriken`).
Erwartung: MAE sinkt moderat, Rangfolge bleibt — dann ist die Slot-Ebene
als Hauptanalyse („Genauigkeit am Entscheidungszeitpunkt") abgesichert.

Ausgabe: out/d4d_aggregation.csv (alle Ebenen inkl. 10-min-Referenz aus D1),
         out/d4d_report.txt
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from d1_ff1_metriken import OUT_DIR, PAIRED_CSV, SEED, SOURCES, evaluate

WINDOWS = [("30 min", "30min", 2), ("60 min", "60min", 3)]  # (Label, floor-Freq, Min-Slots)


def main() -> None:
    df = pd.read_csv(PAIRED_CSV, parse_dates=["ts"])
    cc = df[df["complete_case"] == 1].copy()
    cols = ["camera"] + list(SOURCES)

    tables = []
    d1 = pd.read_csv(OUT_DIR / "d1_ff1_haupttabelle.csv")
    d1.insert(0, "aggregation", "10 min (D1)")
    tables.append(d1)

    for label, freq, min_slots in WINDOWS:
        cc["win"] = cc["ts"].dt.floor(freq)
        grp = cc.groupby("win")
        agg = grp[cols].mean()
        agg["n_slots"] = grp.size()
        agg = agg[agg["n_slots"] >= min_slots].reset_index()
        agg["date"] = agg["win"].dt.date.astype(str)

        rng = np.random.default_rng(SEED)
        tbl = evaluate(agg, "camera", rng)
        tbl.insert(0, "aggregation", f"{label} (N = {len(agg)})")
        tables.append(tbl)

    full = pd.concat(tables, ignore_index=True)
    full.to_csv(OUT_DIR / "d4d_aggregation.csv", index=False)

    # Kompakte Gegenüberstellung MAE + r über die Ebenen
    comp = full.pivot(index="source", columns="aggregation", values=["MAE", "Pearson r"])
    comp = comp.reindex([s for s in SOURCES.values()])

    report = [
        "D4(d) — Robustheitscheck Aggregationsebenen (Mittel der Complete-Case-Slots je Fenster)",
        f"Fenster > 50 % gefüllt erforderlich; Metriken/CIs wie D1 [E3], Seed {SEED}.",
        "",
        full.to_string(index=False),
        "",
        "Kompakt (MAE in pp / Pearson r je Aggregationsebene):",
        comp.round(2).to_string(),
        "",
        "Skript: d4d_stundenrobustheit.py",
    ]
    (OUT_DIR / "d4d_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
