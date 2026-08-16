"""F2 (Vorab-Artefakt) — Empirische Wert-Änderungsintervalle je Quelle.

Misst je Quelle, wie oft aufeinanderfolgende 10-min-Tagfenster-Slots einen
**neuen** Wert tragen, und leitet daraus das effektive Änderungsintervall ab.
Das ist der empirische Beleg dafür, dass Bright Sky, OpenWeatherMap und
WeatherAPI.com faktisch Stundenwerte liefern, während Kamera/CLM im
10-min-Takt aktualisieren — zentral für (a) das FF1-Framing „Genauigkeit am
Entscheidungszeitpunkt" (→ Kap. 4.x/6.2), (b) den Robustheitscheck D4(d) und
(c) die effektive Aktualität in F2/Kap. 5.4 (M5-Framing: Eigenschaft des
Untersuchungsgegenstands, nicht der API-Messgüte).

Fehlerbehandlung — in die Kennzahl geht ausschließlich ein Slot-Paar ein, das
auf beiden Seiten eine **gültige Messung** trägt:

  (a) beide Slots liegen genau 10 Minuten auseinander (keine Lücke),
  (b) beide Slots tragen einen Wert (fehlende Abrufe/NULL fallen raus),
  (c) bei der Kamera zusätzlich: beide Bilder sind nicht als ungültig
      markiert (`camera_invalid`, v106 — z. B. magenta_frame).

Damit erzeugt weder ein Ausfall noch ein fehlerhaftes Bild einen künstlichen
Wertwechsel, und eine Lücke wird nicht als „konstanter Wert" gelesen.

Lesart der Kennzahl: Sie ist der **mittlere Abstand zwischen zwei neuen
Werten**, nicht die Abruflatenz. Eine Quelle, die einen Wert fachlich zu Recht
hält (wolkenloser Himmel, 0 %), liegt deshalb geringfügig über ihrem
tatsächlichen Abruftakt — bei der Kamera 11,6 statt 10,0 min. Für den
Quellenvergleich ist das unerheblich, weil alle Quellen denselben Himmel sehen.

Ausgabe: out/f2_update_intervalle.csv, out/f2_update_intervalle.txt
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).parent / "out"
PAIRED_CSV = OUT_DIR / "paired_slots.csv"

SOURCES = [
    "camera",
    "satellite_clm",
    "weather_brightsky",
    "weather_openmeteo",
    "weather_openweathermap",
    "weather_tomorrowio",
    "weather_weatherapi",
]


def main() -> None:
    df = pd.read_csv(PAIRED_CSV, parse_dates=["ts"])
    day = df[df["daylight"] == 1].sort_values("ts")
    dt_ok = day["ts"].diff().dt.total_seconds().eq(600)  # nur lückenlose Folgen
    cam_bad = day["camera_invalid"].astype(int).eq(1)    # ungültiges Kamerabild

    n_day = len(day)
    rows = []
    for s in SOURCES:
        cur = day[s].astype(float)
        prev = cur.shift()
        valid = dt_ok & cur.notna() & prev.notna()
        if s == "camera":
            valid &= ~cam_bad & ~cam_bad.shift(fill_value=False)

        bad = cur.isna() | cam_bad if s == "camera" else cur.isna()
        n_bad = int(bad.sum())
        chg = (cur != prev) & valid
        frac = chg.sum() / valid.sum() * 100
        rows.append(
            {
                "source": s,
                "n_slots_excluded": n_bad,
                "n_slot_pairs": int(valid.sum()),
                "n_changes": int(chg.sum()),
                "pct_changed": round(frac, 1),
                "eff_interval_min": round(10 / (frac / 100), 1) if frac > 0 else None,
            }
        )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT_DIR / "f2_update_intervalle.csv", index=False)

    text = "\n".join(
        [
            "F2 (Vorab) — Empirische Wert-Änderungsintervalle (Tagfenster-Slots, lückenlose 10-min-Folgen)",
            "Skript: f2_update_intervalle.py | Grundlage: out/paired_slots.csv (C2)",
            "",
            tbl.to_string(index=False),
            "",
            f"Grundgesamtheit: {n_day} Tagfenster-Slots (Sonnenelevation >= 5°).",
            "n_slots_excluded = Slots ohne gültige Messung (fehlender Abruf/NULL,",
            "bei der Kamera zusätzlich als ungültig markierte Bilder). Sie bilden kein Paar",
            "und erzeugen daher weder einen künstlichen Wechsel noch eine künstliche Konstanz.",
            "",
            "Lesart: pct_changed = Anteil Folge-Slots mit neuem Wert;",
            "eff_interval_min = 10 / (pct_changed/100) = mittlerer Abstand zwischen zwei neuen",
            "Werten. Das ist keine Abruflatenz: Eine Quelle, die ihren Wert fachlich zu Recht",
            "hält (z. B. 0 % bei wolkenlosem Himmel), liegt geringfügig über ihrem Abruftakt.",
            "",
            "Befund: Bright Sky, OpenWeatherMap und WeatherAPI.com liefern faktisch Stundenwerte",
            "(69-75 min), Kamera und MTG CLM aktualisieren im bzw. nahe am 10-min-Takt.",
        ]
    )
    (OUT_DIR / "f2_update_intervalle.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
