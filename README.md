# Auswertung: Satellitenbasierte Bewölkungsschätzung vs. Wetter-APIs

Auswerteskripte und Ergebnistabellen einer 34-tägigen Vergleichsmesskampagne (Juli/August 2026, Raum Stuttgart). Verglichen wird der lokale Bewölkungsgrad aus der MTG FCI Level-2 Cloud Mask gegen fünf Wetter-APIs, als Bodenwahrheit dient eine gerichtete Himmelskamera.

Das Repository begleitet eine Forschungsarbeit im Studiengang Embedded Systems and Digital Technologies. Es dokumentiert, wie die berichteten Kennzahlen zustande kommen, und enthält die aggregierten Ergebnisse.

## Inhalt

| Pfad | Inhalt |
| --- | --- |
| `*.py` | Auswerteskripte in der Reihenfolge der Analysekette (Präfixe b3, c, d, e, f, fig) |
| `out/` | Ergebnistabellen (CSV), Kurzberichte (TXT) und Abbildungen (PNG/PDF) der Arbeit |

## Analysekette

Die Skripte bauen aufeinander auf; jede Stufe liest die Ausgaben der vorherigen aus `out/`.

| Stufe | Skript | Ergebnis |
| --- | --- | --- |
| B3 | `b3_batch_klassifikation.py` | R/B-Klassifikation der Kamerabilder (Schwellwert 1,06; Sensitivitätsvariante 0,86 mit Sonnenmaske) |
| B3 | `b3_fehlbilder_scan.py` | Erkennung defekter Kameraframes (ausgefallener Grünkanal) |
| B3 | `b3_e7_kategorien.py` | Wolkenkategorien dünn/dicht und Lux-Referenz „Sonne unverdeckt" |
| B3 | `b3_overlays.py` | Klassifikations-Overlays zur Sichtprüfung |
| C2 | `c2_paired_slots.py` | Gepaarter Slot-Datensatz mit Inklusionskriterien (N = 2 646) |
| C3 | `c3_deskriptiv.py` | Deskriptive Statistik je Quelle |
| D1 | `d1_ff1_metriken.py` | MAE, RMSE, Bias, Pearson-r mit Tagesblock-Bootstrap (10 000 Replikationen) |
| D3 | `d3_bedingt.py` | Bedingte Fehleranalyse und Fehlerprofil über 10-pp-Bins |
| D4 | `d4d_stundenrobustheit.py` | Robustheitscheck auf 30- und 60-Minuten-Mitteln |
| D5 | `d5_quellenvergleich.py` | MAE- und Korrelationsmatrix aller Quellen untereinander |
| E | `e_ff2_entscheidungen.py` | Beschattungsentscheidungen, Konfusionsmatrizen |
| E | `e_ff2_aggregation.py` | Entscheidungen auf 30-/60-Minuten-Mitteln, Schalthäufigkeit |
| E7 | `e7_stufe1_validierung.py` | Validierung der Wolkenkategorien gegen 15 Blind-Masken |
| F1 | `f1_latenz.py` | Latenzkette des Satellitenabrufs |
| F2 | `f2_update_intervalle.py` | Empirische Wert-Änderungsintervalle je Quelle |
| — | `fig51_*.py` … `fig54_*.py` | Abbildungen der Arbeit aus den Ergebnistabellen |

## Was nicht enthalten ist

Das Repository enthält bewusst nur die Auswertung, nicht die Erhebung:

- **Rohdaten.** Kamerabilder, Satellitenszenen und die Kampagnendatenbank sind nicht enthalten. Die Stufen B3, C2, D3 und F1 lesen aus der lokalen SQLite-Datenbank `cloudhub_analysis.db` bzw. aus einem Ordner `daten/` mit den Kamerabildern und sind ohne diese nicht ausführbar. Die Stufen ab C3 arbeiten auf den CSV-Dateien in `out/` und lassen sich damit nachrechnen.
- **Erhebungsinfrastruktur.** Der Erhebungsserver, die Kamerasoftware und die Satelliten-Downloadpipeline sind nicht Teil dieses Repositories. Die B3-Skripte importieren das Kameramodul `skycam`, das ebenfalls unter `daten/` erwartet wird; sie dokumentieren hier das Verfahren der Bodenwahrheit.
- **Originalantworten der Wetter-APIs.** Aus `out/paired_slots.csv` sind die Einzelwerte von OpenWeatherMap, WeatherAPI.com und Tomorrow.io entfernt, deren Nutzungsbedingungen eine Weitergabe nicht zulassen. Enthalten bleiben Kamera, Satellit sowie Bright Sky (DWD) und Open-Meteo. Die aggregierten Kennzahlen aller sechs Quellen sind vollständig vorhanden.

## Ausführen

```bash
pip install -r requirements.txt
python d5_quellenvergleich.py
```

Die Skripte schreiben ihre Ausgaben nach `out/` und überschreiben dabei die mitgelieferten Dateien. Entwickelt und ausgeführt unter Python 3.13.

## Verwendete Hilfsmittel

Die Skripte wurden mit Unterstützung eines KI-Codeassistenten entwickelt. Methodenwahl, Metriken und Parameter stammen vom Autor; der Code wurde geprüft und die Ergebnisse gegen die eingefrorene Kampagnendatenbank verifiziert.

## Lizenz

Code unter MIT (siehe `LICENSE`), Ergebnisdateien in `out/` unter CC BY 4.0. Datenquellen und Attribution: siehe `NOTICE.md`.
