# Datenquellen und Attribution

## Satellitendaten

Die Auswertung beruht auf der **MTG FCI Level-2 Cloud Mask** (Collection `EO:EUM:DAT:0678`, EUMETSAT Data Store). Das Produkt ist nach der EUMETSAT Data Policy als Core Data unter CC BY 4.0 lizenziert.

> Contains modified EUMETSAT Meteosat data 2026.

Diese Attributionszeile gilt für alle aus der Wolkenmaske abgeleiteten Werte und Abbildungen in `out/`.

Level-1C-Daten des MTG wurden nicht verwendet.

## Bodendaten

Die Werte der Quelle `weather_brightsky` stammen über die Bright-Sky-API aus Stationsbeobachtungen des Deutschen Wetterdienstes. Datenbasis: Deutscher Wetterdienst.

Die Werte der Quelle `weather_openmeteo` stammen über Open-Meteo aus dem numerischen Wettermodell DWD ICON-D2.

## Nicht enthaltene Daten

Dieses Repository enthält **keine** EUMETSAT-Originaldaten. Roh-Downloads (`.nc`, Szenenarchive) gehören grundsätzlich nicht in dieses Repository; die Regel ist in `.gitignore` festgehalten.

Einzelmesswerte der Anbieter OpenWeatherMap, WeatherAPI.com und Tomorrow.io sind aus den Zeitreihen entfernt, da deren Nutzungsbedingungen eine Weitergabe nicht zulassen. Aggregierte Kennzahlen dieser Quellen (Fehlermaße, Korrelationen, Konfusionsmatrizen) sind abgeleitete Statistik und enthalten.

## Lizenz der Ergebnisdateien

Die Dateien in `out/` stehen unter CC BY 4.0, passend zur Lizenz der zugrunde liegenden Wolkenmaske.
