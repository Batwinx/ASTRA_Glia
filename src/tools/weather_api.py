"""ASTRA_Glia · Wetter: Open-Meteo (live, ohne API-Key).
Offline-Fallback auf Demo-Daten, damit die Simulation in der Schule nicht abbricht.
Für Ernstbetrieb: ALLOW_DEMO_FALLBACK = False setzen.
"""
import json
from datetime import datetime, timedelta

from src.tools._http import get_json

ALLOW_DEMO_FALLBACK = True

WMO = {0: "klar", 1: "überwiegend klar", 2: "teilweise bewölkt", 3: "bedeckt",
       45: "Nebel", 48: "Reifnebel", 51: "leichter Sprühregen", 53: "Sprühregen",
       55: "starker Sprühregen", 61: "leichter Regen", 63: "Regen",
       65: "starker Regen", 71: "leichter Schneefall", 73: "Schneefall",
       75: "starker Schneefall", 80: "Regenschauer", 81: "kräftige Regenschauer",
       82: "heftige Regenschauer", 95: "Gewitter", 96: "Gewitter mit Hagel",
       99: "Gewitter mit starkem Hagel"}


def _extract_location(query):
    q = " ".join((query or "").split()).strip(" ?.")
    low = q.lower()
    if " in " in low:
        return q[low.rindex(" in ") + 4:].strip(" ?.")
    for junk in ("wie wird das", "wetterbericht", "wetter", "morgen", "heute",
                 "vorhersage", "prognose", "für", "fuer"):
        low = low.replace(junk, " ")
    return " ".join(low.split()).strip(" ?.").title() or q


def _demo(loc, err):
    return json.dumps({
        "location": loc,
        "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "condition": "teilweise bewölkt",
        "temperature": {"min": 8, "max": 15, "unit": "°C"},
        "precipitation_probability": 20,
        "wind": {"speed": 12, "unit": "km/h"},
        "simulated": True,
        "source": "Demo-Fallback (offline)", "error": err}, ensure_ascii=False)


def weather_api(query: str) -> str:
    loc = _extract_location(query)
    print(f"   🌤️ Wetter (Open-Meteo) für: {loc}")
    try:
        geo = get_json("https://geocoding-api.open-meteo.com/v1/search",
                       {"name": loc, "count": 1, "language": "de", "format": "json"})
        results = geo.get("results") or []
        if not results:
            raise ValueError(f"Ort nicht gefunden: {loc}")
        g = results[0]
        fc = get_json("https://api.open-meteo.com/v1/forecast", {
            "latitude": g["latitude"], "longitude": g["longitude"],
            "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                      "precipitation_probability_max,wind_speed_10m_max,"
                      "wind_direction_10m_dominant"),
            "timezone": "auto", "forecast_days": 2})
        d = fc["daily"]
        i = 1 if len(d["time"]) > 1 else 0     # Index 1 = morgen
        return json.dumps({
            "location": g.get("name", loc), "country": g.get("country", ""),
            "date": d["time"][i],
            "condition": WMO.get(int(d["weather_code"][i]), "unbekannt"),
            "temperature": {"min": d["temperature_2m_min"][i],
                            "max": d["temperature_2m_max"][i], "unit": "°C"},
            "precipitation_probability": d["precipitation_probability_max"][i],
            "wind": {"speed": d["wind_speed_10m_max"][i], "unit": "km/h",
                     "direction_deg": d["wind_direction_10m_dominant"][i]},
            "source": "Open-Meteo (live)"}, ensure_ascii=False)
    except Exception as e:
        print(f"   ⚠️ Wetter live fehlgeschlagen: {e}")
        if not ALLOW_DEMO_FALLBACK:
            return json.dumps({"error": str(e), "source": "Open-Meteo"},
                              ensure_ascii=False)
        return _demo(loc, str(e))