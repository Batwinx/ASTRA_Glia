"""ASTRA_Glia · Wikipedia (live, ohne API-Key): Suche + REST-Summary."""
import json

from src.tools._http import get_json, quote

ALLOW_DEMO_FALLBACK = True


def wikipedia_api(query: str) -> str:
    q = " ".join((query or "").split())
    print(f"   📚 Wikipedia (live) für: {q}")
    try:
        s = get_json("https://de.wikipedia.org/w/api.php",
                     {"action": "query", "list": "search", "srsearch": q,
                      "format": "json", "srlimit": 1})
        hits = (s.get("query") or {}).get("search") or []
        title = hits[0]["title"] if hits else q
        summ = get_json(
            f"https://de.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
        extract = (summ.get("extract") or "").strip()
        if not extract:
            raise ValueError("kein Auszug gefunden")
        return json.dumps({
            "query": q, "title": summ.get("title", title),
            "summary": extract[:1200],
            "url": ((summ.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
            "source": "Wikipedia (live)"}, ensure_ascii=False)
    except Exception as e:
        print(f"   ⚠️ Wikipedia live fehlgeschlagen: {e}")
        if not ALLOW_DEMO_FALLBACK:
            return json.dumps({"error": str(e), "source": "Wikipedia"},
                              ensure_ascii=False)
        return json.dumps({
            "query": q, "title": q,
            "summary": f"Kein Live-Abruf möglich ({e}).",
            "simulated": True, "source": "Demo-Fallback (offline)"},
            ensure_ascii=False)