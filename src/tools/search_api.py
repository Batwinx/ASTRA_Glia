"""ASTRA_Glia · Suche (live, keylos).

Reihenfolge:
  1. DuckDuckGo Web-Suche (HTML-Endpoint) -> echte Treffer inkl. Neuigkeiten
  2. DuckDuckGo Instant Answer            -> nur Lexikon-artige Treffer
  3. Wikipedia                            -> letzter Ausweg
"""
import html as _html
import json
import re
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from src.tools._http import get_json

_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
_TIMEOUT_S = 8
_MAX_RESULTS = 4
_SNIPPET_CHARS = 220

_DIV_RE = re.compile(r'<div[^>]+class="([^"]*)"')
_A_RE = re.compile(r'<a\b([^>]*class="result__a"[^>]*)>(.*?)</a>', re.S)
_SNIP_RE = re.compile(
    r'<(?:a|div)\b[^>]*class="result__snippet"[^>]*>(.*?)</(?:a|div)>', re.S)
_HREF_RE = re.compile(r'href="([^"]+)"')
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s):
    return " ".join(_html.unescape(_TAG_RE.sub("", s or "")).split())


def _real_url(href):
    """DuckDuckGo verpackt Ziele als //duckduckgo.com/l/?uddg=<url>."""
    href = _html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    q = parse_qs(urlparse(href).query)
    return q["uddg"][0] if "uddg" in q else href


def _parse_results(page, limit=_MAX_RESULTS):
    """HTML der DDG-Trefferseite -> [{title, snippet, domain}]. Anzeigen werden übersprungen."""
    starts = [m.start() for m in _DIV_RE.finditer(page)
              if "result" in m.group(1).split()
              and "result--ad" not in m.group(1).split()]
    out, seen = [], set()
    for i, st in enumerate(starts):
        block = page[st:(starts[i + 1] if i + 1 < len(starts) else len(page))]
        a = _A_RE.search(block)
        if not a:
            continue
        href = _HREF_RE.search(a.group(1))
        if not href:
            continue
        url = _real_url(href.group(1))
        if "duckduckgo.com/y.js" in url:      # Werbung
            continue
        title = _clean(a.group(2))
        sn = _SNIP_RE.search(block)
        snippet = _clean(sn.group(1))[:_SNIPPET_CHARS] if sn else ""
        domain = urlparse(url).netloc.removeprefix("www.")
        if not title or (title, domain) in seen:
            continue
        seen.add((title, domain))
        out.append({"title": title, "snippet": snippet, "domain": domain})
        if len(out) >= limit:
            break
    return out


def _web_results(q):
    req = Request(
        "https://html.duckduckgo.com/html/?q=" + quote_plus(q) + "&kl=de-de",
        headers={"User-Agent": _UA, "Accept-Language": "de-DE,de;q=0.9"})
    with urlopen(req, timeout=_TIMEOUT_S) as r:
        page = r.read().decode("utf-8", "replace")
    return _parse_results(page)


def search_api(query: str) -> str:
    q = " ".join((query or "").split())
    print(f"   🔍 Suche (DuckDuckGo) für: {q}")

    # 1) Web-Suche
    try:
        results = _web_results(q)
        if results:
            return json.dumps({"query": q, "results": results,
                               "source": "DuckDuckGo Web (live)"}, ensure_ascii=False)
        print("   ↪️ Web-Treffer leer – versuche Instant Answer")
    except Exception as e:
        print(f"   ⚠️ Websuche fehlgeschlagen: {e} – versuche Instant Answer")

    # 2) Instant Answer
    try:
        d = get_json("https://api.duckduckgo.com/",
                     {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1})
        abstract = (d.get("AbstractText") or "").strip()
        related = [{"text": t.get("Text", ""), "url": t.get("FirstURL", "")}
                   for t in (d.get("RelatedTopics") or [])
                   if isinstance(t, dict) and t.get("Text")][:5]
        if abstract or related:
            return json.dumps({
                "query": q, "heading": d.get("Heading", ""),
                "abstract": abstract, "abstract_url": d.get("AbstractURL", ""),
                "related": related,
                "source": "DuckDuckGo Instant Answer (live)"}, ensure_ascii=False)
        print("   ↪️ IA leer – Fallback Wikipedia")
    except Exception as e:
        print(f"   ⚠️ Instant Answer fehlgeschlagen: {e} – Fallback Wikipedia")

    # 3) Wikipedia
    from src.tools.wikipedia_api import wikipedia_api
    return wikipedia_api(q)