"""HTTP-Helfer für echte Tool-APIs – nur Standardbibliothek, kein requests nötig."""
import json
import urllib.parse
import urllib.request

UA = "ASTRA_Glia/1.0 (Schulprojekt; lokaler Agent)"
TIMEOUT = 10.0


def get_json(url, params=None, timeout=TIMEOUT):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def quote(text):
    return urllib.parse.quote(str(text), safe="")