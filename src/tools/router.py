"""ASTRA_Glia · Tool-Router (Mail bewusst nicht registriert)."""
import json
import re
from typing import Any, Dict, Optional

_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)          # greedig: auch verschachtelte Objekte
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ToolRouter:
    def __init__(self, verbose=True):
        self.verbose = bool(verbose)
        self.tools = {}
        self.register_default_tools()

    def register_default_tools(self):
        from src.tools.weather_api import weather_api
        from src.tools.search_api import search_api
        from src.tools.wikipedia_api import wikipedia_api
        self.register_tool("wetter.api", weather_api)
        self.register_tool("weather.api", weather_api)   # Alias, jetzt auch im Gateway bekannt
        self.register_tool("search.api", search_api)
        self.register_tool("wikipedia.api", wikipedia_api)

    def register_tool(self, name: str, func):
        self.tools[name] = func
        if self.verbose:
            print(f"   🔧 Tool registriert: {name}")

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def execute(self, action: str, query: str = "") -> Optional[str]:
        """Fuehrt eine bereits geparste Aktion aus. Gateway nutzt diesen Weg."""
        if action not in self.tools:
            if self.verbose:
                print(f"⚠️ Unbekanntes Tool: {action}")
            return None
        if self.verbose:
            print(f"\n🔧 Rufe Tool auf: {action}")
            print(f"   Query: {query}")
        try:
            result = self.tools[action](query)
        except Exception as e:
            print(f"   ❌ Tool-Fehler: {e}")
            return f"Fehler beim Aufruf von {action}: {e}"
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        if self.verbose:
            print(f"   ✅ Tool-Antwort erhalten ({len(result)} Zeichen)")
        return result

    def parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        cleaned = _THINK_RE.sub("", response or "").strip()
        m = _OBJ_RE.search(cleaned)
        if not m:
            if self.verbose:
                print(f"⚠️ Kein JSON in Antwort gefunden: {cleaned[:100]}")
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON-Parsing fehlgeschlagen: {e}")
            return None
        return obj if isinstance(obj, dict) else None

    def route(self, llm_response: str) -> Optional[str]:
        """Compat-Pfad: Rohtext parsen und ausfuehren."""
        call = self.parse_llm_response(llm_response)
        if not call:
            return None
        return self.execute(str(call.get("action", "")).strip(),
                            str(call.get("query", "")).strip())

    def list_tools(self):
        print("\n📋 Verfügbare Tools:")
        for name in self.tools:
            print(f"   - {name}")