"""ASTRA_Glia · Cortex-Gateway: Thalamus + Hippocampus + RAG + Tool-Loop.
F1  interne Kanaele (Telemetrie) werden NICHT generiert -> nur Embedding + Ablage
F2  ohne Daten: ehrliche Antwort statt Erfindung
F3  ungeerdete Antworten bleiben fluechtig (Salienz 0) -> kein Wissens-Muell
B   Tools werden ausgefuehrt, Ergebnis -> Antwort-Synthese -> Wissens-Episode
"""
from __future__ import annotations

import gc
import json
import logging
import queue
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embedder = None

def _get_embedder():
    """Lazy-Load des Embedding-Modells (einmalig pro Prozess)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedder] Lade {EMBEDDING_MODEL} ...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        dim = _embedder.get_embedding_dimension()
        print(f"[Embedder] Bereit (dim={dim}).")
    return _embedder

log = logging.getLogger("astra.gateway")

INTERNAL_CHANNELS = ("konsolidierung",)
RAG_EXCLUDE_CHANNELS = ("konsolidierung", "system")
CACHE_TOOLS = ("wetter.api", "weather.api")  # Nur Wetter ist kurzlebig
# search.api und wikipedia.api sind Wissen, kein Cache
RAG_SHORTCIRCUIT = 0.50   # ab diesem Top-Score: direkt aus frischer Erinnerung antworten
RAG_SHORTCIRCUIT_TOOL = 0.90  # Tool-Episoden (Wikipedia/Search): nur bei nahezu identischer Frage
SHORTCIRCUIT_TOOLS = ("wikipedia.api", "search.api")
SYNTH_TEMP = 0.3          # Antwort-Synthese aus Daten: kaum Streuung
MULTI_QUERY_TOOLS = ("wetter.api", "weather.api")  # ein Aufruf pro Ort
KNOWN_TOOLS = ("wetter.api", "weather.api", "search.api", "wikipedia.api")
_TOOLS_STR = ", ".join(KNOWN_TOOLS)

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(KNOWN_TOOLS) + ["direct"]},
        "query": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["direct"]},
        "answer": {"type": "string"},
    },
    "required": ["action", "answer"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Du bist der praefrontale Kortex von ASTRA_Glia. "
    "Antworte NUR mit einem JSON-Objekt, ohne Zusatztext. "
    'Tool-Aufruf: {"action": "<tool.name>", "query": "<argumente>"} '
    'Direkte Antwort: {"action": "direct", "answer": "<antwort in einem Satz>"} '
    f"Verfuegbare Tools: {_TOOLS_STR}. "
    "Tool-Argumente (query): bei wetter.api NUR der Ortsname (z.B. Berlin), "
    "bei mehreren Orten alle Ortsnamen durch Semikolon getrennt (z.B. Marl; Essen), "
    "bei wikipedia.api das Stichwort, bei search.api kurze Suchbegriffe. "
    "Braucht die Frage externe Fakten (Wetter, Suche, Nachschlagen), waehle ein Tool. "
    "Steht die Antwort bereits in den Erinnerungen, antworte direct aus den Erinnerungen. "
    "Hast du weder Daten noch Erinnerungen, antworte exakt: "
    '{"action": "direct", "answer": "Keine Daten vorhanden."} '
    "Erfinde NIEMALS Fakten. Kopiere Erinnerungen niemals woertlich, "
    "fasse sie in eigenen Worten zusammen. /no_think "
)

TOOL_ANSWER_PROMPT = (
    "Du bist der praefrontale Kortex von ASTRA_Glia. "
    "Beantworte die Frage in ein bis zwei Saetzen, ausschliesslich mit den gelieferten Tool-Daten. "
    "Formuliere einen grammatikalisch korrekten deutschen Satz, der alle Werte enthaelt: "
    "Wetterbedingung (dekliniert, z.B. 'Bei leichtem Regen'), "
    "Temperaturbereich, Regenwahrscheinlichkeit in Prozent, Wind. "
    "Verwende deutsche Zahlenformate (Komma als Dezimaltrennzeichen) und ein Leerzeichen vor Einheiten. "
    "Erfinde nichts dazu. Antworte NUR als JSON: "
    '{"action": "direct", "answer": "<antwort>"} '
)

def embed_text(llm, text):
    """MiniLM-Embedding (384 dim, multilingual, offline)."""
    return _get_embedder().encode(text, convert_to_numpy=True, normalize_embeddings=True)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _is_echo(answer, memories):
    """True, wenn eine ganze Episode kopiert statt genutzt wurde."""
    a = _norm_text(answer)
    if len(a) < 12:
        return False
    for m in memories:
        b = _norm_text(m.get("text", ""))
        if not b:
            continue
        if b in a:
            return True
        if SequenceMatcher(None, a, b).ratio() >= 0.9:
            return True
    return False

def _compact_json(s, limit=800):
    try:
        return json.dumps(json.loads(s), ensure_ascii=False, separators=(",", ":"))[:limit]
    except (json.JSONDecodeError, TypeError):
        return " ".join((s or "").split())[:limit]

def _de_num(x, nd=1):
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except (TypeError, ValueError):
        return None

def _de_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").strftime("%d.%m.%Y")
    except ValueError:
        return str(s or "")

def _format_weather(result):
    """Wetter-JSON -> fester deutscher Satz (ohne LLM). None bei unbekanntem Format."""
    try:
        d = json.loads(result)
        t = d["temperature"]
        tmin, tmax = _de_num(t["min"]), _de_num(t["max"])
        loc, cond = d.get("location"), d.get("condition")
        if None in (tmin, tmax) or not loc or not cond:
            return None
        unit = t.get("unit", "°C")
        parts = [str(cond), f"{tmin} bis {tmax} {unit}"]
        prob = d.get("precipitation_probability")
        if prob is not None:
            parts.append(f"Regenwahrscheinlichkeit {int(prob)} %")
        wind = d.get("wind") or {}
        wspeed = _de_num(wind.get("speed"))
        if wspeed:
            parts.append(f"Wind {wspeed} {wind.get('unit', 'km/h')}")
        date = _de_date(d.get("date"))
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    head = f"{loc} ({date})" if date else str(loc)
    return f"{head}: " + ", ".join(parts) + "."

def _split_queries(action, query):
    """Mehrere Orte (\"Marl; Essen\" / \"Marl und Essen\") -> ein Tool-Aufruf je Ort."""
    if action in MULTI_QUERY_TOOLS:
        parts = [p.strip() for p in re.split(r"\s*;\s*|\s+und\s+", query or "") if p.strip()]
        return parts[:4] or [query]
    return [query]

def action_to_json(act):
    if act["action"] == "direct":
        return json.dumps({"action": "direct", "answer": act["answer"]}, ensure_ascii=False)
    return json.dumps({"action": act["action"], "query": act["query"]}, ensure_ascii=False)

def parse_action(raw: str) -> dict:
    """Roher LLM-Output -> validiertes Dict. Wirft NIE."""
    out = {"action": "direct", "query": "", "answer": "", "degraded": False,
           "error": "", "raw": raw or ""}
    if not raw or not raw.strip():
        out.update(degraded=True, error="leere Antwort")
        return out
    text = _THINK_RE.sub("", raw).strip()
    if "<think>" in text and "</think>" not in text:
        out.update(degraded=True, error="Thinking-Block abgeschnitten")
        return out
    m = _OBJ_RE.search(text)
    if not m:
        out.update(degraded=True, error="kein JSON-Objekt gefunden")
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        out.update(degraded=True, error=f"JSON ungueltig: {e}")
        return out
    if not isinstance(obj, dict):
        out.update(degraded=True, error="JSON ist kein Objekt")
        return out
    obj = {str(k).strip(): v for k, v in obj.items()}
    action = str(obj.get("action", "")).strip()
    query = str(obj.get("query", "")).strip()
    answer = str(obj.get("answer", "")).strip()
    # Netz: Bolt-Variante {"action":"tool","query":"<tool>","argumente":"..."}
    if action == "tool" and query in KNOWN_TOOLS:
        action, query = query, str(obj.get("argumente") or obj.get("arguments")
                                     or answer).strip()
    if action in KNOWN_TOOLS:
        out.update(action=action, query=query or answer)
    elif action == "direct":
        out.update(action="direct", answer=answer or query)
    elif action == "error":
        out.update(action="direct", answer=answer or query, degraded=True,
                   error=str(obj.get("error", "Modell meldet Fehler")))
    else:
        out.update(action="direct", answer=answer or query or (raw or "")[:200],
                   degraded=True, error=f"unbekanntes Tool '{action}'")
    return out

class CortexState(Enum):
    SLEEPING = "sleeping"
    WAKING = "waking"
    AWAKE = "awake"

class CortexGateway:
    def __init__(self, model_path, on_action=None, n_gpu_layers=-1, n_ctx=2048,
                 buffer_size=512, idle_timeout=30.0,
                 log_path="logs/cortex_gateway.jsonl",
                 rag_enabled=True, rag_k=3, rag_min_score=0.4, rag_max_chars=1400,
                 tool_executor=None, hippocampus=None):
        if hippocampus is None:
            from core.hippocampus import Hippocampus
            hippocampus = Hippocampus()
        self.hippocampus = hippocampus
        self.model_path = str(model_path)
        self.on_action = on_action
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.buffer_size = buffer_size
        self.idle_timeout = idle_timeout
        self.log_path = Path(str(log_path).strip())
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rag_enabled = bool(rag_enabled)
        self.rag_k = int(rag_k)
        self.rag_min_score = float(rag_min_score)
        self.rag_max_chars = int(rag_max_chars)
        self.tool_executor = tool_executor
        self._state = CortexState.SLEEPING
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._llm = None
        self._llm_busy = 0
        self._buffer = deque()
        self._inbox = queue.Queue()
        self._worker = None
        self._stop = threading.Event()
        self._last_activity = 0.0
        self._seq = 0
        self.stats = {"wakes": 0, "buffered": 0, "dropped": 0, "processed": 0,
                      "degraded": 0, "last_wake_latency_s": None,
                      "rag_used": 0, "rag_memories": 0, "rag_empty": 0, "rag_echo": 0,
                      "rag_shortcircuit": 0,
                      "tool_calls": 0, "tool_errors": 0,
                      "grounded": 0, "ungrounded": 0, "internal_ack": 0}

    # ------------------------------------------------------------- Modell
    def _make_llm(self):
        from llama_cpp import Llama
        kw = dict(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers,
                  n_ctx=self.n_ctx, embedding=False, verbose=False)
        try:
            return Llama(flash_attn=True, **kw)
        except Exception as e:
            print(f"[Gateway] Flash-Attention nicht verfuegbar ({e}) -> ohne FA")
            return Llama(**kw)

    @contextmanager
    def _use_llm(self):
        while True:
            with self._lock:
                if self._llm is not None:
                    self._llm_busy += 1
                    llm = self._llm
                    break
            with self._load_lock:
                with self._lock:
                    loaded = self._llm is not None
                if not loaded:
                    new = self._make_llm()
                    with self._lock:
                        self._llm = new
        try:
            with self._infer_lock:
                yield llm
        finally:
            with self._lock:
                self._llm_busy -= 1

    # ------------------------------------------------------------- API
    def submit(self, text, salience=0.5, **meta):
        if self._stop.is_set():
            return -1
        with self._lock:
            ev = {"seq": self._seq, "text": text, "salience": float(salience),
                  "t": time.time(), "meta": meta}
            self._seq += 1
            if self._state is CortexState.AWAKE:
                self._inbox.put(ev)
            else:
                self._push_buffer(ev)
                if self._state is CortexState.SLEEPING:
                    self._state = CortexState.WAKING
                    self._worker = threading.Thread(target=self._run, daemon=True)
                    self._worker.start()
        return ev["seq"]

    @property
    def state(self):
        with self._lock:
            return self._state

    def recall(self, query, k=5, where=None):
        with self._use_llm() as llm:
            vec = embed_text(llm, query)
            result = self.hippocampus.recall(vec, k, where=where)
        with self._lock:
            idle = self._state is CortexState.SLEEPING
        if idle:
            self._try_sleep()
        return result

    def shutdown(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=15)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._llm_busy == 0:
                    break
            time.sleep(0.05)
        self._try_sleep(force=True)

    # ------------------------------------------------------------- Queue
    def _push_buffer(self, ev):
        if len(self._buffer) >= self.buffer_size:
            i = min(range(len(self._buffer)), key=lambda k: self._buffer[k]["salience"])
            self.stats["dropped"] += 1
            if self._buffer[i]["salience"] < ev["salience"]:
                del self._buffer[i]
                self._buffer.append(ev)
                self.stats["buffered"] += 1
        else:
            self._buffer.append(ev)
            self.stats["buffered"] += 1

    @staticmethod
    def _score(ev):
        return ev["salience"] + 0.05 * min(time.time() - ev["t"], 10.0)

    def _pop_next(self):
        with self._lock:
            if self._buffer:
                i = max(range(len(self._buffer)), key=lambda k: self._score(self._buffer[k]))
                ev = self._buffer[i]
                del self._buffer[i]
                return ev
            try:
                return self._inbox.get_nowait()
            except queue.Empty:
                return None

    def _run(self):
        t0 = time.time()
        print("[Gateway] Wecke Kortex - SNN spikt ungestoert weiter...")
        try:
            with self._use_llm():
                pass
        except Exception as e:
            with self._lock:
                self._state = CortexState.SLEEPING
            print(f"[Gateway] Laden fehlgeschlagen: {e}")
            self._log({"event": "cortex_wake_failed", "error": str(e)})
            return
        latency = time.time() - t0
        with self._lock:
            self._state = CortexState.AWAKE
            pending = len(self._buffer) + self._inbox.qsize()
            self.stats["wakes"] += 1
            self.stats["last_wake_latency_s"] = round(latency, 2)
        print(f"[Gateway] Online nach {latency:.2f}s - Replay: {pending} Events")
        self._log({"event": "cortex_wake", "latency_s": round(latency, 2), "replay": pending})
        self._last_activity = time.time()
        while not self._stop.is_set():
            ev = self._pop_next()
            if ev is None:
                if time.time() - self._last_activity >= self.idle_timeout:
                    if self._try_sleep():
                        return
                time.sleep(0.05)
                continue
            try:
                self._process(ev)
            except Exception as e:
                log.exception("Event #%s fehlgeschlagen", ev.get("seq"))
                print(f"[Gateway] Event #{ev.get('seq')} fehlgeschlagen: {e}")
                self._log({"event": "cortex_process_failed", "error": str(e)})
            self._last_activity = time.time()
        self._try_sleep(force=True)

    # ------------------------------------------------------------- RAG
    def _retrieve(self, vec, ev):
        where = {"$or": [{"kind": {"$eq": "knowledge"}}, {"tool": {"$eq": "ingest"}}]}
        try:
            hits = self.hippocampus.recall(vec, k=self.rag_k * 4, where=where)
        except Exception as e:
            print(f"[RAG] Recall fehlgeschlagen: {e}")
            return []
        out, seen, done = [], {_norm_text(ev["text"])}, set()
        for m in hits:
            if m.get("score", 0.0) < self.rag_min_score:
                continue
            raw = (m.get("text") or "").strip()
            t = _norm_text(raw)
            if not t:
                continue
            tool = m.get("tool") or ""
            if not tool:
                # Episoden OHNE Tool: alte Nutzerfragen sind keine Fakten.
                # (Tool-Episoden tragen die Antwort in 'answer' und sind
                # absichtlich "Frage + Kernaussage" -> nie so filtern.)
                if raw.endswith("?") and len(raw) <= 160:
                    continue
                if t in seen:
                    continue
            if (t, tool) in done:
                continue
            done.add((t, tool))
            out.append(m)
            if len(out) >= self.rag_k:
                break
        return out

    @staticmethod
    def _answer_text(raw):
        s = (raw or "").strip()
        if not s:
            return ""
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return str(obj.get("answer") or obj.get("query") or "").strip()
        except (json.JSONDecodeError, TypeError):
            pass
        return s

    def _build_context(self, memories):
        lines, used = [], 0
        for m in memories:
            text = " ".join((m.get("text") or "").split())
            ans = " ".join(self._answer_text(m.get("answer", "")).split())
            line = f"- {text[:400]}"
            if ans and _norm_text(ans) != _norm_text(text):
                line += f" | Kernaussage: {ans[:150]}"
            if used + len(line) > self.rag_max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _build_messages(self, ev, memories):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memories:
            ctx = self._build_context(memories)
            if ctx:
                msgs.append({"role": "system",
                             "content": "Hintergrundwissen aus frueheren Episoden "
                                        "(nur Fakten, NICHT kopieren, eigene Worte):\n" + ctx})
        msgs.append({"role": "user", "content": ev["text"]})
        return msgs

    # ------------------------------------------- Memory-Short-Circuit (H2)
    def _shortcircuit_ok(self, memories):
        if not memories or RAG_SHORTCIRCUIT <= 0:
            return False
        top = memories[0]
        score = top.get("score", 0.0)
        tool = top.get("tool", "")
        kind = top.get("kind", "")
        # Umgehen darf die LLM-Entscheidung nur belastbares Wissen:
        # ingestiertes Faktenwissen, oder Wikipedia/Search-Episoden bei fast
        # identischer Frage. Alte Chat-Episoden (tool="") zaehlen nie.
        if kind != "knowledge":
            return False
        if tool == "ingest":
            if score < RAG_SHORTCIRCUIT:
                return False
        elif tool in SHORTCIRCUIT_TOOLS:
            if score < RAG_SHORTCIRCUIT_TOOL:
                return False
        else:
            return False
        exp = float(top.get("expires_at", 0.0) or 0.0)
        if exp and exp <= time.time():
            return False
        return bool((top.get("text") or "").strip() or (top.get("answer") or "").strip())

    def _answer_from_memory(self, ev, memories):
        ctx = self._build_context(memories)
        msgs = [{"role": "system",
                 "content": "Du bist der praefrontale Kortex von ASTRA_Glia. "
                            "Beantworte die Frage in EINEM kurzen Satz, ausschliesslich "
                            "mit dem Hintergrundwissen. Erfinde nichts. "
                            "Antworte NUR als JSON: "
                            '{"action": "direct", "answer": "<antwort>"} /no_think'},
                {"role": "user",
                 "content": f"Frage: {ev['text']}\nHintergrundwissen:\n{ctx}"}]
        raw = ""
        try:
            with self._use_llm() as llm:
                out = llm.create_chat_completion(messages=msgs, max_tokens=320, temperature=SYNTH_TEMP, top_p=0.95,
                                                 response_format={"type": "json_object",
                                                                    "schema": ANSWER_SCHEMA}) 
            raw = out["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"[RAG] Antwort aus Erinnerung fehlgeschlagen: {e}")
        a = parse_action(raw)
        ans = (a["answer"] or a["query"] or "").strip()
        if not ans or ans.startswith("{"):
            top = memories[0]
            ans = (self._answer_text(top.get("answer", ""))
                   or " ".join((top.get("text") or "").split()))[:220]
        return ans

    # ------------------------------------------------------------- Tools (B)
    def _answer_from_tool(self, ev, action, result):
        if action in MULTI_QUERY_TOOLS:
            sentence = _format_weather(result)
            if sentence:
                return sentence
        compact = _compact_json(result)
        
        # Tool-spezifische Prompt-Vorlage
        if action in ("wetter.api", "weather.api"):
            system_prompt = (
                "Du bist der praefrontale Kortex von ASTRA_Glia. "
                "Beantworte die Frage in EINEM grammatikalisch korrekten deutschen Satz. "
                "Verwende ausschliesslich die gelieferten Tool-Daten. "
                "Format: 'Bei [Bedingung] liegt die Temperatur zwischen [min] und [max] °C, "
                "die Regenwahrscheinlichkeit beträgt [X] % und der Wind weht mit [Y] km/h.' "
                "Achte auf Leerzeichen vor Einheiten (% , °C, km/h) und vor 'und'. "
                "Erfinde nichts dazu. Antworte NUR als JSON: "
                '{"action": "direct", "answer": "<antwort>"}'
            )
        elif action == "wikipedia.api":
            system_prompt = (
                "Du bist der praefrontale Kortex von ASTRA_Glia. "
                "Beantworte die Frage in ein bis zwei Saetzen, ausschliesslich mit den gelieferten Wikipedia-Daten. "
                "Fasse die wichtigsten Fakten zusammen, erfinde nichts dazu. "
                "Antworte NUR als JSON: "
                '{"action": "direct", "answer": "<antwort>"}'
            )
        elif action == "search.api":
            system_prompt = (
                "Du bist der praefrontale Kortex von ASTRA_Glia. "
                "Beantworte die Frage in ein bis zwei Saetzen, ausschliesslich mit den gelieferten Suchergebnissen. "
                "Fasse die relevantesten Informationen zusammen, erfinde nichts dazu. "
                "Antworte NUR als JSON: "
                '{"action": "direct", "answer": "<antwort>"}'
            )
        else:
            system_prompt = TOOL_ANSWER_PROMPT  # Fallback
        
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": f"Frage: {ev['text']}\nTool-Daten ({action}): {compact}"}]
        try:
            with self._use_llm() as llm:
                out = llm.create_chat_completion(messages=msgs, max_tokens=320,
                                                 temperature=SYNTH_TEMP, top_p=0.95,
                                                 response_format={"type": "json_object",
                                                                    "schema": ANSWER_SCHEMA})
            raw = out["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"[Tool] Antwort-Synthese fehlgeschlagen: {e}")
            return f"{action}: {compact[:200]}"
        a = parse_action(raw)
        ans = (a["answer"] or a["query"] or "").strip()
        if not ans or ans.startswith("{"):
            ans = f"{action}: {compact[:200]}"
        return ans

    # ------------------------------------------------------------- Verarbeitung
    def _process(self, ev):
        channel = ev["meta"].get("channel", "")
        internal = channel in INTERNAL_CHANNELS
        rag_active = self.rag_enabled and not internal
        vec, memories, act = None, [], None
        tool_used, tool_ok, tool_name, mem_used = False, False, None, False
        if internal:
            # F1: Telemetrie wird nicht generiert – nur einbetten und ablegen
            try:
                with self._use_llm() as llm:
                    vec = embed_text(llm, ev["text"])
            except Exception as e:
                print(f"[Gateway] Embedding fehlgeschlagen: {e}")
            act = {"action": "direct", "query": "",
                   "answer": "Konsolidierung bestaetigt.", "degraded": False,
                   "error": "", "raw": ""}
            self.stats["internal_ack"] += 1
        else:
            try:
                with self._use_llm() as llm:
                    try:
                        vec = embed_text(llm, ev["text"])
                    except Exception as e:
                        print(f"[Gateway] Embedding fehlgeschlagen: {e}")
                    if rag_active and vec is not None:
                        memories = self._retrieve(vec, ev)
                    if rag_active and self._shortcircuit_ok(memories):
                        mem_used = True     # Antwort kommt unten aus der Erinnerung
                    else:
                        out = llm.create_chat_completion(
                            messages=self._build_messages(ev, memories),
                            max_tokens=512, temperature=1.0, top_p=0.95,
                            response_format={"type": "json_object",
                                         "schema": ACTION_SCHEMA})
                        raw = out["choices"][0]["message"]["content"] or ""
            except Exception as e:
                print(f"[Gateway] Inferenz-Fehler: {type(e).__name__}: {e}")
                raw = json.dumps({"action": "error", "query": str(e)})
            if mem_used:
                answer = self._answer_from_memory(ev, memories)
                act = {"action": "direct", "query": "", "answer": answer,
                       "degraded": False, "error": "", "raw": ""}
                self.stats["rag_shortcircuit"] += 1
                top = memories[0]
                print(f"[RAG] Short-Circuit: Antwort aus Erinnerung "
                      f"(score {top.get('score', 0.0):.2f})")
                self._log({"event": "rag_shortcircuit", "seq": ev["seq"],
                           "score": round(top.get("score", 0.0), 3),
                           "tool": top.get("tool", "")})
            else:
                act = parse_action(raw)
            if memories:
                self.stats["rag_used"] += 1
                self.stats["rag_memories"] += len(memories)
                top = max(m.get("score", 0.0) for m in memories)
                print(f"[RAG] {len(memories)} Wissens-Episode(n) injiziert (top {top:.2f})")
                self._log({"event": "cortex_rag", "seq": ev["seq"], "n": len(memories),
                           "top_score": round(top, 3)})
            elif rag_active and vec is not None:
                self.stats["rag_empty"] += 1
                print("[RAG] keine passende Wissens-Episode")
                self._log({"event": "cortex_rag_empty", "seq": ev["seq"]})
            # B: Tool ausfuehren und Antwort aus den Daten synthetisieren
            if act["action"] in KNOWN_TOOLS:
                tool_used = True
                tool_name = act["action"]
                self.stats["tool_calls"] += 1
                if self.tool_executor is None:
                    self.stats["tool_errors"] += 1
                    act.update(action="direct", degraded=True,
                               error="kein Tool-Executor verdrahtet",
                               answer="Keine Daten vorhanden (Tools nicht verdrahtet).")
                    print("[Tool] kein Executor verdrahtet (main.py: tool_executor=...)")
                else:
                    queries = _split_queries(act["action"], act["query"])
                    results = []
                    for q in queries:
                        try:
                            r = self.tool_executor(act["action"], q)
                        except Exception as e:
                            r = None
                            print(f"[Tool] Ausfuehrung fehlgeschlagen ({q}): {e}")
                        if r:
                            results.append(r)
                    if results:
                        tool_ok = True
                        answer = " ".join(self._answer_from_tool(ev, act["action"], r)
                                          for r in results)
                        if len(results) < len(queries):
                            answer += " (Fuer mindestens einen Ort gab es keine Daten.)"
                        self._log({"event": "tool_result", "seq": ev["seq"],
                                   "tool": act["action"], "query": "; ".join(queries),
                                   "chars": sum(len(r) for r in results)})
                        act.update(action="direct", query="", answer=answer)
                    else:
                        self.stats["tool_errors"] += 1
                        act.update(action="direct", degraded=True,
                                   error="Tool lieferte kein Ergebnis",
                                   answer="Keine Daten vorhanden (Tool fehlgeschlagen).")
        # Echo-Blocker nur bei RAG-basierten Antworten
        echo = False
        if memories and not tool_used and not mem_used and _is_echo(act["answer"], memories):
            echo = True
            self.stats["rag_echo"] += 1
            act.update(degraded=True, error="Antwort war Kopie einer Erinnerung",
                       answer=f"{len(memories)} passende Erinnerung(en) gefunden – "
                              f"Antwort verworfen, da woertliche Kopie.")
            print("[RAG] Echo blockiert")
            self._log({"event": "rag_echo_blocked", "seq": ev["seq"]})
        grounded = internal or tool_ok or mem_used or (bool(memories) and not echo)
        if not internal:
            self.stats["grounded" if grounded else "ungrounded"] += 1
        # F3: nur geerdetes Wissen darf langfristig werden
        if internal:
            store_sal = ev["salience"]
        elif tool_ok:
            store_sal = max(ev["salience"], 0.7)
        elif grounded:
            store_sal = max(ev["salience"], 0.6)
        else:
            store_sal = 0.0
        clean = action_to_json(act)
        self.stats["processed"] += 1
        if act["degraded"]:
            self.stats["degraded"] += 1
            print(f"[Router] degraded: {act['error']}")
            ev["meta"]["degraded"] = True
        self._log({"event": "cortex_answer", "seq": ev["seq"], "action": act["action"],
                   "grounded": grounded, "store_salience": store_sal,
                   "degraded": act["degraded"], "error": act["error"],
                   "raw": act["raw"][:200]})
        try:
            if echo:
                print("[Hippocampus] Echo nicht gespeichert")
            else:
                if vec is None:
                    with self._use_llm() as llm:
                        vec = embed_text(llm, ev["text"])
                # Wetter = cache (kurzlebig), Wikipedia/Search = knowledge (langfristig)
                if internal:
                    ep_kind = "telemetry"
                elif tool_name in ("wetter.api", "weather.api"):
                    ep_kind = "cache"
                else:
                    ep_kind = "knowledge"  # wikipedia.api, search.api → Wissen
                self.hippocampus.encode(
                    ev["text"], store_sal, channel, vec, clean,
                    tool=(tool_name if tool_ok else None),
                    kind=ep_kind)
                if internal:
                    self.hippocampus.consolidate()
                elif tool_ok:
                    moved = self.hippocampus.consolidate()
                    if moved:
                        print(f"[Wissen] Tool-Episode gesichert ({moved} Item(s))")
                elif not grounded:
                    print("[Wissen] ungeerdet -> bleibt fluechtig (nicht konsolidiert)")
        except Exception as e:
            print(f"[Hippocampus] Encoding fehlgeschlagen: {e}")
        if self.on_action is not None and not internal:
            self.on_action(ev, clean)

    # ------------------------------------------------------------- Schlaf
    def _try_sleep(self, force=False):
        with self._lock:
            if not force and (self._buffer or not self._inbox.empty()):
                return False
            if not force and self._llm_busy > 0:
                return False
            if self._state is CortexState.SLEEPING and self._llm is None:
                return False
            llm, self._llm = self._llm, None
            self._state = CortexState.SLEEPING
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
            del llm
            gc.collect()
        print(f"[Gateway] VRAM freigegeben. Stats: {self.stats}")
        self._log({"event": "cortex_sleep", **self.stats})
        return True

    def _log(self, payload):
        payload["timestamp"] = datetime.now().isoformat()
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("Log schreiben fehlgeschlagen: %s", e)