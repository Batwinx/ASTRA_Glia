"""Ketten-Test: stabiles Wissen -> Short-Circuit | volatile Frage -> Tool.
Aufruf:  python tests/test_tool_rag.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hippocampus import Hippocampus
from core.cortex_gateway import CortexGateway, embed_text
from src.tools.router import ToolRouter
from src.tools.wikipedia_api import wikipedia_api

MODEL = Path("checkpoints") / "llm" / "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"


def wait(gw, n, timeout=180):
    t0 = time.time()
    while gw.stats["processed"] < n and time.time() - t0 < timeout:
        time.sleep(0.2)
    return gw.stats["processed"] >= n


def main():
    test_dir = Path("memory/_smoke_tool_rag")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    hc = Hippocampus(long_term_dir=str(test_dir))
    router = ToolRouter()
    gw = CortexGateway(model_path=MODEL, hippocampus=hc,
                       tool_executor=router.execute, idle_timeout=60.0,
                       on_action=lambda ev, ans: print(f"[Antwort #{ev['seq']}] {ans}"))

    print("\n--- 1) Ingest: stabiles Wissen (Wikipedia: Neuron) ---")
    obj = json.loads(wikipedia_api("Neuron"))
    text = f"[Wikipedia: {obj.get('title', 'Neuron')}]\n{obj.get('summary', '')}"
    hc.encode(text, salience=1.0, channel="ingest", vector=embed_text(None, text),
              answer="", tool="ingest", ttl=0, kind="knowledge")
    hc.consolidate()
    print("Wissens-Einträge:", hc.long_term_count(where={"kind": {"$eq": "knowledge"}}))

    print("\n--- 2) Wissensfrage -> Short-Circuit erwartet, KEIN Tool ---")
    gw.submit("Was ist ein Neuron?", salience=0.5, channel="user")
    assert wait(gw, 1), "Timeout bei Wissensfrage"

    print("\n--- 3) Volatile Frage -> Tool erwartet, KEIN Short-Circuit ---")
    gw.submit("Wie wird das Wetter morgen in Berlin?", salience=0.8, channel="user")
    assert wait(gw, 2), "Timeout bei Wetterfrage"

    gw.shutdown()
    s = gw.stats
    print("\nStats:", s)
    assert s["rag_shortcircuit"] >= 1, "Short-Circuit hat nicht gefeuert"
    assert s["tool_calls"] >= 1, "Wetter sollte ein Tool rufen"
    assert s["degraded"] == 0, "degraded Antworten"
    print("✅ Kette OK: Wissen -> Short-Circuit | volatil -> Tool")


if __name__ == "__main__":
    main()