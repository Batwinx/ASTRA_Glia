"""Schneller RAG-Check: python tests/test_rag.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cortex_gateway import CortexGateway


def main():
    gw = CortexGateway(
        model_path=Path("checkpoints") / "llm" / "Qwen3-1.7B-Q8_0.gguf",
        on_action=lambda ev, ans: print(f"[Antwort #{ev['seq']}] {ans}"),
        idle_timeout=10.0)
    print("Long-term:", gw.hippocampus.long_term_count())
    for h in gw.recall("Konsolidierung nach Salienz-Reiz", k=3):
        print(f"  {h['score']:.3f} [{h['source']}] {h['text'][:80]}")
    gw.submit("Was ist ueber bisherige Konsolidierungs-Reize bekannt?",
              salience=0.6, channel="user")
    t0 = time.time()
    while gw.stats["processed"] < 1 and time.time() - t0 < 60:
        time.sleep(0.2)
    gw.shutdown()
    print("Stats:", gw.stats)


if __name__ == "__main__":
    main()