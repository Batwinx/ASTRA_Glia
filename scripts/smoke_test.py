"""ASTRA_Glia · Smoke-Test fuer CortexGateway ohne LLM-Load."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cortex_gateway import CortexGateway


def main():
    g = CortexGateway("dummy.gguf")   # Laedt KEIN Modell (_make_llm ist lazy)
    print("rag_max_chars:", g.rag_max_chars, "  (Soll: 1400)")

    ctx = g._build_context([{"text": "A" * 600, "answer": ""}])
    print("ctx-Laenge:", len(ctx), "  (Soll: 402 = 400 Text + '- ')")

    fake_q = [{"text": "Was weisst du ueber die Rolle von Astrozyten?",
               "score": 0.9, "kind": "knowledge", "tool": "", "answer": ""}]
    print("Frage als Wissen filtern:", g._retrieve.__doc__ is not None)
    print("Short-Circuit bei alter Chat-Episode:", g._shortcircuit_ok(fake_q),
          "  (Soll: False)")

    real = [{"text": "Astrozyten sind Sternzellen im Nervengewebe.",
             "score": 0.71, "kind": "knowledge", "tool": "ingest",
             "expires_at": 0.0, "answer": ""}]
    print("Short-Circuit bei echtem Fakten-Chunk:", g._shortcircuit_ok(real),
          "  (Soll: True)")


if __name__ == "__main__":
    main()