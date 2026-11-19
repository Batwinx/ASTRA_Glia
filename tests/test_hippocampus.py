"""Smoke-Test für den ChromaDB-Hippocampus (ohne LLM/GPU).
Aufruf vom Projekt-Root:  python tests/test_hippocampus.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from core.hippocampus import Hippocampus


def main():
    test_dir = Path("memory/_smoke_hippocampus")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    hc = Hippocampus(long_term_dir=str(test_dir), short_term_size=8)
    rng = np.random.default_rng(42)
    a = rng.normal(size=2048).astype(np.float32)
    b = rng.normal(size=2048).astype(np.float32)

    hc.encode("Wetter in Berlin", 0.9, "test", a, answer='{"action": "wetter.api"}')
    hc.encode("Smalltalk", 0.2, "test", b)          # unter Schwelle -> bleibt short-term
    assert hc.consolidate() == 1, "genau ein Item muss konsolidiert werden"
    assert hc.consolidate() == 0, "nichts doppelt konsolidieren"

    hits = hc.recall(a, k=2)
    assert hits and hits[0]["source"] == "long_term"
    assert hits[0]["text"] == "Wetter in Berlin"
    assert hits[0]["score"] > 0.99, f"Cosine-Score zu niedrig: {hits[0]['score']}"
    assert hc.long_term_count() == 1

    # Persistenz: neue Instanz auf demselben Verzeichnis
    hc2 = Hippocampus(long_term_dir=str(test_dir))
    assert hc2.long_term_count() == 1
    assert hc2.recall(a, k=1)[0]["text"] == "Wetter in Berlin"
    print("✅ Hippocampus/ChromaDB OK:", hc.stats)


if __name__ == "__main__":
    main()