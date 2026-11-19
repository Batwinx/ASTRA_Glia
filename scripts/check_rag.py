"""ASTRA_Glia · RAG-Diagnose: Retrieval, Short-Circuit-Schwelle, Bestandsinventar.

Aufruf:
    python scripts/check_rag.py
    python scripts/check_rag.py --inventory
    python scripts/check_rag.py --query "Unterschied Axon vs. Dendrit"
    python scripts/check_rag.py --purge-title Dendrit --purge-title Cortex
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hippocampus import Hippocampus
from core.cortex_gateway import embed_text, RAG_SHORTCIRCUIT

# Exakt der Filter aus CortexGateway._retrieve()
RAG_WHERE = {"$or": [{"kind": {"$eq": "knowledge"}}, {"tool": {"$eq": "ingest"}}]}

DEFAULT_QUERIES = [
    "Unterschied Axon vs. Dendrit",
    "Was ist der Unterschied zwischen Axon und Dendrit?",
    "Wie entsteht ein Aktionspotenzial?",
    "Was macht ein Astrozyt?",
    "Wozu dient Myelin?",
    "Was passiert an einer Synapse?",
]


def purge(hc, titles):
    print("\n🧹 Cleanup")
    for t in titles:
        n = hc.delete(where={"article_title": {"$eq": t}})
        print(f"   {n:4d} Chunk(s) gelöscht  article_title={t!r}")


def inventory(hc):
    counts = Counter()
    for m in hc._scan_meta():
        if m.get("kind") == "knowledge":
            counts[m.get("article_title") or "(ohne Titel)"] += 1
    print("\n📚 Wissensbestand pro Artikel")
    for title, n in sorted(counts.items(), key=lambda x: -x[1]):
        flag = "   ⚠️ verdächtig klein" if n <= 2 else ""
        print(f"   {n:4d}  {title}{flag}")
    print(f"   ---- Summe: {sum(counts.values())}")
def show_legacy(hc, limit=20):
    rows = hc.peek(where={"kind": {"$eq": "knowledge"}}, limit=5000)
    legacy = [m for m in rows if not m.get("article_title")]
    print(f"\n🕳️  {len(legacy)} Wissens-Eintrag/Eintraege ohne article_title")
    for i, m in enumerate(legacy[:limit], 1):
        print(f"   {i}. tool={m.get('tool')!r} | {(m.get('text') or '')[:150]}")


def purge_contains(hc, fragments):
    print("\n🧹 Text-Cleanup")
    for frag in fragments:
        n = hc.delete(where_document={"$contains": frag})
        print(f"   {n:4d} Chunk(s) gelöscht  enthält={frag!r}")


def check(hc, queries, k):
    for q in queries:
        hits = hc.recall(embed_text(None, q), k=k, where=RAG_WHERE)
        print("\n" + "=" * 72)
        print(f"Frage: {q}")
        if not hits:
            print("   ❌ KEINE TREFFER – Filter oder Bestand prüfen")
            continue

        score = float(hits[0].get("score", 0.0))
        fired = score >= RAG_SHORTCIRCUIT
        print(f"   Top-Score {score:.3f} | Schwelle {RAG_SHORTCIRCUIT:.2f} -> "
              f"{'✅ SHORT-CIRCUIT FEUERT' if fired else '⚠️ unter Schwelle'}")

        for i, h in enumerate(hits, 1):
            print(f"    #{i} {h.get('score', 0.0):.3f} | {h.get('article_title', '?')} "
                  f"| kind={h.get('kind')} tool={h.get('tool')}")
            print(f"        {h.get('text', '')[:120]}")

        low = q.lower()
        if "axon" in low and "dendrit" in low:
            joined = " ".join(h.get("text", "") for h in hits[:2]).lower()
            ok = "axon" in joined and "dendrit" in joined
            print(f"   Vergleichs-Check (Axon & Dendrit in Top-2): "
                  f"{'✅ OK' if ok else '❌ FEHLT -> kuratierter Vergleichs-Chunk nötig'}")


def main():
    ap = argparse.ArgumentParser(description="ASTRA_Glia RAG-Diagnose")
    ap.add_argument("--query", action="append", default=None, help="eigene Testfrage(n)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--purge-title", action="append", default=None,
                    help="Artikel mit exakt diesem article_title löschen")
    ap.add_argument("--purge-contains", action="append", default=None,
                    help="Chunks löschen, die diesen Text enthalten")
    ap.add_argument("--show-legacy", action="store_true",
                    help="Wissens-Einträge ohne article_title anzeigen")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--only-inventory", action="store_true")
    ap.add_argument("--find-glued", action="store_true",
                    help="Chunks mit verklebten Woertern auflisten")
    args = ap.parse_args()

    hc = Hippocampus()
    if args.purge_title:
        purge(hc, args.purge_title)
    if args.purge_contains:
        purge_contains(hc, args.purge_contains)
    def find_glued(hc, limit=25):
        import re
        # Echte Klebe-Stellen, aber NICHT wissenschaftliche Einheiten (mV, pH, kDa ...)
        units = {"mV", "pH", "kD", "aV", "mR", "sB", "oP", "eG", "bH", "dU",
                "xP", "kF", "nD", "nF", "eD", "oS", "sG", "mW", "eN"}
        rows = hc.peek(where={"kind": {"$eq": "knowledge"}}, limit=5000)
        pat = re.compile(r"[a-zäöüß][A-ZÄÖÜ]")
        hits = []
        for m in rows:
            t = m.get("text") or ""
            real = [g for g in pat.findall(t) if g not in units]
            if real:
                hits.append((m, real))
        print(f"\n🔤 {len(hits)} Chunk(s) mit echten Klebe-Stellen (ohne Einheiten)")
        for i, (m, found) in enumerate(hits[:limit], 1):
            print(f"   {i}. {m.get('article_title')} | {found[:5]}")
            print(f"      {(m.get('text') or '')[:140]}")

    if args.show_legacy:
        show_legacy(hc)
    if args.find_glued:
        find_glued(hc)
    if args.inventory or args.only_inventory:
        inventory(hc)
    if not args.only_inventory:
        check(hc, args.query or DEFAULT_QUERIES, args.k)
    print(f"\n💾 Wissen gesamt: {hc.long_term_count(where={'kind': {'$eq': 'knowledge'}})}")

if __name__ == "__main__":
    main()