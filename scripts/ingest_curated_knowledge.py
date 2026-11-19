"""ASTRA_Glia · Kuratierte Vergleichs-Chunks fuer praezise RAG-Antworten.

Ergaenzt die Wikipedia-Wissensbasis um dichte Gegenueberstellungen.
Gleiche Metadaten-Struktur wie der Batch-Ingest, nur source="astra_curated".

Aufruf:
    python scripts/ingest_curated_knowledge.py
    python scripts/ingest_curated_knowledge.py --force
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hippocampus import Hippocampus
from core.cortex_gateway import embed_text

CURATED = [
    {
        "title": "Vergleich: Axon und Dendrit",
        "section": "Gegenueberstellung",
        "text": (
            "Unterschied zwischen Axon und Dendrit: Dendriten sind meist kurze, "
            "stark verzweigte Zellfortsätze eines Neurons, die Signale von anderen "
            "Nervenzellen empfangen und zum Zellkörper (Soma) hinleiten; sie sind "
            "die Eingangsseite der Zelle. Das Axon ist in der Regel ein einzelner, "
            "langer Fortsatz, der Aktionspotenziale vom Zellkörper weg zu den "
            "Axonendigungen leitet und dort über Synapsen an andere Neuronen, "
            "Muskeln oder Drüsen weitergibt; es ist die Ausgangsseite. Axone können "
            "myelinisiert sein und dadurch schneller leiten, Dendriten nicht. "
            "Pro Neuron gibt es viele Dendriten, aber meist nur ein Axon."
        ),
    },
    {
        "title": "Vergleich: Astrozyt und Oligodendrozyt",
        "section": "Gegenueberstellung",
        "text": (
            "Unterschied zwischen Astrozyten und Oligodendrozyten: Astrozyten sind "
            "sternförmige Gliazellen, die das chemische Milieu der Synapsen "
            "regulieren, Kalium und Neurotransmitter aufnehmen, die Blut-Hirn-"
            "Schranke stützen und den Stoffwechsel der Neuronen versorgen. "
            "Oligodendrozyten bilden im Zentralnervensystem die Myelinscheiden um "
            "Axone; eine Zelle kann mehrere Axone gleichzeitig myelinisieren. "
            "Im peripheren Nervensystem übernehmen Schwann-Zellen diese Aufgabe, "
            "dort myelinisiert jede Zelle nur einen Axonabschnitt."
        ),
    },
    {
        "title": "Vergleich: erregende und hemmende Synapse",
        "section": "Gegenueberstellung",
        "text": (
            "Unterschied zwischen erregender und hemmender Synapse: Erregende "
            "(exzitatorische) Synapsen wirken meist über Glutamat, depolarisieren "
            "die postsynaptische Membran (EPSP) und erhöhen die Wahrscheinlichkeit "
            "eines Aktionspotenzials. Hemmende (inhibitorische) Synapsen wirken "
            "meist über GABA oder Glycin, hyperpolarisieren die Membran (IPSP) und "
            "senken diese Wahrscheinlichkeit. Erregende Synapsen sitzen bevorzugt "
            "an Dendriten-Dornen, hemmende eher am Soma und am Axonhügel."
        ),
    },
]


def main():
    ap = argparse.ArgumentParser(description="Kuratierte Vergleichs-Chunks einspielen")
    ap.add_argument("--force", action="store_true", help="vorhandene Titel ersetzen")
    args = ap.parse_args()

    hc = Hippocampus()
    added = skipped = 0

    for item in CURATED:
        title = item["title"]
        try:
            exists = bool(hc._col.get(where={"article_title": {"$eq": title}},
                                      limit=1).get("ids"))
        except Exception:
            exists = False

        if exists and not args.force:
            print(f"⏭️  vorhanden: {title}")
            skipped += 1
            continue
        if exists and args.force:
            hc.delete(where={"article_title": {"$eq": title}})

        text = item["text"]
        vec = embed_text(None, text)
        meta = {
            "source": "astra_curated",
            "article_title": title,
            "section": item["section"],
            "chunk_index": 0,
            "total_chunks": 1,
        }
        hc.encode(text=text, salience=0.9, channel="ingest", vector=vec,
                  answer="", tool="ingest", ttl=0, kind="knowledge", metadata=meta)
        moved = hc.consolidate()
        added += 1
        print(f"✅ {title}  ({len(text)} Zeichen, {moved} konsolidiert)")

    print(f"\n📊 neu: {added} | übersprungen: {skipped}")
    print(f"💾 Wissen gesamt: {hc.long_term_count(where={'kind': {'$eq': 'knowledge'}})}")


if __name__ == "__main__":
    main()