"""Batch-Ingest für die Wissensbasis (FIAE-Standard: Robust, dokumentiert, mit Report).
Aufruf:  python scripts/ingest_knowledge_batch.py
"""

import sys
import time
import re
from pathlib import Path
from typing import Optional, Tuple, List, Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hippocampus import Hippocampus
from core.cortex_gateway import embed_text


# ---------------------------------------------------------------------------
# 1. Konfiguration
# ---------------------------------------------------------------------------

TOPICS = [
    "Neuron", "Synapse", "Astrozyt", "Gliazelle", "Myelin",
    "Axon", "Dendrit_(Biologie)", "Neurotransmitter", "Aktionspotenzial",
    "Gehirn", "Hippocampus", "Großhirnrinde", "Cerebellum",
    "Neuroplastizität", "Gedächtnis", "Lernen", "Lernen_lernen", "Schlaf"
]

WIKI_API_URL = "https://de.wikipedia.org/w/api.php"
CHUNK_MAX_LENGTH = 800  # Zeichen pro Chunk für optimale RAG-Performance

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
TOPIC_DELAY_SECONDS = 0.25  # kleine Pause zwischen Artikeln, höflich gegenüber API

# WICHTIG: Kontakt ersetzen, z. B. echte E-Mail oder Projekt-URL.
USER_AGENT = (
"ASTRA_Glia/1.0 (Batch-Ingest Wissensbasis; https://github.com/Batwinx/ASTRA_Glia) "
    f"python-requests/{requests.__version__}"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Api-User-Agent": USER_AGENT,
    "Accept": "application/json",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# 2. Wikipedia-Abruf
# ---------------------------------------------------------------------------
def clean_wiki_text(text):
    # Wiki-Headings entfernen
    text = re.sub(r"={2,}\s*[^=\n]+?\s*={2,}", "\n", text)
    # Klebe-Stellen nach Kleinbuchstabe + Großbuchstabe auftrennen
    # (aber Einheiten wie mV bewusst NICHT – die stehen schon getrennt oder sind ok)
    text = re.sub(r"([a-zäöüß])([A-ZÄÖÜ][a-zäöüß])", r"\1 \2", text)
    # Mehrfache Leerzeichen/Newlines normalisieren
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fetch_wikipedia_summary(title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Holt den Plain-Text eines Wikipedia-Artikels.

    Rückgabe:
        (real_title, text)
    oder:
        (None, None), falls Artikel nicht gefunden / Abruf fehlgeschlagen.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=20)

            # Wikipedia blockiert häufig fehlende/generische User-Agents.
            if response.status_code == 403:
                print("  ❌ 403 Forbidden: Wikipedia blockiert die Anfrage.")
                print("     Prüfe User-Agent/Kontakt. Falls gerade viele 403 kamen:")
                print("     möglichen temporären IP-Ban ca. 15–30 Minuten abwarten.")
                return None, None

            # Temporäre Server-/Rate-Limit-Fehler: kurzer Retry.
            if response.status_code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"  ⚠️ HTTP {response.status_code}. Neuer Versuch in {wait:.1f}s ...")
                time.sleep(wait)
                continue

            if response.status_code >= 400:
                print(f"  ❌ HTTP {response.status_code} beim Abrufen von '{title}'.")
                return None, None

            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None, None

            page_id = next(iter(pages))
            page = pages[page_id]

            if page_id == "-1" or "missing" in page:
                return None, None

            real_title = page.get("title", title)
            text = (page.get("extract") or "").strip()

            if not text:
                return None, None

            return real_title, text

        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"  ⚠️ Netzwerkfehler: {exc}. Neuer Versuch in {wait:.1f}s ...")
                time.sleep(wait)
                continue

            print(f"  ❌ Fehler beim Abrufen von '{title}': {exc}")
            return None, None

        except Exception as exc:
            print(f"  ❌ Unerwarteter Fehler beim Abrufen von '{title}': {exc}")
            return None, None

    return None, None


# ---------------------------------------------------------------------------
# 3. Robustes Chunking
# ---------------------------------------------------------------------------

def split_long_paragraph(paragraph: str, max_len: int) -> List[str]:
    """
    Zerlegt einen zu langen Absatz bevorzugt an Satzgrenzen.
    Nur wenn ein einzelner Satz immer noch zu lang ist, wird wortweise getrennt.
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    if len(paragraph) <= max_len:
        return [paragraph]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    if not sentences:
        sentences = [paragraph]

    pieces: List[str] = []
    current = ""

    for sentence in sentences:
        # Einzelner Satz länger als max_len: wortweise fallbacken.
        if len(sentence) > max_len:
            if current:
                pieces.append(current)
                current = ""

            words = sentence.split()
            buffer = ""

            for word in words:
                candidate = (buffer + " " + word).strip()

                if len(candidate) <= max_len:
                    buffer = candidate
                else:
                    if buffer:
                        pieces.append(buffer)

                    # Extremfall: einzelnes Wort länger als max_len.
                    while len(word) > max_len:
                        pieces.append(word[:max_len])
                        word = word[max_len:]

                    buffer = word

            current = buffer
            continue

        candidate = (current + " " + sentence).strip()

        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return [p for p in pieces if p.strip()]


def chunk_text(text: str, max_len: int = CHUNK_MAX_LENGTH) -> List[str]:
    """
    Zerlegt Text in sinnvolle Absätze/Chunks, ohne Wörter mitten zu trennen,
    außer im absoluten Fallback bei extrem langen Sätzen/Wörtern.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        for piece in split_long_paragraph(paragraph, max_len):
            candidate = (current_chunk + "\n\n" + piece).strip() if current_chunk else piece

            if len(candidate) <= max_len:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# 4. Hauptlogik
# ---------------------------------------------------------------------------

def main() -> None:
    print("🧠 Initialisiere Hippocampus...")
    hc = Hippocampus()

    stats = {
        "articles_success": 0,
        "articles_skipped": 0,
        "articles_failed": 0,
        "chunks_added": 0,
        "chunks_failed": 0,
    }

    start_time = time.time()

    print(f"\n🚀 Starte Batch-Ingest für {len(TOPICS)} Themen...\n")

    for topic in TOPICS:
        print(f"📥 Verarbeite: {topic}")
        title, text = fetch_wikipedia_summary(topic)

        if not title or not text:
            stats["articles_failed"] += 1
            continue

        # Deduplizierung: Titel-basiert über Metadaten (Chroma-konform)
        try:
            existing = hc._col.get(where={"article_title": {"$eq": title}}, limit=1)
            already = bool(existing.get("ids"))
        except Exception:
            already = False
        if already:
            print(f"  ⏭️  Übersprungen (bereits vorhanden): {title}")
            stats["articles_skipped"] += 1
            continue

        chunks = chunk_text(text)
        print(f"  ✂️  In {len(chunks)} Chunks zerlegt.")

        chunk_errors = 0
        for i, chunk in enumerate(chunks):
            try:
                vec = embed_text(None, chunk)
                meta = {
                    "source": "wikipedia",
                    "article_title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                hc.encode(
                    text=chunk,
                    salience=0.8,
                    channel="ingest",
                    vector=vec,
                    answer="",
                    tool="ingest",
                    ttl=0,
                    kind="knowledge",
                    metadata=meta,
                )
                stats["chunks_added"] += 1
            except Exception as e:
                chunk_errors += 1
                stats["chunks_failed"] += 1
                print(f"  ❌ Fehler bei Chunk {i}: {e}")

        # WICHTIG: Short-term (RAM) -> Long-term (ChromaDB) überführen!
        moved = hc.consolidate()
        print(f"  💾 {moved} Chunk(s) nach Long-term konsolidiert.")

        if chunk_errors == 0:
            stats["articles_success"] += 1
            print(f"  ✅ Erfolgreich: {title}")
        elif chunk_errors < len(chunks):
            stats["articles_success"] += 1
            print(f"  ⚠️  Teilweise fehlgeschlagen: {title} "
                  f"({chunk_errors}/{len(chunks)} Chunks)")
        else:
            stats["articles_failed"] += 1
            print(f"  ❌ Artikel fehlgeschlagen: {title}")

        # Embeddings vorab berechnen:
        # Wenn ein Embedding fehlschlägt, wird der Artikel nicht halb ingestiert.
        vectors: List[Any] = []
        embedding_failed = False

        for chunk in chunks:
            try:
                vec = embed_text(None, chunk)
                vectors.append(vec)
            except Exception as exc:
                print(f"  ❌ Embedding-Fehler: {exc}")
                embedding_failed = True
                break

        if embedding_failed or len(vectors) != len(chunks):
            stats["articles_failed"] += 1
            stats["chunks_failed"] += len(chunks)
            continue

        article_ok = True

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            try:
                meta = {
                    "source": "wikipedia",
                    "article_title": title,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }

                hc.encode(
                    text=chunk,
                    salience=0.8,
                    channel="ingest",
                    vector=vec,
                    answer="",      # Bei reinem Wissen leer lassen
                    tool="ingest",
                    ttl=0,          # 0 = nie verfallen
                    kind="knowledge",
                    metadata=meta,
                )

                stats["chunks_added"] += 1

            except Exception as exc:
                print(f"  ❌ Fehler beim Speichern von Chunk {i}: {exc}")
                stats["chunks_failed"] += 1
                article_ok = False

        if article_ok:
            stats["articles_success"] += 1
            print(f"  ✅ Erfolgreich: {title}")
        else:
            stats["articles_failed"] += 1
            print(f"  ⚠️ Teilweise fehlgeschlagen: {title}")

        time.sleep(TOPIC_DELAY_SECONDS)

    # Abschluss-Report
    duration = time.time() - start_time

    print("\n" + "=" * 60)
    print("📊 ABSCHLUSS-REPORT: WISSENSBASIS-UPDATE")
    print("=" * 60)
    print(f"⏱️  Gesamtzeit:            {duration:.1f} Sekunden")
    print(f"✅ Erfolgreiche Artikel:   {stats['articles_success']}")
    print(f"⏭️  Übersprungen:          {stats['articles_skipped']}")
    print(f"❌ Fehlgeschlagene Artikel: {stats['articles_failed']}")
    print(f"🧩 Neue Chunks hinzugefügt: {stats['chunks_added']}")
    print(f"💥 Chunk-Fehler:           {stats['chunks_failed']}")

    try:
        total_knowledge = hc.long_term_count(where={"kind": "knowledge"})
        print(f"💾 Gesamtbestand Wissen:   {total_knowledge} Einträge")
    except Exception as exc:
        print(f"⚠️ Gesamtbestand nicht ermittelbar: {exc}")

    print("=" * 60)


if __name__ == "__main__":
    main()