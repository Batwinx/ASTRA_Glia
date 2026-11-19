"""
ASTRA_Glia · Hippocampus (Vektor-Gedächtnis, ChromaDB embedded)
Short-term: flüchtig (deque, RAM). Long-term: ChromaDB, Cosine-HNSW.

Schema: kind, tool, expires_at (0 = nie), hash (Inhalt), dedup_key (Slot).
Dedup ist DETERMINISTISCH: gleicher Slot = Refresh, sonst neu.
Cosine-Dedup ist deaktiviert – er löschte bei mean-gepoolten Embeddings
verschiedene Erinnerungen.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError("chromadb fehlt – bitte installieren: pip install chromadb") from e

_BATCH = 500
TELEMETRY_CHANNELS = ("konsolidierung", "system")

TOOL_TTL = {
    "wetter.api": 3600.0,
    "weather.api": 3600.0,
    "search.api": 86400.0,
    "wikipedia.api": 2592000.0,
}


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class Hippocampus:
    def __init__(self, embed_dim=2048, short_term_size=256,
                 long_term_dir="memory/long_term", consolidate_min_salience=0.5,
                 collection_name="astra_long_term", dedup_cosine=0.0):
        self.embed_dim = int(embed_dim)
        self.short_term = deque(maxlen=short_term_size)
        self.consolidate_min_salience = float(consolidate_min_salience)
        self.dir = Path(long_term_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._collection_name = collection_name
        if dedup_cosine:
            print("[Hippocampus] WARNUNG: Cosine-Dedup ist deaktiviert "
                  "(er überschrieb verschiedene Erinnerungen). "
                  "Dedup läuft deterministisch über den Slot-Key.")
        self._client = chromadb.PersistentClient(
            path=str(self.dir / "chroma"),
            settings=Settings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        self._migrate_legacy_numpy()
        self._backfill_schema()
        self._sync_dim_from_db()
        self.stats = {"encoded": 0, "consolidated": 0, "updated": 0,
                      "recalled": 0, "purged": 0}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _normalize(vec):
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def _hash(text, answer):
        return hashlib.sha1((_norm(text) + "|" + _norm(answer)).encode("utf-8")).hexdigest()

    @staticmethod
    def _dedup_key(kind, tool, text):
        """Telemetrie: leer = nie dedup. Wissen: gleicher Slot = Cache-Refresh."""
        if kind == "telemetry":
            return ""
        return hashlib.sha1(f"{kind}|{tool}|{_norm(text)}".encode("utf-8")).hexdigest()

    @staticmethod
    def _combine(where, extra):
        if where and extra:
            return {"$and": [where, extra]}
        return where or extra

    @staticmethod
    def _fresh_clause():
        now = time.time()
        return {"$or": [{"expires_at": {"$eq": 0.0}}, {"expires_at": {"$gt": now}}]}

    @staticmethod
    def _matches(meta, where):
        if not where:
            return True
        for key, cond in where.items():
            if key == "$and":
                if not all(Hippocampus._matches(meta, c) for c in cond):
                    return False
            elif key == "$or":
                if not any(Hippocampus._matches(meta, c) for c in cond):
                    return False
            elif isinstance(cond, dict):
                val = meta.get(key)
                for op, want in cond.items():
                    if op == "$eq" and val != want:
                        return False
                    elif op == "$ne" and (val is None or val == want):
                        return False
                    elif op == "$in" and (val is None or val not in want):
                        return False
                    elif op == "$nin" and (val is None or val in want):
                        return False
                    elif op in ("$gt", "$gte", "$lt", "$lte"):
                        if val is None:
                            return False
                        if op == "$gt" and not val > want:
                            return False
                        if op == "$gte" and not val >= want:
                            return False
                        if op == "$lt" and not val < want:
                            return False
                        if op == "$lte" and not val <= want:
                            return False
            elif meta.get(key) != cond:
                return False
        return True

    def _meta_for(self, item):
        channel = str(item.get("channel", ""))
        tool = str(item.get("tool") or "")
        kind = item.get("kind") or ("telemetry" if channel in TELEMETRY_CHANNELS
                                    else "knowledge")
        ttl = item.get("ttl")
        if ttl is None:
            ttl = TOOL_TTL.get(tool)
        expires = float(time.time() + ttl) if ttl else 0.0
        text = item.get("text", "")
        answer = item.get("answer", "")
        return {"salience": float(item.get("salience", 0.0)),
                "channel": channel, "kind": kind, "tool": tool,
                "answer": answer, "t": float(item.get("t", time.time())),
                "expires_at": expires,
                "hash": self._hash(text, answer),
                "dedup_key": self._dedup_key(kind, tool, text)}

    def _sync_dim_from_db(self):
        if self._col.count() == 0:
            return
        try:
            peek = self._col.get(limit=1, include=["embeddings"])
            emb = peek.get("embeddings")
            if emb is not None and len(emb) > 0:
                new_dim = int(np.asarray(emb[0]).shape[0])
                if new_dim != self.embed_dim:
                    print(f"[Hippocampus] Dimension angepasst: {self.embed_dim} -> {new_dim}")
                    self.embed_dim = new_dim
        except Exception as e:
            print(f"[Hippocampus] Dimension konnte nicht synchronisiert werden: {e}")

    def _check_dim(self, dim):
        if self._col.count() == 0:
            self.embed_dim = int(dim)
        elif dim != self.embed_dim:
            raise ValueError("Embedding-Dimension passt nicht zum Long-term-Speicher")

    # ------------------------------------------------------------- Diagnose
    def long_term_count(self, where=None, where_document=None):
        with self._lock:
            if where is None and where_document is None:
                return int(self._col.count())
            try:
                kw = {}
                if where:
                    kw["where"] = where
                if where_document:
                    kw["where_document"] = where_document
                return len(self._col.get(include=["metadatas"], **kw)["ids"])
            except Exception as e:
                print(f"[Hippocampus] count(filter) fehlgeschlagen: {e}")
                return 0

    def peek(self, where=None, limit=10):
        with self._lock:
            try:
                kw = {"where": where} if where else {}
                res = self._col.get(limit=limit, include=["metadatas", "documents"], **kw)
            except Exception as e:
                print(f"[Hippocampus] peek fehlgeschlagen: {e}")
                return []
            out = []
            for doc, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
                m = dict(meta or {})
                m["text"] = doc
                out.append(m)
            return out

    def _scan_meta(self):
        with self._lock:
            offset = 0
            while True:
                b = self._col.get(limit=_BATCH, offset=offset, include=["metadatas"])
                ids = b.get("ids") or []
                if not ids:
                    return
                for m in b.get("metadatas") or []:
                    yield dict(m or {})
                offset += len(ids)

    def kind_stats(self):
        kinds, channels, tools, expired = Counter(), Counter(), Counter(), 0
        now = time.time()
        for m in self._scan_meta():
            kinds[m.get("kind", "?")] += 1
            channels[m.get("channel", "?")] += 1
            if m.get("tool"):
                tools[m["tool"]] += 1
            exp = float(m.get("expires_at", 0.0) or 0.0)
            if exp and exp < now:
                expired += 1
        return {"total": sum(kinds.values()), "kinds": dict(kinds),
                "channels": dict(channels), "tools": dict(tools), "expired": expired}

    def dedup_report(self):
        """Slot-/Inhalts-Kollisionen: sollten nach H1 immer 0 sein."""
        slots, hashes = Counter(), Counter()
        for m in self._scan_meta():
            if m.get("dedup_key"):
                slots[m["dedup_key"]] += 1
            if m.get("hash"):
                hashes[m["hash"]] += 1
        return {"slots": len(slots),
                "slot_duplicates": sum(1 for v in slots.values() if v > 1),
                "content_duplicates": sum(1 for v in hashes.values() if v > 1)}

    # ------------------------------------------------------------- Wartung
    def delete(self, where=None, where_document=None):
        if where is None and where_document is None:
            return 0
        with self._lock:
            before = self._col.count()
            try:
                kw = {}
                if where:
                    kw["where"] = where
                if where_document:
                    kw["where_document"] = where_document
                self._col.delete(**kw)
            except Exception as e:
                print(f"[Hippocampus] delete fehlgeschlagen: {e}")
                return 0
            n = int(before - self._col.count())
            self.stats["purged"] += n
            return n

    def purge_expired(self):
        now = time.time()
        return self.delete(where={"$and": [{"expires_at": {"$gt": 0.0}},
                                           {"expires_at": {"$lt": now}}]})

    def dedup_content(self, keep="newest"):
        """Exakte Inhalts-Duplikate entfernen: pro Hash bleibt ein Eintrag."""
        groups = {}
        with self._lock:
            offset = 0
            while True:
                b = self._col.get(limit=_BATCH, offset=offset, include=["metadatas"])
                ids = b.get("ids") or []
                if not ids:
                    break
                for i, m in zip(ids, b.get("metadatas") or []):
                    h = (m or {}).get("hash") or ""
                    if h:
                        groups.setdefault(h, []).append((float((m or {}).get("t", 0.0)), i))
                offset += len(ids)
            drop = []
            for items in groups.values():
                if len(items) < 2:
                    continue
                items.sort(reverse=(keep == "newest"))   # neuester zuerst -> bleibt
                drop.extend(i for _, i in items[1:])
            for s in range(0, len(drop), _BATCH):
                self._col.delete(ids=drop[s:s + _BATCH])
            self.stats["purged"] += len(drop)
            return len(drop)

    def reset(self):
        with self._lock:
            n = self._col.count()
            self._client.delete_collection(self._collection_name)
            self._col = self._client.get_or_create_collection(
                name=self._collection_name, metadata={"hnsw:space": "cosine"})
            self.stats["purged"] += n
            return n

    def _backfill_schema(self):
        if self._col.count() == 0:
            return
        needed = ("kind", "tool", "expires_at", "hash", "dedup_key")
        offset, fixed = 0, 0
        while True:
            b = self._col.get(limit=_BATCH, offset=offset,
                              include=["metadatas", "documents"])
            ids = b.get("ids") or []
            if not ids:
                break
            new_ids, new_metas = [], []
            for i, m, d in zip(ids, b.get("metadatas") or [], b.get("documents") or []):
                m = dict(m or {})
                if all(k in m for k in needed):
                    continue
                ch = str(m.get("channel", ""))
                m.setdefault("kind", "telemetry" if ch in TELEMETRY_CHANNELS else "knowledge")
                m.setdefault("tool", "")
                m.setdefault("expires_at", 0.0)
                m.setdefault("hash", self._hash(d or "", m.get("answer", "")))
                m["dedup_key"] = self._dedup_key(m["kind"], m["tool"], d or "")
                new_ids.append(i)
                new_metas.append(m)
            if new_ids:
                try:
                    self._col.update(ids=new_ids, metadatas=new_metas)
                    fixed += len(new_ids)
                except Exception as e:
                    print(f"[Hippocampus] Backfill fehlgeschlagen: {e}")
                    break
            offset += len(ids)
        if fixed:
            print(f"🧬 [Hippocampus] Schema-Backfill: {fixed} Alt-Eintrag/Eintraege ergänzt.")

    # ------------------------------------------------------------- Schreiben

    def encode(self, text, salience, channel, vector, answer="",
               tool=None, ttl=None, kind=None, metadata=None):
        vec = self._normalize(vector)
        with self._lock:
            self.short_term.append({
                "text": text, "salience": float(salience), "channel": channel,
                "answer": answer, "t": time.time(), "consolidated": False,
                "vector": vec, "tool": tool, "ttl": ttl, "kind": kind,
                "metadata": dict(metadata or {}),
            })
            self.stats["encoded"] += 1

    def _query(self, q, k, n, where):
        args = dict(query_embeddings=[q.tolist()], n_results=min(max(k, 1), n),
                    include=["metadatas", "distances", "documents"])
        if where:
            try:
                return self._col.query(where=where, **args)
            except Exception as e:
                print(f"[Hippocampus] where serverseitig nicht moeglich ({e}) -> lokal")
                args["n_results"] = min(max(k * 10, 50), n)
        return self._col.query(**args)

    def consolidate(self):
        """Saliente Items nach Long-term. Dedup: Inhalt-Hash, dann Slot-Key."""
        new_count, upd_count, total = 0, 0, 0
        with self._lock:
            items = [it for it in self.short_term
                     if not it["consolidated"]
                     and it["salience"] >= self.consolidate_min_salience]
            if items:
                vecs = np.stack([it["vector"] for it in items])
                self._check_dim(vecs.shape[1])
                adds = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
                for it in items:
                    meta = self._meta_for(it)
                    target = None
                    # Telemetrie hat keinen Slot -> niemals deduplizieren
                    if meta["dedup_key"]:
                        try:      # 1) identischer Inhalt innerhalb derselben kind
                            ex = self._col.get(
                                where={"$and": [{"hash": {"$eq": meta["hash"]}},
                                                {"kind": {"$eq": meta["kind"]}}]},
                                include=["metadatas"])
                            for i, m in zip(ex.get("ids") or [],
                                            ex.get("metadatas") or []):
                                if (m or {}).get("kind") == meta["kind"]:
                                    target = i
                                    break
                        except Exception:
                            pass
                        if target is None:   # 2) gleicher Slot
                            try:
                                ex = self._col.get(
                                    where={"dedup_key": {"$eq": meta["dedup_key"]}},
                                    include=["metadatas"])
                                if ex.get("ids"):
                                    target = ex["ids"][0]
                            except Exception:
                                pass
                    if target is not None:
                        self._col.update(ids=[target],
                                         embeddings=[it["vector"].tolist()],
                                         documents=[it["text"]], metadatas=[meta])
                        upd_count += 1
                    else:
                        adds["ids"].append(uuid.uuid4().hex)
                        adds["embeddings"].append(it["vector"].tolist())
                        adds["documents"].append(it["text"])
                        adds["metadatas"].append(meta)
                    it["consolidated"] = True
                n_new = len(adds["ids"])
                for s in range(0, n_new, _BATCH):
                    self._col.add(ids=adds["ids"][s:s + _BATCH],
                                  embeddings=adds["embeddings"][s:s + _BATCH],
                                  documents=adds["documents"][s:s + _BATCH],
                                  metadatas=adds["metadatas"][s:s + _BATCH])
                new_count = n_new
                self.stats["consolidated"] += len(items)
                self.stats["updated"] += upd_count
            total = self._col.count()
        if new_count or upd_count:
            print(f"🧬 [Hippocampus] {new_count} neu, {upd_count} aktualisiert "
                  f"(Long-term gesamt {total}).")
        return new_count + upd_count

    # ------------------------------------------------------------- Lesen
    def recall(self, query_vector, k=5, where=None, fresh_only=False):
        q = self._normalize(query_vector)
        eff = self._combine(where, self._fresh_clause()) if fresh_only else where
        results = []
        with self._lock:
            n = self._col.count()
            if n > 0:
                if q.shape[0] != self.embed_dim:
                    raise ValueError("Query-Dimension passt nicht zum Long-term-Speicher")
                res = self._query(q, k, n, eff)
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    m = dict(meta or {})
                    m["text"] = doc
                    m["score"] = float(1.0 - dist)
                    m["source"] = "long_term"
                    if eff and not self._matches(m, eff):
                        continue
                    results.append(m)
            st = [it for it in self.short_term
                  if not it["consolidated"] and self._matches(it, eff)]
            if st:
                sims = np.stack([it["vector"] for it in st]) @ q
                for j in np.argsort(-sims)[:k]:
                    it = st[j]
                    results.append({"text": it["text"], "salience": it["salience"],
                                    "channel": it["channel"], "answer": it["answer"],
                                    "kind": it.get("kind") or "knowledge",
                                    "tool": it.get("tool") or "", "expires_at": 0.0,
                                    "score": float(sims[j]), "source": "short_term"})
            results.sort(key=lambda m: -m["score"])
            self.stats["recalled"] += 1
        return results[:k]

    # ------------------------------------------------- Migration numpy -> DB
    def _migrate_legacy_numpy(self):
        vp, mp = self.dir / "vectors.npy", self.dir / "meta.jsonl"
        if not (vp.exists() and mp.exists()) or self._col.count() > 0:
            return
        try:
            vectors = np.load(vp)
            meta = []
            with mp.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        meta.append(json.loads(line))
            n = min(len(vectors), len(meta))
            if n == 0:
                return
            vectors = np.asarray(vectors[:n], dtype=np.float32)
            ids = [uuid.uuid4().hex for _ in range(n)]
            docs = [m.get("text", "") for m in meta[:n]]
            metas = [self._meta_for(m) for m in meta[:n]]
            for s in range(0, n, _BATCH):
                self._col.add(ids=ids[s:s + _BATCH],
                              embeddings=vectors[s:s + _BATCH].tolist(),
                              documents=docs[s:s + _BATCH], metadatas=metas[s:s + _BATCH])
            vp.rename(vp.with_suffix(".npy.migrated"))
            mp.rename(mp.with_suffix(".jsonl.migrated"))
            print(f"🧬 [Hippocampus] {n} Alt-Erinnerungen (numpy) nach ChromaDB migriert.")
        except Exception as e:
            print(f"[Hippocampus] Legacy-Migration übersprungen: {e}")