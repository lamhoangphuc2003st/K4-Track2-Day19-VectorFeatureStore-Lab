"""HybridMemoryAgent — episodic memory (Qdrant + BM25) + stable profile (Feast).

Design notes live in bonus/ARCHITECTURE.md. Short version:
  * remember(): sentence-aware chunking → embed → upsert into ONE Qdrant
    collection, every point tagged with `user_id` (tenant isolation by filter).
  * recall(): Feast online lookup (profile + recent activity) → per-user hybrid
    search (BM25 + vector, RRF k=60, rank 1-based) + a half-weight "affinity"
    ranker from the profile → assembled context string (no LLM call).
"""
from __future__ import annotations

import re
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

FEAST_REPO = Path(__file__).resolve().parent.parent / "app" / "feast_repo"
COLLECTION = "episodic_memory"
PROFILE_FEATURES = [
    "user_profile_features:preferred_language",
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]


def fold(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics ("tự động" → "tu dong", "đ" → "d")."""
    text = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    # Index the folded form only, so users who type without dấu still match.
    return re.findall(r"\w+", fold(text))


# Code-switching: VN users mix English tech terms into Vietnamese queries, while
# the stored text says "bảo mật", "đám mây"… Expand the query side only.
GLOSSARY = {
    "security": "bảo mật", "cloud": "đám mây điện toán", "database": "cơ sở dữ liệu",
    "scale": "mở rộng", "autoscaling": "tự động mở rộng", "deploy": "triển khai",
    "cost": "chi phí", "network": "mạng", "monitoring": "giám sát",
}
RECENCY_CUES = ("gan day", "recent", "moi doc", "vua doc", "hom nay")


def expand(query: str) -> str:
    extra = [GLOSSARY[t] for t in tokenize(query) if t in GLOSSARY]
    return " ".join([query, *extra])


def chunk(text: str, max_words: int = 80) -> list[str]:
    """Group whole sentences into chunks of ≤ max_words, 1-sentence overlap."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-1:]                       # overlap keeps cross-sentence context
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@dataclass
class Memory:
    point_id: str
    memory_id: str                                        # all chunks of one remember() call
    text: str
    title: str
    topic: str
    source: str
    created_at: float
    score: float = 0.0


@dataclass
class HybridMemoryAgent:
    feast_repo: Path = FEAST_REPO
    rrf_k: int = 60
    top_k: int = 3
    embedder: Embedder = field(default_factory=Embedder)
    _activity: dict[str, list[tuple[float, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE))
        self._mem: dict[str, dict[str, Memory]] = {}     # user_id → point_id → Memory (BM25 side)
        self._bm25: dict[str, tuple[BM25Okapi, list[str]]] = {}
        try:
            from feast import FeatureStore
            self.fs = FeatureStore(repo_path=str(self.feast_repo))
        except Exception:                                 # Feast optional: degrade to memory-only
            self.fs = None

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001", topic: str = "note",
                 source: str = "note") -> None:
        """Add a new piece of episodic memory for this user."""
        pieces = chunk(text)
        title = pieces[0].split(". ")[0][:70]            # first sentence names the memory
        vectors = list(self.embedder.embed(pieces))
        now, mid = time.time(), str(uuid.uuid4())
        points = []
        for piece, vec in zip(pieces, vectors):
            pid = str(uuid.uuid4())
            self._mem.setdefault(user_id, {})[pid] = Memory(pid, mid, piece, title, topic, source, now)
            points.append(PointStruct(id=pid, vector=vec.tolist(), payload={
                "user_id": user_id, "memory_id": mid, "text": piece, "title": title,
                "topic": topic, "source": source, "created_at": now}))
        self.client.upsert(COLLECTION, points=points)
        self._bm25.pop(user_id, None)                     # invalidate; rebuilt lazily on recall

    # ── read path ───────────────────────────────────────────────────────
    def profile(self, user_id: str) -> dict:
        if self.fs is None:
            return {}
        try:
            row = self.fs.get_online_features(
                features=PROFILE_FEATURES, entity_rows=[{"user_id": user_id}]).to_dict()
        except Exception:
            return {}
        return {k: v[0] for k, v in row.items() if k != "user_id" and v[0] is not None}

    def _keyword(self, user_id: str, query: str, depth: int) -> list[str]:
        if user_id not in self._bm25:
            ids = list(self._mem.get(user_id, {}))
            if not ids:
                return []
            corpus = [tokenize(self._mem[user_id][i].text) for i in ids]
            self._bm25[user_id] = (BM25Okapi(corpus), ids)
        bm25, ids = self._bm25[user_id]
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(ids)), key=lambda i: -scores[i])
        return [ids[i] for i in ranked[:depth] if scores[i] > 0]

    def _semantic(self, user_id: str, query: str, depth: int, topic: str | None = None) -> list[str]:
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if topic:
            must.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
        q = next(self.embedder.embed([query])).tolist()
        res = self.client.query_points(COLLECTION, query=q, query_filter=Filter(must=must), limit=depth)
        return [str(p.id) for p in res.points]

    def search(self, query: str, user_id: str, affinity: str | None = None) -> list[Memory]:
        depth = max(self.top_k * 5, 20)
        kw = self._keyword(user_id, expand(query), depth)
        rankers = [(kw, 1.0), (self._semantic(user_id, query, depth), 1.0)]
        # Personalisation ranker only for under-specified queries ("recommend gì?"):
        # if the query already carries strong lexical intent, the profile must not
        # drown it (Q5 "cloud security" must still surface security memories).
        if affinity and len(kw) < self.top_k:
            rankers.append((self._semantic(user_id, query, depth, topic=affinity), 0.5))
        if any(cue in fold(query) for cue in RECENCY_CUES):   # "gần đây" → recency ranker
            mems = self._mem.get(user_id, {})
            rankers.append((sorted(mems, key=lambda p: -mems[p].created_at)[:depth], 1.0))
        rrf: dict[str, float] = {}
        for ids, w in rankers:
            for rank, pid in enumerate(ids, start=1):     # rank is 1-based
                rrf[pid] = rrf.get(pid, 0.0) + w / (self.rrf_k + rank)
        mems = self._mem.get(user_id, {})
        out, seen = [], set()
        for pid, s in sorted(rrf.items(), key=lambda kv: -kv[1]):
            if mems[pid].memory_id in seen:               # one slot per memory, not per chunk
                continue
            seen.add(mems[pid].memory_id)
            out.append(Memory(**{**mems[pid].__dict__, "score": s}))
            if len(out) == self.top_k:
                break
        return out

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features → return assembled context."""
        prof = self.profile(user_id)
        hits = self.search(query, user_id, prof.get("topic_affinity"))
        log = self._activity.setdefault(user_id, [])
        now = time.time()
        if hits:                                          # in-process "stream" of recent topics
            log.append((now, hits[0].topic))
        recent = Counter(t for ts, t in log if now - ts < 3600).most_common(3)

        lines = [f"[user] {user_id}"]
        if prof:
            lines.append(f"[profile] ngôn ngữ={prof.get('preferred_language')}, "
                         f"đọc {prof.get('reading_speed_wpm')} wpm, quan tâm: {prof.get('topic_affinity')}")
            lines.append(f"[recent] {prof.get('queries_last_hour')} query/1h, "
                         f"{prof.get('distinct_topics_24h')} topic/24h (Feast)")
        else:
            lines.append("[profile] (Feast chưa sẵn sàng — chỉ dùng episodic memory)")
        lines.append("[session] topic hay gặp 1h qua: " + ", ".join(f"{t}×{n}" for t, n in recent))
        lines.append("[memories]")
        for i, h in enumerate(hits, 1):
            lines.append(f"  {i}. ({h.topic}/{h.source}, rrf={h.score:.4f}) {h.title}")
        if not hits:
            lines.append("  (không có memory liên quan)")
        return "\n".join(lines)
