"""Bonus demo — 5 queries against HybridMemoryAgent.

Run from the repo root:   python bonus/demo.py
Needs data/corpus_vn.jsonl (make seed). Feast is bootstrapped here if NB4 was
never run, so the script also works on a clean checkout.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bonus"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")       # Windows console + tiếng Việt

import pandas as pd  # noqa: E402

from agent import FEAST_REPO, HybridMemoryAgent  # noqa: E402

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def ensure_feature_store() -> None:
    """Same synthetic data as NB4 → apply → materialize (skipped if already online)."""
    if (FEAST_REPO / "online_store.db").exists() and (FEAST_REPO / "registry.db").exists():
        return
    print("· bootstrapping Feast (NB4 chưa chạy)…")
    data = FEAST_REPO / "data"
    data.mkdir(exist_ok=True)
    users = [f"u_{i:03d}" for i in range(100)]
    frames = {"user_profile": pd.DataFrame({
        "user_id": users,
        "reading_speed_wpm": [180 + (i * 7) % 200 for i in range(100)],
        "preferred_language": ["vi" if i % 3 else "en" for i in range(100)],
        "topic_affinity": [["ai_ml", "cloud", "security", "database", "devops"][i % 5] for i in range(100)],
        "event_timestamp": [NOW - timedelta(hours=i % 48) for i in range(100)],
    }), "query_velocity": pd.DataFrame({
        "user_id": users,
        "queries_last_hour": [(i * 11) % 50 for i in range(100)],
        "distinct_topics_24h": [1 + (i * 3) % 10 for i in range(100)],
        "event_timestamp": [NOW - timedelta(minutes=i % 30) for i in range(100)],
    }), "item_popularity": pd.DataFrame({
        "doc_id": [f"item_{i:04d}" for i in range(10)], "click_count_24h": [0] * 10,
        "ctr_7d": [0.0] * 10, "avg_dwell_seconds": [0.0] * 10, "event_timestamp": [NOW] * 10,
    })}
    for name, df in frames.items():                # never overwrite NB4's own data
        if not (data / f"{name}.parquet").exists():
            df.to_parquet(data / f"{name}.parquet")

    spec = importlib.util.spec_from_file_location("feature_views", FEAST_REPO / "feature_views.py")
    fv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fv)
    from feast import FeatureStore
    fs = FeatureStore(repo_path=str(FEAST_REPO))
    fs.apply([fv.user, fv.item, fv.user_profile_features,
              fv.item_popularity_features, fv.query_velocity_features])
    fs.materialize_incremental(end_date=NOW)


def seed(agent: HybridMemoryAgent) -> None:
    docs = [json.loads(line) for line in (ROOT / "data" / "corpus_vn.jsonl").open(encoding="utf-8")]
    by_topic: dict[str, list[dict]] = {}
    for d in docs:
        by_topic.setdefault(d["topic"], []).append(d)
    # u_001 (topic_affinity = cloud): "documents the user has read"
    reads = [d for d in by_topic["cloud"] if "Kubernetes" in d["text"]][:3]
    reads += [d for d in by_topic["cloud"] if "tự động mở rộng" in d["title"]][:3]
    reads += by_topic["security"][:4] + by_topic["devops"][:3] + by_topic["database"][:2]
    for d in reads:
        agent.remember(f"{d['title']}. {d['text']}", "u_001", topic=d["topic"], source="doc")
    # u_001 personal notes — code-switching and typed without dấu
    agent.remember("Note: cần review lại IAM policy cho cluster prod, nhớ bật MFA cho account admin. "
                   "Security audit deadline thứ 6.", "u_001", topic="security", source="note")
    agent.remember("chat: hoi ve cach giam chi phi cloud bang spot instance cho batch job", "u_001",
                   topic="cloud", source="chat")
    # u_002 — a private memory that u_001 must never see
    agent.remember("Ghi chú riêng của u_002: hợp đồng bảo mật với khách hàng ACME, mã dự án X-77.",
                   "u_002", topic="security", source="note")


def push_velocity(agent: HybridMemoryAgent, queries: int, topics: int) -> None:
    agent.fs.write_to_online_store("query_velocity_features", pd.DataFrame({
        "user_id": ["u_001"], "queries_last_hour": [queries], "distinct_topics_24h": [topics],
        "event_timestamp": [datetime.now(timezone.utc)],
    }))


QUERIES = [
    ("1. vector hit đơn giản", "Tôi đã đọc gì về Kubernetes?"),
    ("2. cần profile (topic_affinity)", "Recommend đọc gì tiếp"),
    ("3. cần fresh activity", "Tôi đang quan tâm gì gần đây?"),
    ("4. paraphrase (vector thắng)", "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5. mixed (hybrid + profile)", "Cho tôi summary cloud security"),
]


def main() -> int:
    ensure_feature_store()
    agent = HybridMemoryAgent()
    seed(agent)
    n = sum(len(m) for m in agent._mem.values())
    print(f"· {n} memory chunks for {len(agent._mem)} users; Feast={'on' if agent.fs else 'off'}\n")

    original = agent.profile("u_001")
    for label, q in QUERIES:
        if label.startswith("3") and original:
            # Streaming freshness: push fresh activity straight to the online store
            # (what a Kafka → Push API consumer would do), then read it back at once.
            push_velocity(agent, original["queries_last_hour"] + 3, original["distinct_topics_24h"])
            print(f"   (push: queries_last_hour {original['queries_last_hour']} → "
                  f"{agent.profile('u_001')['queries_last_hour']}, visible on the next read)")
        print(f"── Q{label}: {q!r}")
        print(agent.recall(q, "u_001"))
        print()
    if original:   # leave the shared NB4 online store as we found it
        push_velocity(agent, original["queries_last_hour"], original["distinct_topics_24h"])

    # Privacy isolation: u_001 asks about u_002's secret, must get nothing from u_002.
    leaked = "X-77" in agent.recall("hợp đồng ACME mã dự án", "u_001")
    print(f"isolation check: u_001 thấy memory của u_002? {'LEAK' if leaked else 'không (OK)'}")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
