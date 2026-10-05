# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB7 — Semantic Cache: rẻ nhất, và nguy hiểm nhất
#
# **Stack:** `app.cache.SemanticCache` (Qdrant collection riêng). Maps to deck §6
# "Semantic Cache" + slide Bảo mật.
#
# > Semantic cache là **đòn bẩy chi phí rẻ nhất** trong cả stack RAG: AWS đo trên
# > 63.796 query thật thấy giảm tới **86% chi phí inference** và **88% latency**
# > ở ngưỡng 0,75. Nó cũng là thứ dễ biến thành **sự cố bảo mật** nhất trong lab
# > hôm nay. Notebook này làm cả hai chuyện đó hiện ra bằng số.
#
# Ba tham số, ba loại lỗi *khác nhau*:
#
# | Tham số | Đặt sai | Hậu quả |
# |---|---|---|
# | `threshold` | quá thấp | **false hit** — trả lời của câu hỏi khác |
# | `ttl_s` | thiếu | **stale hit** — câu trả lời cũ sống mãi |
# | `namespaced` | thiếu | **rò chéo tenant** — sự cố bảo mật |

# %%
import _setup  # noqa: F401
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from app.cache import SemanticCache

DATA = Path(_setup.__file__).resolve().parent.parent / "data"
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(":memory:")

# %% [markdown]
# ## 1. Hit và miss cơ bản

# %%
cache = SemanticCache(client=client, embedder=embedder, threshold=0.75, ttl_s=3600)
cache.put("acme", "làm sao tối ưu chi phí cloud", "Dùng spot instance và autoscaling.")

for probe in ["làm sao tối ưu chi phí cloud",        # y hệt
              "cách giảm chi phí hạ tầng đám mây",   # diễn đạt khác
              "cách kiểm thử unit test"]:            # chủ đề khác hẳn
    hit = cache.get("acme", probe)
    print(f"{probe:<38} → {'HIT  ' + f'{hit.score:.3f}' if hit else 'MISS'}")

# %% [markdown]
# ## 2. Sweep ngưỡng: tiết kiệm mua bằng câu trả lời sai
#
# Một cache phải làm **hai** việc, và chúng kéo ngược nhau:
#
# * **HIT** khi câu hỏi mới chỉ là cách diễn đạt khác của câu đã hỏi → tiết kiệm
# * **MISS** khi câu hỏi thật sự khác → tránh trả lời sai
#
# Nên bộ probe có **hai nửa**: biến thể của câu **đã** nằm trong cache (đáng lẽ
# HIT) và biến thể của câu **chưa** nằm trong cache (đáng lẽ MISS). Chỉ đo hit
# rate mà không đo nửa còn lại là cách tự lừa mình.
#
# `peek()` trả điểm thô nên toàn bộ bảng dưới đây chỉ tốn **một lượt embedding**.

# %%
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]
warm, cold = golden[::2], golden[1::2]

sweep = SemanticCache(client=client, embedder=embedder, threshold=0.0, ttl_s=None)
for g in warm:
    sweep.put("acme", g["query"], f"ANSWER::{g['query_id']}")


def variants(q: str) -> list[str]:
    """Cách người thật hỏi lại cùng một câu."""
    w = q.split()
    return [f"cho tôi hỏi {q}",
            f"{q} thì làm thế nào",
            " ".join(w[:-1]) if len(w) > 2 else q]


positives = []   # nguồn CÓ trong cache -> hit là đúng
for g in warm:
    for v in variants(g["query"]):
        p = sweep.peek("acme", v)
        if p:
            positives.append((p[0], p[1]["question"] == g["query"]))

negatives = []   # nguồn KHÔNG có trong cache -> mọi hit đều là trả lời sai
for g in cold:
    for v in variants(g["query"]):
        p = sweep.peek("acme", v)
        if p:
            negatives.append(p[0])

print(f"cache: {len(warm)} câu   probe: {len(positives)} positive / {len(negatives)} negative\n")
print(f"{'ngưỡng':>8}{'tiết kiệm':>12}{'trả lời sai':>14}   {'':<4}")
for th in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
    saved = sum(1 for sc, ok in positives if sc >= th and ok) / len(positives)
    wrong = sum(1 for sc in negatives if sc >= th) / len(negatives)
    flag = "NGUY HIỂM" if wrong > 0.20 else ("quá chặt" if saved < 0.80 else "cân bằng")
    print(f"{th:>8.2f}{saved:>12.0%}{wrong:>14.0%}   {flag}")

# %% [markdown]
# **Đọc bảng này thật kỹ.** Ngưỡng 0,75 — con số AWS công bố — trên corpus *này*
# vẫn để lọt một tỉ lệ đáng kể câu trả lời sai. Phải lên ~0,85 mới vừa giữ được
# gần như toàn bộ phần tiết kiệm vừa đưa tỉ lệ sai về 0. Lên 0,95 thì an toàn
# nhưng mất một nửa phần tiết kiệm.
#
# > Không có ngưỡng đúng phổ quát. 0,75 là **điểm bắt đầu để đo**, không phải
# > hằng số để copy. Phân bố query của bạn quyết định con số cuối cùng.

# %% [markdown]
# ### Lựa chọn ngưỡng của tôi: **0,85**
#
# * Ở 0,85 bảng sweep cho **tiết kiệm 100% / trả lời sai 0%** — điểm thấp nhất mà
#   tỉ lệ sai về 0, nên giữ được toàn bộ phần tiết kiệm.
# * **Vì sao 0,75 chưa đủ:** corpus này gồm câu tiếng Việt kỹ thuật rất giống nhau
#   về khuôn ("cách … chi phí …", "làm sao … hạ tầng …"); với `bge-small-en`
#   (English-trained) các câu khác nghĩa vẫn có cosine 0,75–0,80. Ở 0,75 có **36%**
#   probe âm bị trả nhầm câu trả lời của câu khác — với cache, một false hit tệ hơn
#   nhiều một miss (miss chỉ tốn thêm một LLM call; false hit trả lời sai một cách
#   tự tin).
# * 0,80 vẫn còn 5% sai; 0,90 bắt đầu mất tiết kiệm (96%); 0,95 mất gần nửa (53%).
#   Vì chi phí bất đối xứng, tôi chọn 0,85 thay vì 0,80, và sẽ đo lại khi đổi
#   embedding model (bge-m3 sẽ dịch toàn bộ phân bố điểm).

# %% [markdown]
# ## 3. TTL: câu trả lời cũ không tự biết mình cũ
#
# `SemanticCache` dùng **đồng hồ ảo** (`advance()`) nên ta test được TTL mà không
# phải `sleep()` một giờ trong notebook.

# %%
ttl_cache = SemanticCache(client=client, embedder=embedder, threshold=0.75, ttl_s=1800)
ttl_cache.put("acme", "giá GPU hiện tại là bao nhiêu", "Khoảng $2/giờ cho A100.")

for jump in (0, 600, 3600):
    ttl_cache.advance(jump)
    hit = ttl_cache.get("acme", "giá GPU hiện tại là bao nhiêu")
    print(f"t = {ttl_cache._clock:>6.0f}s  → {'HIT' if hit else 'MISS (hết hạn)'}")

print(f"\nstale evictions: {ttl_cache.stats.stale_evictions}")

# %% [markdown]
# Câu hỏi nhạy thời gian ("giá hiện tại", "còn hàng không", "trạng thái đơn hàng")
# **phải** có TTL ngắn — hoặc không được cache.

# %% [markdown]
# ## 4. Rò chéo tenant — đây là lỗ hổng bảo mật, không phải bug cache
#
# Ba khách hàng dùng chung một index. `namespaced=False` mô phỏng đúng lỗi hay
# gặp nhất: quên filter theo tenant.

# %%
leaky = SemanticCache(client=client, embedder=embedder, threshold=0.70,
                      ttl_s=None, namespaced=False)
leaky.put("acme", "doanh thu quý 3 của chúng tôi",
          "Doanh thu ACME quý 3: 4,2 tỷ VND.")   # dữ liệu của ACME

stolen = leaky.get("globex", "doanh thu quý 3 của chúng tôi")
print("namespaced=False → GLOBEX nhận được:")
print("   ", stolen.answer if stolen else "(miss)")
print("    chủ sở hữu thật:", stolen.tenant if stolen else "-")

safe = SemanticCache(client=client, embedder=embedder, threshold=0.70,
                     ttl_s=None, namespaced=True)
safe.put("acme", "doanh thu quý 3 của chúng tôi", "Doanh thu ACME quý 3: 4,2 tỷ VND.")
blocked = safe.get("globex", "doanh thu quý 3 của chúng tôi")
print("\nnamespaced=True  → GLOBEX nhận được:", blocked.answer if blocked else "MISS (đúng)")

# %% [markdown]
# Không có exception, không có stack trace, không có dòng log đỏ. Chỉ là một
# khách hàng đọc được dữ liệu của khách hàng khác — và hệ thống coi đó là
# *cache hit thành công*.
#
# > Đây chính là **OWASP LLM08:2025** trong 15 dòng code: tách tenant bằng
# > metadata filter là isolation **mềm**; quên một filter là rò toàn bộ.
#
# **Và:** đổi embedding model ⇒ phải **xoá sạch cache**. Vector cũ và vector mới
# không cùng một không gian, nên điểm similarity giữa chúng vô nghĩa.

# %% [markdown]
# ## Deliverable evidence
#
# 1. §2: bảng sweep ngưỡng với hit rate và false-hit rate.
# 2. §3: TTL — HIT ở t=600s, MISS ở t=3600s, `stale_evictions ≥ 1`.
# 3. §4: output cho thấy GLOBEX đọc được câu trả lời của ACME khi `namespaced=False`,
#    và MISS khi `namespaced=True`.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** wrapper put/get, vòng sweep, bảng kết quả, đồng hồ ảo.
#
# **Think hard yourself:** *định nghĩa thế nào là "false hit"*. Nếu bạn chỉ hỏi
# AI "đo hit rate của semantic cache", bạn sẽ nhận được một con số đẹp và **không
# có** cột false-hit — vì đo hit rate thì dễ, đo *hit sai* thì cần một định nghĩa
# ground truth mà chỉ bạn mới có (ở đây là `topic`). Một cache 95% hit rate nghe
# tuyệt vời cho tới khi bạn biết một phần ba số hit đó là câu trả lời của câu hỏi
# khác. Luôn báo cáo hai cột cạnh nhau.
