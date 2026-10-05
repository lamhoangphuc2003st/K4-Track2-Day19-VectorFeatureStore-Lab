# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB8 — Feature Engineering: sáu họ feature và hai cách rò rỉ
#
# **Stack:** `app.features` (pandas + numpy) + Feast on-demand feature view.
# Maps to deck §7 "Feature Engineering: 6 Họ Feature" + "On-Demand Feature".
#
# > NB4 dạy *feature **store*** — nơi cất và phục vụ feature. Notebook này dạy
# > *feature **engineering*** — nghĩa là bản thân việc nghĩ ra feature. Đây là
# > phần quyết định chất lượng model, và cũng là phần dễ tự bắn vào chân nhất.
#
# Sáu họ:
#
# | # | Họ | Ví dụ |
# |---|---|---|
# | 1 | Aggregation theo cửa sổ | `searches_1h`, `searches_7d` |
# | 2 | Tỷ lệ / chuẩn hoá | `query_len / avg của chính user` |
# | 3 | Lag & delta | `prev_query_len`, `query_len_delta` |
# | 4 | Recency | `seconds_since_last` |
# | 5 | Mã hoá categorical | frequency / target encoding |
# | 6 | Embedding làm feature | vector user/item |

# %%
import _setup  # noqa: F401
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

from app.features import (auc, frequency_encode, generate_events, latest_join,
                          leakage_experiment, leaked_row_fraction, pit_join,
                          window_aggregates)

ROOT = Path(_setup.__file__).resolve().parent.parent

# %% [markdown]
# ## 1. Event log
#
# Log tìm kiếm tổng hợp: ai tìm gì, lúc nào, có click không. `clicked` là nhãn.
# Người dùng gắn bó (`engagement` cao) vừa **click nhiều hơn** vừa **tìm nhiều
# hơn** — nên feature đếm hoạt động mang tín hiệu *thật*, và ta có thể so sánh
# nó với tín hiệu *giả* do rò rỉ tạo ra.

# %%
events = generate_events(n_users=200, n_days=30, seed=42)
print(f"events   : {len(events)}")
print(f"users    : {events.user_id.nunique()}")
print(f"sessions : {events.session_id.nunique()}  (~{len(events)/events.session_id.nunique():.1f} event/session)")
print(f"click rate: {events.clicked.mean():.3f}")
events.head(3)

# %% [markdown]
# ## 2. Bốn họ đầu — tính **nhân quả**
#
# Mọi feature ở đây chỉ nhìn vào *quá khứ của chính dòng đó*: cửa sổ đếm dùng
# `<` chứ không `<=`, trung bình dùng `.shift()`. Bỏ `shift()` là rò rỉ ngay,
# và không có gì báo cho bạn biết.

# %%
feat = window_aggregates(events)
cols = ["searches_1h", "searches_24h", "searches_7d",
        "query_len_vs_user_avg", "query_len_delta", "seconds_since_last"]
print(feat[cols].describe().loc[["mean", "50%", "max"]].round(2).to_string())

# %% [markdown]
# ## 3. Feature trung thực trông như thế nào?
#
# Chia train/holdout rồi đo AUC của **chính feature đó** (không train model —
# nếu feature có tín hiệu, thứ hạng của nó đã tách được nhãn). Feature trung
# thực cho AUC **gần bằng nhau** ở hai bên.

# %%
rng = np.random.default_rng(0)
mask = rng.random(len(feat)) < 0.7
print(f"{'feature':<26}{'train':>8}{'holdout':>9}{'gap':>8}")
for c in ["searches_24h", "searches_7d", "seconds_since_last"]:
    tr = auc(feat.loc[mask, c], feat.loc[mask, "clicked"])
    ho = auc(feat.loc[~mask, c], feat.loc[~mask, "clicked"])
    print(f"{c:<26}{tr:8.3f}{ho:9.3f}{tr-ho:+8.3f}")

# %% [markdown]
# ## 4. Rò rỉ #1 — target encoding
#
# **Giao thức đúng: chia trước, mã hoá sau.** Sai lầm phổ biến khi *minh hoạ*
# rò rỉ là mã hoá cả bảng rồi mới chia — lúc đó nhãn holdout đã nằm sẵn trong
# encoding, hai bên đều đẹp, và cái rò rỉ **biến mất khỏi phép đo**.
#
# Đọc cột `gap`: lớn = feature đang *thuộc lòng*, không phải *học*.

# %%
print("── key = session_id (cardinality rất cao, ~1 event/nhóm) ──")
print(leakage_experiment(events, "session_id").round(3).to_string(index=False))
print("\n── key = user_id (cardinality thấp hơn, ~45 event/nhóm) ──")
print(leakage_experiment(events, "user_id").round(3).to_string(index=False))

# %% [markdown]
# `target-naive` trên `session_id` cho **train AUC ≈ 0.99** và **test AUC ≈ 0.52**.
# Nhóm chỉ có ~1 dòng, nên "trung bình nhãn của nhóm" *chính là nhãn của dòng đó*.
# Model học thuộc đáp án.
#
# So sánh với `user_id`: cùng công thức, nhưng nhóm lớn hơn nên phần đóng góp của
# chính dòng đó bị pha loãng → gap nhỏ hơn nhiều. **Mức độ rò rỉ tỉ lệ với
# cardinality.** Đó là lý do target encoding trên `session_id`, `device_id`,
# `zip+product` là bẫy kinh điển.
#
# `target-in-fold` cho train ≈ test — đúng như một feature trung thực phải thế.

# %% [markdown]
# ## 5. Rò rỉ #2 — join "giá trị mới nhất" thay vì point-in-time
#
# Đây là bản `GROUP BY user_id + MAX(timestamp)` mà ai cũng từng viết. Nó lặng lẽ
# kéo giá trị được ghi **sau** thời điểm của nhãn.

# %%
fe = events[["user_id", "event_timestamp"]].copy().sort_values("event_timestamp")
fe["feature_value"] = fe.groupby("user_id").cumcount() + 1     # hoạt động tích luỹ

rng = np.random.default_rng(1)
ent = (events.loc[rng.random(len(events)) < 0.4,
                  ["user_id", "event_timestamp", "clicked"]]
       .sort_values("event_timestamp").reset_index(drop=True))

lat, pit = latest_join(ent, fe), pit_join(ent, fe)
auc_lat = auc(lat["feature_value"], lat["clicked"])
auc_pit = auc(pit["feature_value"], pit["clicked"])

print(f"training rows                    : {len(ent)}")
print(f"dòng bị rò (giá trị ghi SAU nhãn): {leaked_row_fraction(ent, fe):.1%}")
print(f"\nAUC với latest-value join        : {auc_lat:.3f}   ← dùng tương lai")
print(f"AUC với point-in-time join       : {auc_pit:.3f}   ← phục vụ được thật")
print(f"\n'lift ảo' sẽ mất khi lên production: {auc_lat - auc_pit:+.3f} AUC")

# %% [markdown]
# Offline báo cáo một con số, production trả lại con số thấp hơn hẳn — và không
# ai sửa được vì *code không hề sai*, chỉ là join sai ngữ nghĩa thời gian.
# `get_historical_features()` của Feast (NB4) làm PIT join cho bạn; đây là lý do
# nó tồn tại.

# %% [markdown]
# ## 6. On-demand feature: công thức không thể materialize
#
# `amount_vs_avg = amount / avg_amount_7d` là feature fraud mạnh nhất — nhưng
# `amount` **chưa tồn tại** lúc materialize. On-demand feature view ghép
# *feature đã lưu* với *dữ liệu của request* và áp **cùng một công thức** cho cả
# training lẫn serving.
#
# > **Gotcha feast 0.65:** `from feast import on_demand_feature_view` trả về
# > **module**, không phải decorator (docs vẫn ghi import ngắn). Phải dùng
# > `from feast.on_demand_feature_view import on_demand_feature_view`.

# %%
repo = ROOT / "app" / "feast_repo_ondemand"
subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_spend.py")], check=True,
               capture_output=True)
subprocess.run(["feast", "apply"], cwd=repo, check=True, capture_output=True)
subprocess.run(["feast", "materialize-incremental", "2027-01-01T00:00:00"],
               cwd=repo, check=True, capture_output=True)
print("feast apply + materialize OK")

# %%
from feast import FeatureStore  # noqa: E402

fs = FeatureStore(repo_path=str(repo))
out = fs.get_online_features(
    features=["user_spend_stats:avg_amount_7d",
              "amount_vs_avg:amount_vs_avg", "amount_vs_avg:is_spike"],
    entity_rows=[
        {"user_id": "u_000", "amount": 100_000.0},      # giao dịch nhỏ
        {"user_id": "u_000", "amount": 15_000_000.0},   # CÙNG user, bất thường
        {"user_id": "u_001", "amount": 250_000.0},
    ],
).to_dict()

for i in range(3):
    print(f"user={out['user_id'][i]}  avg7d={out['avg_amount_7d'][i]:>12,.0f}  "
          f"ratio={out['amount_vs_avg'][i]:6.2f}  spike={out['is_spike'][i]}")

# %% [markdown]
# Hai dòng đầu là **cùng một user, cùng một feature đã lưu** — chỉ khác `amount`
# của request. Không có cách nào pre-compute chuyện đó.

# %% [markdown]
# ## Deliverable evidence
#
# 1. §3: bảng AUC train/holdout của feature trung thực — gap nhỏ.
# 2. §4: bảng leakage cho `session_id` (gap lớn) và `user_id` (gap nhỏ hơn).
# 3. §5: % dòng bị rò + chênh lệch AUC giữa latest-join và PIT join.
# 4. §6: `amount_vs_avg` khác nhau cho cùng một user với hai `amount` khác nhau.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** groupby/rolling boilerplate, `describe()`, bảng in ra,
# code sinh event log.
#
# **Think hard yourself:** *thứ tự chia dữ liệu và mã hoá*. Nếu bạn nhờ AI
# "demo target encoding leakage", nó gần như luôn viết: mã hoá cả DataFrame →
# `train_test_split` → so sánh. Kết quả là **không thấy rò rỉ nào** (bạn vừa
# chứng minh điều đó ở §4 nếu làm ngược lại), và bạn sẽ kết luận sai rằng target
# encoding an toàn. Cùng một lỗi thứ-tự đó, khi xảy ra trong pipeline thật, tạo
# ra model 0.99 AUC offline và vô dụng online. Tự viết thứ tự: **split → fit
# encoder trên train → transform cả hai**.
