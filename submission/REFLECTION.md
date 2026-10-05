# Reflection — Lab 19

**Tên:** Lâm Hoàng Phúc
**Cohort:** A20-K4
**Path đã chạy:** lite (fastembed `bge-small-en-v1.5` + Qdrant in-memory + Feast SQLite, Windows 10 / Python 3.12)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 trung bình: hybrid **78,6%** > BM25 77,8% > vector 73,2%.

- **exact** (15): BM25 = hybrid = 96,7%, vector 88,7%. Query chứa đúng thuật ngữ trong doc nên tín hiệu từ vựng đã đủ; RRF không thêm được gì.
- **paraphrase** (15): BM25 33,3% nhỉnh hơn vector 24,0%, hybrid 32,0%. Bất ngờ là vector thua: `bge-small-en` học chủ yếu tiếng Anh nên biểu diễn câu tiếng Việt diễn đạt lại kém. Đây là vấn đề chọn model, không phải giới hạn của dense retrieval.
- **mixed** (20): hybrid **100%** > vector 98,5% > BM25 97%. Hai retriever sai ở những chỗ khác nhau, RRF thưởng cho doc xếp hạng cao ở cả hai danh sách.

Không dùng hybrid khi: tra mã/ID/tên lỗi chính xác, hoặc ngân sách latency rất chặt (BM25 P99 3,7ms so với hybrid 29,7ms) → **pure BM25**. Query đa ngữ/diễn đạt tự do với một embedding model mạnh (bge-m3), corpus ít thuật ngữ riêng → **pure vector**.

---

## Điều ngạc nhiên nhất khi làm lab này

Vector search lại thua BM25 trên câu paraphrase tiếng Việt. Embedding model là một quyết định kiến trúc, phải đo trên ngôn ngữ thật của mình.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`): `ARCHITECTURE.md` + `agent.py` (`HybridMemoryAgent.remember()/.recall()`) + `demo.py` (5 query, exit 0)
- [ ] Pair work: làm một mình
