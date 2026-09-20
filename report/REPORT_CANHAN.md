# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Cao Đức Hiệp  
**MSSV:** 2A2022602550  
**Nhóm:** Nhóm E-Commerce Policy (K4-L3B)  
**Ngày:** 20/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1) nghĩa là hai vector embedding chỉ về cùng một hướng trong không gian đa chiều, thể hiện hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa bất kể độ dài hay từ vựng bề mặt khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Khách hàng có 15 ngày để yêu cầu hoàn tiền sau khi nhận hàng."
- Câu B: "Người mua được phép gửi khiếu nại trả hàng trong thời hạn mười lăm ngày kể từ ngày giao thành công."
- Tại sao tương đồng: Dù từ vựng khác nhau ("khách hàng" vs "người mua", "15 ngày" vs "mười lăm ngày", "hoàn tiền" vs "khiếu nại trả hàng"), nhưng cả hai câu cùng mô tả chung một điều khoản chính sách đổi trả.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Đơn vị vận chuyển sẽ thu gom gói hàng tại nhà của người mua."
- Câu B: "Mạng nơ-ron tích chập (CNN) được sử dụng rộng rãi trong thị giác máy tính."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn độc lập (vận chuyển thương mại điện tử vs kiến trúc deep learning), các vector ngữ nghĩa gần như trực giao (orthogonal).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị ảnh hưởng bởi độ dài (magnitude) của vector (văn bản dài hay ngắn sẽ có độ dài vector khác nhau), trong khi độ tương tự cosine chỉ đo góc giữa các vector, giúp đánh giá chính xác độ tương đồng ngữ nghĩa mà không bị thiên vị bởi độ dài đoạn văn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> $\text{Số chunk} = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil = 23$  
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100: $\text{Số chunk} = \lceil \frac{10000 - 100}{500 - 100} \rceil = \lceil \frac{9900}{400} \rceil = 25$ chunks (tăng thêm 2 chunks).  
> Tăng overlap giúp bảo toàn trọn vẹn ngữ cảnh ở ranh giới giữa hai chunk kế tiếp, tránh hiện tượng một câu hoặc một điều khoản quan trọng bị cắt đôi khiến mô hình không truy xuất được thông tin đầy đủ.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regex `re.split(r"(?<=[.!?])(?:\s+|\n+)", text.strip())` với positive lookbehind để tách câu tại các dấu `. `, `! `, `? `, `.\n` mà vẫn giữ nguyên vẹn dấu câu ở cuối. Gom nhóm tối đa `max_sentences_per_chunk` câu lại thành một chunk, loại bỏ chuỗi rỗng và strip khoảng trắng thừa. Edge case chữ viết tắt (`v.v.`, `TS.`) và số thập phân được ghi nhận là hạn chế cần xử lý bằng thư viện tokenizer chuyên biệt khi mở rộng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo hai chiều: đệ quy xuống sâu (thử lần lượt các separator ưu tiên `["\n\n", "\n", ". ", " ", ""]` để tách ở ranh giới lớn trước) và gom lên (merge các mẩu nhỏ liền kề lại cho tới khi chạm ngưỡng `chunk_size` để tránh sinh ra các chunk vụn). Base case dừng khi đoạn văn có độ dài $\le \text{chunk\_size}$, hoặc khi danh sách separator còn lại bị rỗng thì cắt thẳng theo số ký tự.

**Chiến lược riêng: `SemanticChunker` (Markdown Section/Policy Chunking)** — hướng tiếp cận:
> Phân tách văn bản chính sách dựa theo các tiêu đề Markdown (`#`, `##`, `###`). Mỗi mục điều khoản hoặc câu hỏi chính sách tạo thành một chunk độc lập mang trọn vẹn một ý nghĩa hoàn chỉnh. Với các section dài vượt `max_chunk_size`, hạ cấp chia nhỏ đệ quy nhưng luôn gắn kèm tiêu đề mục vào đầu mỗi chunk con để bảo toàn ngữ cảnh xuất xứ.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ các records trong bộ nhớ (in-memory list) với cấu trúc chuẩn hoá gồm `id`, `content`, `metadata` (bảo đảm luôn có `doc_id`), và `embedding`. Khi `search()`, nhúng câu truy vấn và tính tích vô hướng (dot product) với toàn bộ vector trong store (do vector embedding đã được chuẩn hoá độ dài bằng 1 nên dot product trùng với cosine similarity), sau đó sắp xếp giảm dần và lấy top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` bắt buộc thực hiện tiền lọc (pre-filter) trước: lọc toàn bộ store theo các cặp key-value trong `metadata_filter` để tạo tập ứng viên hợp lệ, sau đó mới tính độ tương tự và xếp hạng top-k. `delete_document` lọc bỏ mọi record có `doc_id` hoặc `id` trùng khớp và trả về `True` nếu số lượng phần tử giảm đi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Kiểm tra an toàn: nếu store rỗng thì thông báo ngay mà không gọi LLM lãng phí. Truy xuất top-k chunks liên quan, đánh số thứ tự `[1]`, `[2]` kèm thông tin nguồn (`metadata['doc_id']`), đưa vào prompt có ràng buộc nghiêm ngặt: chỉ sử dụng các thông tin trong ngữ cảnh được cấp để trả lời và trích dẫn số thứ tự nguồn, không bịa thông tin bên ngoài.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: G:\AI_20K\CodeLab\K4-DAY07-CaoDucHiep-2A2022602550
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.09s ==============================
```

**Số lượng bài test vượt qua (pass):** **42** / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (`_mock_embed`) | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Người mua có 15 ngày để yêu cầu đổi trả." | "Khách hàng có thể gửi yêu cầu hoàn tiền trong vòng mười lăm ngày." | cao | -0.1262 | Không |
| 2 | "Chính sách hoàn tiền cho người mua hàng." | "Quy định bảo hành sản phẩm điện tử." | thấp | -0.0469 | Đúng |
| 3 | "Kiểm tra hàng trả về trong 2 ngày." | "Hạn xử lý khiếu nại của người bán là hai ngày sau nhận hàng." | cao | -0.0044 | Không |
| 4 | "Đơn vị vận chuyển lấy hàng tận nơi." | "Học máy và mô hình mạng nơ-ron nhân tạo." | thấp | -0.1303 | Đúng |
| 5 | "Phí vận chuyển hoàn hàng do người bán chịu." | "Người bán phải thanh toán chi phí ship hàng trả lại." | cao | +0.0926 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là ở Cặp 1 và Cặp 3, hai câu hoàn toàn cùng ngữ nghĩa nhưng điểm thực tế lại ra số âm (-0.1262). Lý do là vì bài test sử dụng `_mock_embed` (băm MD5 sinh số giả ngẫu nhiên) nên hoàn toàn không mã hóa được ngữ nghĩa ngôn ngữ thực tế. Điều này chứng minh rằng trong một hệ thống RAG sản xuất, bắt buộc phải dùng mô hình embedding ngôn ngữ thực thụ (như multilingual-MiniLM, OpenAI text-embedding-3, hoặc Gemini) để nắm bắt được quan hệ đồng nghĩa và ngữ cảnh ngữ nghĩa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** với chiến lược **Semantic Chunking** trên 3 tài liệu chính sách TikTok Shop (`tiktok-buyer-return-refund.md`, `tiktok-return-methods.md`, `tiktok-seller-return-refund.md`):

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người mua có bao nhiêu ngày để gửi yêu cầu trả hàng hoàn tiền sau khi nhận hàng? | `tiktok-buyer-return-refund#chunk_0` - Mục: `# Thời hạn gửi yêu cầu` (15 ngày dương lịch sau khi trạng thái đơn thành Đã giao hàng) | 0.1130 | Có (Khớp 100%) | Người mua có thể gửi yêu cầu trong vòng 15 ngày dương lịch sau khi nhận hàng. |
| 2 | Người bán có bao nhiêu ngày để xem xét và phản hồi yêu cầu trả hàng hoàn tiền của khách? | `tiktok-seller-return-refund#chunk_0` - Mục: `# Xem xét yêu cầu` (1 ngày dương lịch kể từ khi nhận yêu cầu) | 0.1715 | Có (Khớp 100%) | Người bán phải xem xét trong vòng 1 ngày dương lịch, nếu không sẽ tự động phê duyệt. |
| 3 | Có những phương thức trả lại gói hàng nào cho người mua trên TikTok Shop? | `tiktok-return-methods#chunk_0` - Mục: `# Phương thức trả hàng` (Bưu cục, lấy hàng tại nhà, tự gửi hàng) | 0.1608 | Có (Khớp 100%) | Có 3 phương thức: Gửi tại bưu cục, Lấy tại nhà và Tự gửi hàng. |
| 4 | Nếu người bán không chấp nhận nhận lại kiện hàng hoàn trả sau 3 lần giao thì xử lý ra sao? | `tiktok-return-methods#chunk_1` - Mục: `# Trách nhiệm của người bán` (Tiêu hủy sau 7 ngày từ lần giao đầu) | 0.1295 | Có (Khớp 100%) | Đơn vị vận chuyển sẽ ngừng liên lạc và kiện hàng bị tiêu hủy sau 7 ngày. |
| 5 | Sau khi nhận sản phẩm hoàn trả tại bưu cục hoặc tại nhà, người bán có mấy ngày để kiểm tra và từ chối? | `tiktok-seller-return-refund#chunk_2` - Mục: `# Kiểm tra hàng trả về` (2 ngày dương lịch sau khi nhận) | 0.1274 | Có (Khớp 100%) | Người bán có 2 ngày dương lịch sau khi nhận sản phẩm để kiểm tra và từ chối. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5** / 5 (100% queries đều trả về chunk chính xác)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Nhận ra rằng với các văn bản quy định chính sách thương mại điện tử, việc chia nhỏ theo cấu trúc ngữ nghĩa (Semantic/Section Chunking) vượt trội hơn hẳn so với chia nhỏ cố định (Fixed-Size). Khi kết hợp với việc giữ lại metadata `audience` (`buyer` vs `seller`) và lọc trước (`search_with_filter`), hệ thống hoàn toàn tránh được việc nhầm lẫn điều khoản giữa Người mua và Người bán.

---

## Tự Đánh Giá (Phần Cá Nhân)

- **Mã nguồn `src/`:** Đầy đủ, mạch lạc, vượt qua 42/42 kiểm thử tự động.
- **Chiến lược:** Thiết kế thành công `SemanticChunker` tích hợp với `bench.py` cho kết quả truy xuất chính xác.
- **Báo cáo:** Đầy đủ phép tính, phân tích kỹ thuật và số liệu benchmark thực tế.
