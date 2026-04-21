# Reflection - Chu Tuan Nghia

## 1. Engineering Contribution

Trong lab này, phần đóng góp chính của em tập trung vào evaluation pipeline cho RAG Agent, gồm bốn phần chính: tạo synthetic golden dataset, đánh giá retrieval, xây dựng multi-judge LLM, và điều phối benchmark/regression trong `main.py`.

### `data/synthetic_gen.py` - Synthetic Golden Dataset

Em xây dựng script sinh bộ test case tự động từ các file dữ liệu Markdown trong thư mục `data`. Phần này không chỉ tạo câu hỏi/câu trả lời đơn giản, mà còn tạo dataset có cấu trúc để phục vụ đánh giá retrieval và generation.

Các phần em đã làm:

- Viết hàm `chunk_markdown_document` để chia tài liệu Markdown theo heading trước, sau đó tiếp tục split bằng `RecursiveCharacterTextSplitter`. Cách này giúp chunk giữ được cấu trúc tài liệu tốt hơn so với việc cắt text thuần theo độ dài.
- Gắn metadata cho từng chunk gồm `source_doc_id`, `source_chunk_id`, và `source_file`. Đây là phần quan trọng vì các metric retrieval như Hit Rate và MRR cần biết chunk nào là ground truth.
- Thiết kế schema Pydantic `GeneratedQuestion` và `GeneratedQuestionSet` để ép LLM trả về dữ liệu có cấu trúc, giảm lỗi format khi ghi ra JSONL.
- Viết prompt sinh test case bằng tiếng Việt với nhiều loại câu hỏi: `fact-check`, `reasoning`, `multi-hop`, `edge-case`, `adversarial`, `out-of-context`, và `multi-turn`.
- Thêm logic xử lý riêng cho `out-of-context`: nếu câu hỏi nằm ngoài tài liệu thì `expected_retrieval_ids` để rỗng và expected answer bắt buộc là câu từ chối theo chuẩn.
- Kiểm tra `supporting_chunk_ids` do LLM sinh ra, chỉ giữ những chunk id thật sự tồn tại trong tài liệu để tránh golden set chứa id sai.
- Ghi kết quả ra `data/golden_set.jsonl`, mỗi dòng là một test case độc lập gồm `question`, `expected_answer`, `context`, `expected_retrieval_ids`, `source_doc_id`, `source_chunk_id`, và `metadata`.

Kết quả là pipeline có thể tạo được golden dataset đủ lớn để benchmark. Trong lần chạy hiện tại, báo cáo `summary.json` ghi nhận tổng cộng 59 test cases.

### `engine/retrieval_eval.py` - Retrieval Metrics và RAGAS

Em phụ trách phần đánh giá retrieval để tách lỗi retrieval khỏi lỗi generation. Nếu chỉ chấm câu trả lời cuối cùng thì rất khó biết agent sai do lấy nhầm context hay do model trả lời sai.

Các phần em đã làm:

- Xây dựng class `RetrievalEvaluator` để gom toàn bộ logic evaluation liên quan đến retrieval và RAGAS.
- Cài đặt `calculate_hit_rate`: kiểm tra trong top-k retrieved chunks có ít nhất một chunk thuộc ground truth hay không.
- Cài đặt `calculate_mrr`: tính reciprocal rank của chunk đúng đầu tiên trong danh sách retrieved chunks.
- Xử lý trường hợp `expected_ids` rỗng, đặc biệt cho các câu `out-of-context`, để metric không bị lỗi khi test case không yêu cầu retrieval chunk cụ thể.
- Viết hàm `score` trả về output thống nhất gồm:
  - `faithfulness`
  - `relevancy`
  - `retrieval.hit_rate`
  - `retrieval.mrr`
  - `expected_ids`
  - `retrieved_ids`
  - `is_applicable`
- Tích hợp RAGAS qua `Faithfulness` và `AnswerRelevancy` để đánh giá thêm chất lượng câu trả lời, không chỉ dừng ở retrieval.
- Viết `evaluate_batch` để tính trung bình Hit Rate, MRR, Faithfulness, và Relevancy trên nhiều record.

Trong kết quả benchmark hiện tại, hệ thống ghi nhận Hit Rate khoảng 96.61%, cho thấy phần lớn câu hỏi có retrieval lấy được chunk đúng trong top-k.

### `engine/llm_judge.py` - Multi-Judge Consensus

Em xây dựng module LLM-as-a-Judge để chấm điểm câu trả lời của agent so với ground truth. Mục tiêu là không phụ thuộc vào một judge duy nhất mà dùng nhiều model để tăng độ khách quan.

Các phần em đã làm:

- Tạo class `LLMJudge` với hai judge độc lập:
  - OpenAI model: `gpt-5-mini`
  - Gemini model: `gemini-3-flash-preview`
- Viết `evaluate_multi_judge` chạy hai judge song song bằng `asyncio.gather`, giúp giảm thời gian benchmark so với chạy tuần tự.
- Chuẩn hóa kết quả judge thành cùng một format gồm `score`, `passed`, `reasoning`, `issues`, và `error`.
- Thiết kế prompt chấm điểm theo thang 1-5, có tiêu chí rõ ràng:
  - Accuracy
  - Completeness
  - Faithfulness
  - Safety
- Yêu cầu judge chỉ trả về JSON hợp lệ để dễ parse tự động.
- Viết `_parse_json_response` để xử lý cả trường hợp model trả JSON trực tiếp và trường hợp model bọc JSON trong markdown code block.
- Viết `_normalize_score` để ép điểm về số thực và giới hạn trong khoảng 1.0 đến 5.0.
- Tính `final_score` bằng trung bình điểm các judge hợp lệ.
- Tính `agreement_rate` dựa trên độ lệch điểm giữa các judge.
- Thêm logic phát hiện conflict bằng `max_delta` và `conflict_threshold`. Nếu hai judge lệch nhau quá ngưỡng thì hệ thống đánh dấu `has_conflict = True`.
- Viết `_error_result` để nếu một judge lỗi thì pipeline vẫn trả kết quả có cấu trúc, không làm sập toàn bộ benchmark.
- Chuẩn bị placeholder `check_position_bias` cho hướng mở rộng kiểm tra position bias bằng pairwise judging.

Phần này đóng góp trực tiếp vào tiêu chí Multi-Judge consensus trong rubric vì có nhiều judge, có agreement rate, có conflict metadata, và có reasoning tổng hợp.

### `main.py` - Benchmark Orchestration và Regression Gate

Em viết phần điều phối benchmark end-to-end trong `main.py`, kết nối agent, retrieval evaluator, LLM judge, runner, và report output.

Các phần em đã làm:

- Viết `run_benchmark_with_results(agent_version)` để chạy benchmark cho một phiên bản agent cụ thể.
- Kiểm tra sự tồn tại của `data/golden_set.jsonl` trước khi chạy, tránh lỗi khi chưa sinh dataset.
- Load dataset từ JSONL bằng UTF-8 và bỏ qua dòng rỗng.
- Khởi tạo `BenchmarkRunner(MainAgent(), RetrievalEvaluator(), LLMJudge())` để gom ba thành phần chính vào một pipeline.
- Gọi `runner.run_all(dataset)` để chạy toàn bộ test cases.
- Tổng hợp summary gồm:
  - `metadata.version`
  - `metadata.total`
  - `metadata.timestamp`
  - `metrics.avg_score`
  - `metrics.hit_rate`
  - `metrics.agreement_rate`
- Viết `run_benchmark(version)` để tái sử dụng logic khi chỉ cần summary.
- Chạy hai lượt benchmark với `Agent_V1_Base` và `Agent_V2_Optimized` để phục vụ regression comparison.
- Tính `delta` giữa V2 và V1 theo `avg_score`.
- Ghi output ra:
  - `reports/summary.json`
  - `reports/benchmark_results.json`
- Cài đặt release gate đơn giản: nếu delta dương thì approve, ngược lại block release.

Phần này giúp biến các module riêng lẻ thành một evaluation factory hoàn chỉnh, có input dataset, agent execution, retrieval metrics, judge score, regression comparison, và file báo cáo để nộp.

## 2. Technical Depth

### MRR là gì?

MRR là viết tắt của Mean Reciprocal Rank. Metric này đo xem kết quả đúng đầu tiên xuất hiện ở vị trí thứ mấy trong danh sách retrieval.

Nếu chunk đúng nằm ở vị trí 1 thì reciprocal rank là `1/1 = 1.0`. Nếu nằm ở vị trí 2 thì là `1/2 = 0.5`. Nếu nằm ở vị trí 5 thì là `1/5 = 0.2`. Nếu không tìm thấy chunk đúng thì điểm là `0`.

MRR quan trọng trong RAG vì không chỉ cần lấy đúng tài liệu, mà còn cần tài liệu đúng xuất hiện càng sớm càng tốt. Chunk đúng nằm ở top 1 thường giúp LLM trả lời tốt hơn vì context quan trọng được ưu tiên.

### Cohen's Kappa là gì?

Cohen's Kappa là metric đo mức độ đồng thuận giữa hai người chấm hoặc hai judge, có tính đến khả năng họ đồng ý ngẫu nhiên. Nếu chỉ dùng accuracy agreement thông thường, ta có thể đánh giá quá cao mức đồng thuận vì đôi khi hai judge chọn giống nhau chỉ do may mắn.

Giá trị Kappa thường nằm trong khoảng từ -1 đến 1:

- Gần 1: đồng thuận rất cao.
- Gần 0: mức đồng thuận tương đương ngẫu nhiên.
- Nhỏ hơn 0: hai judge còn bất đồng nhiều hơn cả ngẫu nhiên.

Trong bài lab này, hệ thống hiện đang dùng `agreement_rate` dựa trên độ lệch điểm giữa hai judge. Nếu mở rộng tiếp, Cohen's Kappa có thể dùng để đo agreement chặt chẽ hơn khi biến điểm 1-5 thành các nhãn như fail/pass hoặc low/medium/high.

### Position Bias là gì?

Position Bias là hiện tượng judge thiên vị câu trả lời xuất hiện ở một vị trí nhất định, ví dụ luôn ưu tiên answer A hơn answer B chỉ vì A được đặt trước. Trong pairwise evaluation, nếu không kiểm soát position bias thì kết quả có thể không phản ánh chất lượng thật của câu trả lời.

Cách kiểm tra là chạy judge hai lần với thứ tự đảo ngược:

- Lần 1: Answer A trước, Answer B sau.
- Lần 2: Answer B trước, Answer A sau.

Nếu judge đổi lựa chọn chỉ vì thứ tự bị đảo thì có dấu hiệu position bias. Trong `llm_judge.py`, em đã để sẵn hàm `check_position_bias` như một điểm mở rộng cho pairwise judging.

### Trade-off giữa chi phí và chất lượng

Trong evaluation pipeline, chất lượng đánh giá càng cao thì thường chi phí càng lớn. Ví dụ, dùng nhiều judge model giúp kết quả khách quan hơn nhưng tốn nhiều API calls hơn. Dùng RAGAS faithfulness và relevancy cũng giúp phân tích sâu hơn, nhưng làm benchmark chậm hơn và tốn thêm token.

Một số trade-off chính:

- Tăng số lượng test cases giúp kết quả ổn định hơn nhưng tăng thời gian chạy.
- Tăng số judge giúp giảm thiên vị của một model nhưng tăng chi phí.
- Dùng model mạnh làm judge thường chấm tốt hơn nhưng đắt hơn.
- Chạy đầy đủ RAGAS cho mọi case cho kết quả chi tiết hơn nhưng không tối ưu nếu chỉ cần regression nhanh.

Cách cân bằng hợp lý là dùng batch async để giảm latency, dùng model nhỏ cho vòng regression thường xuyên, chỉ dùng model mạnh hoặc nhiều judge cho các case khó, case fail, hoặc trước khi release.

## 3. Problem Solving

Trong quá trình làm pipeline này, em gặp một số vấn đề kỹ thuật và xử lý theo hướng chia nhỏ lỗi theo từng tầng: dataset, retrieval, judge, và orchestration.

Vấn đề đầu tiên là golden dataset cần vừa có câu hỏi/câu trả lời, vừa có mapping về chunk ground truth. Nếu chỉ sinh question-answer thông thường thì không thể tính Hit Rate và MRR. Em giải quyết bằng cách gắn `source_chunk_id` cho từng chunk ngay từ bước chunking, sau đó yêu cầu LLM trả về `supporting_chunk_ids` và kiểm tra lại các id này trước khi ghi ra JSONL.

Vấn đề thứ hai là các câu `out-of-context` không có chunk đúng. Nếu xử lý giống câu hỏi bình thường thì retrieval metric sẽ bị sai hoặc bị chia cho dữ liệu không hợp lệ. Em tách riêng loại câu này bằng cách đặt `expected_retrieval_ids = []`, đồng thời trong evaluator xử lý `expected_ids` rỗng như một case đặc biệt.

Vấn đề thứ ba là output của LLM judge không phải lúc nào cũng sạch. Có trường hợp model trả JSON trong markdown block hoặc trả thêm text ngoài JSON. Em viết `_parse_json_response` để strip markdown và fallback bằng cách lấy đoạn từ dấu `{` đầu tiên đến dấu `}` cuối cùng. Cách này giúp benchmark ít bị gãy hơn khi judge không tuân thủ format tuyệt đối.

Vấn đề thứ tư là benchmark nhiều case dễ bị chậm và có nguy cơ rate limit. Em dùng async ở hai tầng: runner chạy test cases theo batch bằng `asyncio.gather`, còn `LLMJudge` chạy hai judge song song cho cùng một câu trả lời. Nhờ vậy pipeline có thể mở rộng lên nhiều test cases mà không phải chờ từng bước hoàn toàn tuần tự.

Vấn đề thứ năm là nếu một judge lỗi thì không nên làm hỏng toàn bộ evaluation. Em thêm `_error_result` để lỗi của từng model được ghi vào kết quả, còn các judge hợp lệ vẫn có thể được dùng để tính điểm cuối cùng.

Vấn đề cuối cùng là cần có quyết định release rõ ràng thay vì chỉ in metric rời rạc. Em thêm phần regression comparison trong `main.py`, chạy V1 và V2, tính delta điểm trung bình, ghi report ra JSON, sau đó approve hoặc block release dựa trên delta. Cách này giúp kết quả evaluation có thể dùng trực tiếp cho quy trình release gate.

Qua phần này, em hiểu rõ hơn rằng evaluation cho RAG không chỉ là gọi LLM chấm điểm. Một hệ thống evaluation tốt cần dataset có ground truth rõ, retrieval metrics riêng, judge có consensus, xử lý lỗi tốt, chạy được async, và có report đủ rõ để ra quyết định kỹ thuật.
