# Huấn luyện MTP draft cho AREX-2 trên MI50

Repo này huấn luyện một khối Multi-Token Prediction (MTP) để đề xuất token cho `BAAI/AREX-2` khi chạy speculative decoding trong llama.cpp. Pipeline hiện hành trích hidden state và continuation từ AREX-2, huấn luyện khối MTP bằng cross-entropy, rồi xuất sidecar GGUF. Model AREX-2 xác minh các token được đề xuất khi inference.

## Chạy nhanh

Chạy lệnh sau từ thư mục gốc repo.

```bash
set -o pipefail
mkdir -p output
bash scripts/run_pipeline.sh 2>&1 | tee output/arex-mtp-train.log
```

Pipeline dùng cache teacher hiện có nếu file cache có dữ liệu. Hãy kiểm tra cache và manifest thuộc cùng lượt trích, đồng thời khớp model đích trong config. Một lượt chạy ghi checkpoint và GGUF vào các tên cố định trong `output/`. Lưu bản cần giữ sang đường dẫn khác trước khi chạy lại.

### Môi trường đã kiểm chứng

- AMD Radeon Instinct MI50, kiến trúc gfx906, khoảng 32 GiB VRAM. Trên máy đã thử, lượt huấn luyện và lượt chạy teacher dùng GPU lần lượt.
- ROCm 6.0 và PyTorch 2.4.1 bản ROCm. PyTorch truy cập MI50 qua `cuda:0`.
- Python có `torch`, `numpy` và `PyYAML`. Module `gguf` đến từ `gguf-py` của checkout llama.cpp.
- Bash, compiler hỗ trợ C++17, cùng source và binary llama.cpp tương thích với API `llama_set_embeddings_nextn`, `llama_get_embeddings_nextn` và chế độ `draft-mtp`. Dùng cùng checkout cho header và thư viện.
- Cần `llama-quantize` để xuất Q4_0 và `llama-server`, `curl` để chạy ví dụ benchmark. Pilot tạo khoảng 60 MiB cache, 1,6 GiB checkpoint mới và 4,7 GiB cho hai bản GGUF. Dành thêm dung lượng cho file tạm và các artifact đã có.

```bash
export LLAMA_CPP_DIR="$(realpath ../llama.cpp)"
export MTP_PYTHON=python3
export PYTHONPATH="$LLAMA_CPP_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR/build/bin${ROCM_PATH:+:$ROCM_PATH/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

rocm-smi --showproductname --showmeminfo vram
"$MTP_PYTHON" -c 'import torch; print(torch.__version__, torch.version.hip); print(torch.ones(1, device="cuda"))'
"$LLAMA_CPP_DIR/build/bin/llama-server" --list-devices
```

Kích hoạt môi trường Python ROCm trước khi chạy. Nếu llama.cpp nằm ở vị trí khác, đặt `LLAMA_CPP_DIR` theo checkout của bạn. Script tự thêm `gguf-py` vào `PYTHONPATH`. Khi gọi module Python riêng, giữ cấu hình `PYTHONPATH` như ví dụ hoặc cài package `gguf` từ checkout tương ứng.

## Cấu hình và artifact

[Config YAML](configs/training_config.yaml) cung cấp ba đường dẫn cho pipeline đang dùng.

| Khóa | Vai trò |
|---|---|
| `model.target_gguf_path` | AREX-2 Q5_K_M để sinh trace, lấy embedding và LM head, và chạy xác minh |
| `model.base_mtp_gguf_path` | MTP Qwen3.8-27B Q4_0 làm warm-start cho 15 tensor và cung cấp metadata kiến trúc sidecar |
| `model.raw_data_path` | JSONL seed, mỗi dòng có trường `prompt` |

Mặc định, đặt hai file GGUF nguồn trong `models/` theo tên trong YAML. Thư mục này được Git bỏ qua. Nếu các file nằm nơi khác, đặt `MTP_TEACHER_GGUF` và `MTP_WARM_START_GGUF` trong môi trường của bạn. [Teacher GGUF](https://huggingface.co/bartowski/BAAI_AREX-2-GGUF) · [Qwen3.8-27B nguồn](https://huggingface.co/Qwen/Qwen3.8-27B)

Các mục `training`, `architecture` và `export` trong YAML mô tả cấu hình cũ. Entry point hiện tại là [run_pipeline.sh](scripts/run_pipeline.sh), với các tùy chọn trong bảng.

| Biến môi trường | Mặc định | Tác dụng |
|---|---|---|
| `LLAMA_CPP_DIR` | `../llama.cpp` | Checkout chứa header và `build/bin` |
| `MTP_PYTHON` | `python3` | Môi trường Python dùng cho train và export |
| `ROCM_PATH` | Theo môi trường | Thêm thư mục `lib` của ROCm vào đường dẫn thư viện khi có giá trị |
| `MTP_TEACHER_GGUF` | `model.target_gguf_path` | Đường dẫn teacher do người chạy cung cấp |
| `MTP_WARM_START_GGUF` | `model.base_mtp_gguf_path` | Đường dẫn GGUF warm-start do người chạy cung cấp |
| `CXX` | `g++` | Compiler cho extractor |
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.6` | Cấu hình ROCm của lượt chạy |
| `MTP_TEACHER_CACHE` | `data/cached_states/teacher.bin` | Cache teacher hoặc tên cache mới cần tạo |
| `MTP_PROMPT_MANIFEST` | `<cache>.prompts.txt` | Danh sách prompt theo đúng thứ tự record |
| `MTP_PROMPT_COUNT` | `64` | Giới hạn số prompt khi tạo cache. Sau khi bỏ biến thể lặp, seed hiện tại cho 45 prompt |
| `MTP_EPOCHS` | `4` | Số epoch lượt huấn luyện chính |
| `MTP_LEARNING_RATE` | `3e-6` | Learning rate lượt huấn luyện chính |

Để trích lại bằng seed hoặc model đích mới, dùng tên cache mới. Thay đổi `MTP_PROMPT_COUNT` chỉ có hiệu lực khi trích cache.

```bash
MTP_TEACHER_CACHE=data/cached_states/teacher-v2.bin \
MTP_PROMPT_COUNT=64 MTP_EPOCHS=4 MTP_LEARNING_RATE=3e-6 \
bash scripts/run_pipeline.sh
```

| Artifact | Nội dung |
|---|---|
| `data/cached_states/teacher.bin` | Token ID, độ dài prompt và hidden state từ teacher |
| `data/cached_states/teacher.bin.prompts.txt` | Một prompt mỗi dòng, thứ tự trùng cache |
| `output/teacher_mtp.pt` | State dict của khối MTP ở epoch tốt nhất trên validation |
| `output/mtp-BAAI_AREX-2-teacher-F16.gguf` | Sidecar với trọng số ma trận MTP F16, norm F32 và tensor dùng chung từ teacher |
| `output/mtp-BAAI_AREX-2-teacher-Q4_0.gguf` | Sidecar lượng tử hóa để chạy benchmark và inference |
| `output/best_mtp.pt` | Checkpoint lịch sử. Pipeline tự đo baseline này nếu file tồn tại |

## Từ prompt đến batch huấn luyện

1. **Chọn prompt.** Script bỏ tiền tố `[Variant N]`, khử trùng lặp, xáo trộn với seed 42, rồi lưu một prompt trên mỗi dòng. Prompt tự chuẩn bị cần là văn bản một dòng để khớp cách extractor đọc.
2. **Sinh trace.** [Extractor C++](src/extract_teacher.cpp) nạp AREX-2, reset bộ nhớ ngữ cảnh giữa các prompt, token hóa prompt và sinh tối đa 40 token bằng argmax. Nó dừng sớm nếu gặp EOG. Context của extractor là 512 token và prompt tối đa 192 token sau token hóa. Extractor lấy hidden state từ API nextn của llama.cpp tại mỗi token đã decode.
3. **Chia validation theo họ prompt.** [Trainer](src/train_teacher.py) gom các biến thể seed cùng chủ đề vào một nhóm, xáo trộn nhóm với seed 42 và lấy khoảng 20% số nhóm cho validation, tối thiểu 4 nhóm. Cần ít nhất 12 nhóm có continuation từ 2 token trở lên.
4. **Căn hàng cho MTP.** Tại vị trí `j`, runtime ghép `hidden[j-1]` với embedding của `token[j]` rồi dự đoán `token[j+1]`. Hàng đầu dùng hidden toàn 0. Ví dụ, với `A B C`, khối MTP nhận hidden sau `A` và embedding `B`, rồi dự đoán `C`. Prefix tham gia causal attention. Nhãn bắt đầu tại token continuation đầu tiên, nghĩa là hàng cuối của prompt dùng để dự đoán nhãn đầu tiên.

### Định dạng cache

Định dạng đang dùng trên máy little-endian được đọc tuần tự theo bảng.

| Phạm vi | Trường | Kiểu và kích thước |
|---|---|---|
| Đầu file | `hidden_width` | Một `int32`, hiện là 5120 |
| Mỗi record | `token_count`, `prompt_length` | Hai `int32` |
| Mỗi record | Token ID | `token_count` giá trị `int32` |
| Mỗi record | Các hàng hidden | `(token_count - 1) × hidden_width` giá trị FP32 liên tiếp |

Token cuối có thể dùng làm nhãn mà chưa cần decode để lấy hidden của chính nó. Cache và manifest là một cặp. Định dạng hiện tại chưa lưu fingerprint của model, tokenizer, prompt hoặc phiên bản llama.cpp. Khi thay một trong các thành phần đó, tạo cache riêng và lưu thông tin bản chạy cùng log. Trainer hiện đọc cache vào RAM, nên với tập lớn cần tính trước dung lượng hoặc triển khai đọc theo shard.

## Khối MTP và vòng train

[Qwen35MTPBlock](src/model_mtp.py) có hidden size 5120, FFN 17408, 24 attention head, 4 KV head, head dimension 256 và 64 chiều rotary. Khối chuẩn hóa embedding và hidden riêng, ghép theo thứ tự embedding trước, chiếu về 5120 chiều rồi chạy gated causal attention và SwiGLU. Với text, phần RoPE xoay cặp qua hai nửa vùng rotary để khớp graph mRoPE của llama.cpp. Đầu ra qua shared head norm trước khi nhân LM head của AREX-2.

- **Kiểm tra alignment.** Tái dựng argmax teacher từ hidden state, output norm và LM head cố định trên một chuỗi validation. Trainer dừng nếu tỷ lệ khớp dưới 90%. Pilot khớp 39/40 token. Bài kiểm tra này kiểm tra liên kết hidden và nhãn, với LM head đã được dequantize sang FP16.
- **Baseline và probe.** Đo checkpoint lịch sử nếu có, rồi đo warm-start Qwen. Chạy overfit 4 chuỗi trong 4 lượt để kiểm tra đường gradient, sau đó nạp lại warm-start và tạo optimizer mới cho lượt chính.
- **Loss.** Dùng cross-entropy trên token greedy teacher sinh. Embedding và LM head được lấy từ AREX-2 GGUF và giữ cố định. Chỉ cập nhật khối MTP.
- **Bộ nhớ.** Dùng một chuỗi mỗi bước, autocast FP16 và GradScaler. LM head có vocab 248.320, nên chia phép chiếu logits thành nhóm 8 vị trí. Gradient từ các nhóm được gom ở đầu ra MTP trước khi lan ngược qua khối. Optimizer là AdamW và gradient norm được clip ở 1,0.
- **Chọn epoch.** Ưu tiên top-1 validation, rồi chọn loss thấp hơn nếu bằng điểm. Checkpoint chỉ chứa trọng số. Chạy lại pipeline sẽ bắt đầu từ warm-start Qwen, còn `--previous` chỉ dùng để đo checkpoint đối chứng.
- **Export.** [Exporter](src/export_gguf.py) ánh xạ 15 tensor của khối sang `blk.64`, lấy tokenizer metadata và các tensor dùng chung từ AREX-2, giữ các trọng số norm ở F32, rồi gọi `llama-quantize`. Tên F16 chỉ nói đến các ma trận MTP được xuất ở F16. Tensor embedding và output đã lượng tử hóa của teacher vẫn được đóng gói trong sidecar.

### Chạy riêng train và export

Dùng hai lệnh dưới đây khi cần đổi tên artifact hoặc xuất lại checkpoint. Đọc đường dẫn model từ YAML trước khi chạy phần train, export hoặc inference.

```bash
export TEACHER="${MTP_TEACHER_GGUF:-$("$MTP_PYTHON" -c 'import yaml; print(yaml.safe_load(open("configs/training_config.yaml"))["model"]["target_gguf_path"])')}"
export WARM_START="${MTP_WARM_START_GGUF:-$("$MTP_PYTHON" -c 'import yaml; print(yaml.safe_load(open("configs/training_config.yaml"))["model"]["base_mtp_gguf_path"])')}"
```

```bash
"$MTP_PYTHON" -m src.train_teacher \
  --cache data/cached_states/teacher.bin \
  --prompts data/cached_states/teacher.bin.prompts.txt \
  --teacher "$TEACHER" --warm-start "$WARM_START" \
  --output output/teacher_trial.pt --epochs 4 --learning-rate 3e-6

"$MTP_PYTHON" -m src.export_gguf \
  --weights output/teacher_trial.pt \
  --base-gguf "$WARM_START" --target-gguf "$TEACHER" \
  --out-f16 output/teacher_trial-F16.gguf \
  --out-q4 output/teacher_trial-Q4_0.gguf \
  --quantize-bin "$LLAMA_CPP_DIR/build/bin/llama-quantize" --keep-f16
```

## Chạy inference và benchmark

Các ví dụ chạy trên máy có một MI50. Giữ thư mục lệnh là gốc repo. Chạy lệnh server trong terminal riêng.

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export GGML_CUDA_NO_PINNED=1
export BENCH_DIR="$(mktemp -d)"
printf 'Socket: %s/server.sock\n' "$BENCH_DIR"

"$LLAMA_CPP_DIR/build/bin/llama-server" \
  -m "$TEACHER" \
  -md output/mtp-BAAI_AREX-2-teacher-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 -ngld 99 --fit off --load-mode none --no-warmup \
  -c 768 -np 1 --host "$BENCH_DIR/server.sock"
```

Trong terminal gửi request, đặt `BENCH_DIR` theo đường dẫn server đã in. Đợi health trả `status: ok`, gửi một request warm-up và lưu các response đo chính.

```bash
curl -fsS --unix-socket "$BENCH_DIR/server.sock" http://localhost/health

curl -fsS --unix-socket "$BENCH_DIR/server.sock" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Count from one to ten:","n_predict":16,"temperature":0,"cache_prompt":false,"seed":42}' \
  http://localhost/completion >/dev/null

curl -fsS --unix-socket "$BENCH_DIR/server.sock" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain why prefix sums can count subarrays whose sum equals K.","n_predict":384,"temperature":0,"cache_prompt":false,"seed":42}' \
  http://localhost/completion >"$BENCH_DIR/draft-result.json"
```

Để lấy baseline, dừng server bằng Ctrl-C rồi chạy lại với cùng model, context và GPU, bỏ `-md`, `-ngld`, `--spec-type draft-mtp` và `--spec-draft-n-max`. Dùng thư mục socket mới. Warm-up lại, gửi cùng request và lưu baseline ở file khác. Với F16, đổi file `-md`. So 1 và 2 draft token bằng cách chạy lại server với giá trị `--spec-draft-n-max` tương ứng.

- **Acceptance** tính bằng tổng `timings.draft_n_accepted` chia tổng `timings.draft_n`. Một trường đếm vắng mặt tương ứng với 0 draft token ở response đó.
- **Tốc độ sinh** tính bằng tổng `timings.predicted_n` chia tổng `timings.predicted_ms / 1000`. Chỉ số này tính phần decode. Nếu cần đo độ trễ cả request, ghi thêm thời gian prefill và thời gian từ client.
- **Khớp đầu ra** đối chiếu trường `content` trên từng cặp request greedy. Ghi `stop_type` để phân biệt kết thúc tự nhiên với hết ngân sách token. Kiểm tra chất lượng câu trả lời riêng dựa trên đáp án hoặc tiêu chí đề bài.
- Khi ghi nhận kết quả, lưu prompt, giới hạn token, seed, bản GGUF, số draft token, build llama.cpp và response JSON. Giữ các điều kiện đó giống nhau khi so sánh.

## Kết quả trên MI50

Pilot dùng 38 chuỗi train và 7 chuỗi validation, có 280 token validation. Probe tăng từ 83,12% lên 100% trên 4 chuỗi của probe. Lượt chính chọn epoch 1 với loss 0,4026 và top-1 90,36%. Warm-start đạt loss 0,5228 và top-1 85,00%. Checkpoint lịch sử đạt 0,71%.

### Năm prompt ngắn, tối đa 48 token mỗi prompt

| Cấu hình | Draft được chấp nhận | Token/giây |
|---|---:|---:|
| Tắt draft | — | 20,77 |
| Warm-start Qwen Q4_0 với embedding và head AREX-2, 1 draft | 109/122 | 30,64 |
| F16 đã train, 1 draft | 109/122 | 29,85 |
| Q4_0 đã train, 1 draft | 108/122 | 29,93 |
| Q4_0 đã train, 2 draft | 138/188 | 31,54 |

Đầu ra greedy khi bật draft khớp baseline ở 4/5 prompt. Prompt Python còn lại khác ở chữ ký hàm. Top-1 validation tăng sau train. Trong phép so Q4_0 với 1 draft, warm-start nhỉnh hơn một token được chấp nhận. Benchmark này chưa đo warm-start ở chế độ 2 draft.

### Bốn prompt khó, tối đa 384 token mỗi prompt

| Prompt | Draft được chấp nhận | Bật draft / tắt draft, token/giây |
|---|---:|---:|
| Đếm mảng con tổng K, kể cả K = 0 | 233/298 | 32,64 / 20,32 |
| Tìm số có 5 chữ số mà số đảo bằng 4 lần số gốc | 229/307 | 31,72 / 20,17 |
| Bayes ba hộp bi bằng tiếng Việt | 228/308 | 31,37 / 20,03 |
| API giữ ghế và xử lý race khi xác nhận hoặc hết hạn | 203/359 | 27,03 / 19,90 |
| **Tổng hợp** | **893/1.272 (70,2%)** | **30,61 / 20,15** |

Bản Q4_0 với 2 draft tăng tốc khoảng 52% trên cả hai bộ. Với bộ khó, đầu ra greedy khác baseline ở cả 4 prompt và tất cả chạm giới hạn 384 token. Ví dụ, bản bật draft dùng đúng phương pháp prefix sum nhưng đưa ra xác suất sai ở phần thứ hai bài Bayes. Acceptance là số đo của quá trình xác minh draft. Tỷ lệ này không trực tiếp đo độ đúng của câu trả lời hay bảo đảm khớp từng token với baseline.

<details>
<summary>Prompt nguyên văn của hai bộ benchmark</summary>

Năm prompt ngắn, mỗi dòng là một request.

    The capital of France is
    One, two, three, four,
    Thủ đô của Việt Nam là
    Explain why a rainbow appears after rain in three sentences.
    Implement a Python function that finds the longest common prefix in a list of strings.

Bốn prompt khó, mỗi khối là một request. Giữ dấu xuống dòng trước câu dẫn cuối.

```text
Problem: Given an array of up to 200000 signed integers, count the subarrays whose sum is exactly K. Explain the invariant, prove correctness, and give Python 3 code. Address the case K = 0 explicitly.
Solution:
```

```text
Solve exactly: A five-digit positive integer N has distinct nonzero digits. The number formed by reversing its digits is 4N. Find N and prove uniqueness using the carry equations, without exhaustive search.
Solution:
```

```text
Có 3 hộp: hộp A có 2 bi đỏ, hộp B có 1 bi đỏ và 1 bi xanh, hộp C có 2 bi xanh. Chọn hộp đều, rút ngẫu nhiên một bi thấy đỏ. Tính xác suất viên còn lại trong cùng hộp cũng đỏ. Sau đó xét trường hợp hộp có xác suất được chọn lần lượt là 1/2, 1/3, 1/6. Giải thích bằng Bayes.
Lời giải:
```

```text
Design an API that reserves seats for 10 minutes and then confirms payment. Two requests may race for the same seat, payment webhooks may be duplicated, and expiration can race with confirmation. Give precise transaction invariants, state transitions, and a test matrix.
Design:
```

</details>

## Xử lý sự cố

| Dấu hiệu | Cách kiểm tra và xử lý |
|---|---|
| PyTorch không thấy ROCm hoặc llama.cpp báo không có GPU | Kiểm tra `/dev/kfd`, `/dev/dri/render*`, quyền truy cập của user, PyTorch bản ROCm và kết quả `llama-server --list-devices`. Khi CPU phải đọc lại model khoảng 20 GB trên máy RAM 15 GiB, benchmark có thể mất nhiều phút cho một prompt. |
| Server kẹt ở bước nạp, kernel báo `SVM mapping failed, exceeds resident system memory limit` | Cấu hình đã chạy trên MI50 dùng `GGML_CUDA_NO_PINNED=1` và `--load-mode none`. Dùng `--no-warmup` khi khởi tạo server rồi warm-up qua request trước khi đo. |
| CPU báo phép toán F32/F16 không hỗ trợ hoặc ROCm dừng ở `binbcast.cu` với file sidecar cũ | Xuất lại bằng exporter hiện tại để các tensor norm là F32. Cả hai bản GGUF mới đã được kiểm tra có 8 tensor norm F32. |
| Số record lệch manifest hoặc thiếu họ prompt | Kiểm tra đúng cặp cache và manifest. Tạo cache mới nếu seed thay đổi. Giữ ít nhất 12 họ prompt có continuation đủ dài và tránh để các biến thể cùng chủ đề rơi vào hai tập. |
| Alignment dưới 90% | Kiểm tra version teacher, tokenizer, thứ tự record, hidden state từ API nextn và dịch nhãn trong batch. Đọc log alignment trước khi điều chỉnh hyperparameter. |
| Train top-1 tăng nhưng validation tụt | So từng epoch với warm-start, theo dõi loss validation và giữ checkpoint tốt nhất. Trong pilot, lượt LR `3e-5` chọn được 83,57%, còn lượt `3e-6` đạt 90,36%. |
| Acceptance cao nhưng kết quả khác baseline | Lưu chính xác request, response và cấu hình. So đầu ra ở `temperature=0` trên cùng model đích. Đánh giá riêng chất lượng, độ lặp lại và tốc độ trên prompt đại diện trước khi đổi cấu hình sử dụng thường xuyên. |

## Kiểm tra mã và chạy tiếp thí nghiệm

```bash
bash -n scripts/run_pipeline.sh
"$MTP_PYTHON" -m unittest discover -s tests -v
```

Bộ test gồm 7 bài, có kiểm tra RoPE split-half, shape forward, warm-start và seed dataset. Kiểm tra toàn tuyến dùng kết quả alignment, validation và benchmark llama.cpp trên model export. [train.py](src/train.py) chứa luồng thử nghiệm lịch sử dùng loss trên biểu diễn, với dữ liệu ngẫu nhiên khi thiếu cache. Dùng [run_pipeline.sh](scripts/run_pipeline.sh) cho bài train teacher-token mô tả ở đây.

Khi mở rộng thí nghiệm, tăng số họ prompt theo workload, giữ một tập đánh giá độc lập, lưu checkpoint theo lượt và so warm-start với bản đã train ở cùng số draft token. Pilot hiện tại chủ yếu dùng seed theo chủ đề kỹ thuật, nên tỷ lệ đo được phản ánh đúng các bộ prompt đã chạy.
