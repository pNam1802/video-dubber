# Video Dubber AI

## Giới Thiệu

Video Dubber AI là ứng dụng web tự động chuyển đổi giọng nói tiếng Anh trong video sang tiếng Việt. Hệ thống kết hợp các công nghệ AI tiên tiến để cung cấp giải pháp thuyết minh video chuyên nghiệp.

## Tính Năng Chính

- **Nhận diện giọng nói**: Sử dụng Whisper để trích xuất text từ audio tiếng Anh
- **Tách giọng nói (tuỳ chọn)**: Dùng Demucs để tách vocal, giúp tăng độ chính xác nhận diện
- **Dịch tự động**: Chuyển đổi text sang tiếng Việt bằng MarianMT
- **Chọn chế độ dịch**: Product (MarianMT) hoặc Research (OpenAI LLM)
- **Bảo vệ thuật ngữ ML**: Tự động giữ nguyên các từ khóa chuyên ngành trong lĩnh vực Machine Learning
- **Tổng hợp giọng nói**: Tạo audio tiếng Việt tự nhiên với Edge TTS
- **Tối ưu hóa timing**: Tự động điều chỉnh tốc độ phát âm (speed-up) nếu audio dài hơn segment gốc
- **Xử lý video**: Ghép audio mới vào video gốc mà không mất chất lượng
- **Theo dõi tiến độ**: Giao diện người dùng hiển thị trạng thái xử lý real-time

## Yêu Cầu Hệ Thống

- Python 3.10 trở lên
- FFmpeg (xử lý video/audio)
- 4GB RAM tối thiểu
- GPU CUDA (tùy chọn, để tăng tốc độ)

## Cài Đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd video-over
```

### 2. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

Lưu ý: **Demucs** sẽ tự tải model lần đầu khi chạy (có thể mất vài phút).

Nếu dùng OpenAI LLM, hãy đặt biến môi trường `OPENAI_API_KEY`.
Mặc định model dịch là `gpt-4o-mini` (có thể override bằng `OPENAI_TRANSLATE_MODEL`).

### 3. Thiết Lập OPENAI_API_KEY (Nếu dùng Research mode)

**Cách 1: Biến môi trường hệ thống (khuyến nghị)**

**Windows:**
1. Start → tìm “Environment Variables”
2. User variables → New
3. Name: `OPENAI_API_KEY`, Value: `<your-key>`
4. Mở lại terminal

**macOS/Linux:**
```bash
export OPENAI_API_KEY="<your-key>"
```

**Cách 2: File .env (local)**
Tạo file `.env` ở thư mục dự án:
```
OPENAI_API_KEY=<your-key>
```

### 4. Cài Đặt FFmpeg

Tải từ [ffmpeg.org](https://ffmpeg.org/download.html) và giải nén vào hệ thống của bạn, hoặc sử dụng package manager:

**Windows (Chocolatey):**
```bash
choco install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

### 5. Cấu Hình FFmpeg Path (Nếu cần)

Mở file `tasks.py` và cập nhật đường dẫn FFmpeg nếu cần thiết:
```python
ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # Windows
# hoặc
ffmpeg_path = '/usr/local/bin/ffmpeg'      # macOS/Linux
```

## Hướng Dẫn Sử Dụng

### Khởi Động Ứng Dụng

```bash
python app.py
```

Ứng dụng sẽ chạy tại `http://localhost:5000`

### Quy Trình Sử Dụng

1. Truy cập giao diện web
2. Chọn file video (định dạng hỗ trợ: MP4, AVI, MOV, WebM)
3. Chọn chế độ dịch: **Product (MarianMT)** hoặc **Research (OpenAI LLM)**
4. Nhấn "Bắt đầu Dubbing"
5. Chờ hệ thống xử lý (thời gian tùy theo độ dài video)
6. Xem kết quả và **chất lượng dubbing** trong dashboard
7. Tải video đã thuyết minh về máy

## Cấu Trúc Dự Án

```
video-over/
├── app.py                    # Ứng dụng Flask chính
├── config.py                 # Cấu hình hệ thống
├── tasks.py                  # Logic xử lý video
├── requirements.txt          # Danh sách dependencies
├── README.md                 # Tài liệu này
├── .gitignore                # File ignore cho Git
├── templates/
│   ├── index.html            # Trang upload video + result + metrics
│   └── result.html           # (dùng chung index.html)
├── static/
│   └── js/
│       └── progress.js       # Xử lý giao diện + hiển thị metrics
├── uploads/                  # Thư mục chứa video upload
└── outputs/                  # Thư mục chứa video đã xử lý
```

## Quy Trình Xử Lý (Pipeline)

### Phiên Bản Cơ Bản
```
Video đầu vào
    ↓
Tách giọng nói (Demucs, tuỳ chọn)
    ↓
Nhận diện giọng nói (Whisper)
    ↓
Tối ưu segments (merge + filter)
    ↓
Bảo vệ thuật ngữ ML + Dịch sang Việt (MarianMT/OpenAI)
    ↓
Tổng hợp giọng Việt (Edge TTS)
    ↓
Điều chỉnh timing thông minh
    ↓
Ghép audio vào video (FFmpeg)
    ↓
Video đầu ra
```

### Phiên Bản Nâng Cao (Hiện Tại)
```
Video đầu vào
    ↓
Tách giọng nói (Demucs)
    ↓
Nhận diện giọng nói (Whisper)
    ↓
[MỚI] Phân tích ngữ nghĩa segments (extract keywords)
    ↓
[MỚI] Merge thông minh (dựa trên topic + câu + duration)
    │     • Phát hiện thay đổi chủ đề (semantic similarity)
    │     • Kiểm tra kết thúc câu tự nhiên
    │     • Giới hạn độ dài segment (2-8 giây)
    ↓
[MỚI] Dịch có ngữ cảnh (OpenAI/MarianMT với ngữ cảnh)
    ↓
Bảo vệ + Khôi phục thuật ngữ ML
    ↓
Tổng hợp giọng Việt (Edge TTS)
    ↓
[MỚI] Điều chỉnh timing thông minh (pause + speed)
    │     • Thêm pause tự nhiên nếu ngắn
    │     • Tăng tốc nhẹ nếu dài (max 1.3x)
    ↓
Ghép audio vào video (FFmpeg)
    ↓
Video đầu ra
```
    ↓
Khôi phục thuật ngữ ML
    ↓
Tổng hợp giọng Việt (Edge TTS)
    ↓
Đo thời lượng audio (ffprobe)
    ↓
Tăng tốc audio nếu cần (ffmpeg atempo)
    ↓
Ghép audio vào video (FFmpeg)
    ↓
Video đầu ra
```

## Công Nghệ Sử Dụng

| Thành Phần | Công Nghệ | Mục Đích |
|-----------|----------|---------|
| Framework | Flask | Web server |
| Nhận diện giọng | OpenAI Whisper | Trích xuất text từ audio |
| Tách giọng | Demucs | Tách vocal khỏi nhạc nền |
| Merge thông minh | Semantic Analysis | Phá hiện thay đổi chủ đề |
| Dịch | MarianMT + OpenAI | Dịch Anh → Việt (có ngữ cảnh) |
| Tổng hợp giọng | Edge TTS | Tạo audio Việt |
| Điều chỉnh timing | FFmpeg + Semantic | Pause + Speed adjustment |
| Xử lý video | FFmpeg + ffprobe | Ghép audio/video, đo thời lượng |
| Backend | Python | Logic chính |
| Frontend | HTML/CSS/JS | Giao diện người dùng |

## Cấu Hình

### File `config.py`

```python
UPLOAD_FOLDER = 'uploads'      # Nơi lưu video upload
OUTPUT_FOLDER = 'outputs'      # Nơi lưu video đã xử lý
ENABLE_VOICE_SEPARATION = True # Bật tách giọng (Demucs)
VOICE_SEPARATION_MODEL = 'htdemucs'
```

## Merge Thông Minh Dựa Trên Ngữ Nghĩa

### Tính Năng

Hệ thống **không chỉ merge dựa trên thời gian**, mà còn phân tích ngữ nghĩa để merge intelligently:

```
Segment 1: "Neural networks are powerful models"
Segment 2: "that can learn complex patterns"      ← Merge (cùng câu)
    ↓
Merged: "Neural networks are powerful models that can learn complex patterns"

Segment 3: "However, they require large datasets"  ← Không merge (đổi ý)
```

### Chiến Lược Merge

**3 Điều Kiện Cần Đạt:**

1. **Kiểm Tra Kết Thúc Câu**
   - Nếu segment hiện tại kết thúc câu (`.`, `!`, `?`) → Không merge
   - Nếu có từ nối (`and`, `but`, `or`) → Có thể merge

2. **Phát Hiện Thay Đổi Chủ Đề (Semantic Similarity)**
   ```
   Current group keywords: [neural, network, learn, patterns]
   New segment keywords: [dataset, training, samples]
   
   Overlap: 0 / 4 = 0% < 35% threshold → Đổi chủ đề, không merge
   ```

3. **Kiểm Tra Độ Dài Segment**
   - Merged duration: 2 - 8 giây (tối ưu cho TTS)
   - Tránh segments quá ngắn hoặc quá dài

### Lợi Ích

- ✅ **Dịch tự nhiên hơn**: Không cắt câu giữa chừng
- ✅ **Giảm số segments**: Từ ~100 xuống ~40-60
- ✅ **Timing tốt hơn**: Segments 2-8s phù hợp cho TTS
- ✅ **Consistency cao**: Dịch có ngữ cảnh tốt hơn

## Dịch Có Ngữ Cảnh (Context-Aware Translation)

### Tính Năng

Hệ thống sử dụng **dịch có ngữ cảnh** (context-aware) thay vì dịch từng segment độc lập:

```
Segment trước: "Neural networks are powerful..."
    ↓
Segment hiện tại: "We use attention mechanism"  ← Dịch có thêm ngữ cảnh
    ↓
Segment sau: "to improve model performance"
```

### Lợi Ích

- ✅ **Dịch chính xác hơn**: Hiểu toàn bộ ý tưởng
- ✅ **Consistency**: Duy trì cách dịch thống nhất trong video
- ✅ **Tự nhiên hơn**: Dịch ý nghĩa, không dịch sát từng từ
- ✅ **Cache thông minh**: Tránh dịch lại cùng đoạn

### Chiến Lược Fallback

```
1. Cố gắng dịch có ngữ cảnh
   ├─ OpenAI: Sử dụng LLM với prompt ngữ cảnh
   └─ MarianMT: Dịch nhanh, cache kết quả
   
2. Nếu có lỗi → Dịch đơn giản (không ngữ cảnh)
3. Nếu lỗi tiếp → Fallback sang MarianMT
```

### Validation Chất Lượng

- Kiểm tra độ dài dịch ±20% so với gốc (fit timing)
- Tránh dịch rỗng hoặc quá ngắn
- Log cảnh báo nếu chất lượng không đạt

### Ví Dụ

**Gốc:**
> "The transformer model uses self-attention to process sequences in parallel"

**Dịch không ngữ cảnh (MarianMT):**
> "Mô hình biến áp sử dụng self-attention để xử lý các chuỗi song song"

**Dịch có ngữ cảnh (OpenAI):**
> "Mô hình Transformer dùng cơ chế self-attention để xử lý các chuỗi song song"

## Đánh Giá Chất Lượng Dubbing

### Metrics Tự Động

Sau khi xử lý xong, hệ thống tự động hiển thị **dashboard chất lượng** trên giao diện:

| Metric | Mô Tả | Công Thức |
|--------|-------|-----------|
| **Quality Score** | Điểm chất lượng tổng thể (0-100) | Weighted average của các metrics |
| **Timing Accuracy** | Độ chính xác timing (%) | So sánh duration gốc vs audio tạo |
| **Length Ratio** | Tỷ lệ độ dài dịch | trans_length / original_length |
| **Total Segments** | Số segments sau merge | Đã tối ưu từ segments gốc |

### Công Thức Tính Overall Score

```
Score = (Timing Accuracy × 0.4) +
        (Length Ratio Accuracy × 0.3) +
        (Pause Naturalness × 0.2) +
        (Speed Variance × 0.1)
```

- ⭐ **≥ 85**: Tuyệt vời 🌟
- 👍 **70-84**: Tốt 👍
- 👌 **50-69**: Bình thường 👌
- ⚠️ **< 50**: Cần cải thiện ⚠️

### Ví Dụ Dashboard

```
Điểm chất lượng: ████████░░ 82.5%
Tuyệt vời 🌟

Timing Accuracy: 85.2%
Length Ratio: 1.05x
Total Segments: 42
```

## Tối Ưu Hóa Audio Timing (Thông Minh)

Hệ thống sử dụng chiến lược **tối ưu hóa thông minh** thay vì chỉ tăng tốc:

### Chiến Lược 3 Tầng

```
1. Chênh lệch < 10% → Giữ nguyên (âm thanh tự nhiên)
   ├─ 0.9x ≤ TTS/Original ≤ 1.1x
   
2. Ngắn hơn 20% → Thêm pause tự nhiên
   ├─ Tìm vị trí dấu câu, từ nối
   ├─ Phân bổ thời gian pause đều
   ├─ Pause: 50-300ms tùy vị trí
   
3. Dài hơn 20% → Tăng tốc từng bước
   ├─ Giới hạn max 1.3x (tự nhiên hơn)
   ├─ Sử dụng FFmpeg atempo (giữ pitch)
```

### Ví Dụ Thực Tế

```python
# Trường hợp 1: Gần đúng → Giữ nguyên
# Segment = 5s, TTS = 5.1s (ratio = 1.02) → Không điều chỉnh

# Trường hợp 2: Ngắn hơn → Thêm pause
# Segment = 5s, TTS = 3.8s (ratio = 0.76)
# → Tìm 3 vị trí pause, thêm 0.4s total

# Trường hợp 3: Dài hơn → Tăng tốc
# Segment = 5s, TTS = 6.5s (ratio = 1.3)
# → Tăng tốc: tempo = 5/6.5 = 0.77 (giới hạn 1.3x)
```

### Lợi Ích

- ✅ **Âm thanh tự nhiên hơn**: Pause ở vị trí hợp lý
- ✅ **Tránh tốc độ quá nhanh**: Max 1.3x thay vì 2.0x
- ✅ **Giữ chất lượng**: Sử dụng atempo (không mất chất lượng như stretching)

## Bảo Vệ Thuật Ngữ ML

Hệ thống tự động giữ nguyên các từ khóa chuyên ngành trong quá trình dịch:

**Từ khóa được bảo vệ** (47 thuật ngữ):
- Neural Networks: `neural network`, `transformer`, `LSTM`, `attention`, ...
- Training: `gradient descent`, `loss function`, `dropout`, `regularization`, ...
- Data: `dataset`, `preprocessing`, `augmentation`, `normalization`, ...
- Algorithms: `classification`, `clustering`, `random forest`, ...
- Frameworks: `TensorFlow`, `PyTorch`, `scikit-learn`, `Keras`, ...
- Evaluation: `accuracy`, `precision`, `recall`, `F1-score`, `AUC`, ...

**Cơ chế hoạt động**:
```
Text gốc: "Neural network uses gradient descent"
    ↓ (Replace keywords)
Text để dịch: "__ML_TERM_0__ uses __ML_TERM_8__"
    ↓ (Translate)
Text dịch: "__ML_TERM_0__ sử dụng __ML_TERM_8__"
    ↓ (Restore keywords)
Kết quả: "Neural network sử dụng gradient descent"
```

## Tài Liệu Tham Khảo

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Edge TTS](https://github.com/rany2/edge-tts)
- [FFmpeg](https://ffmpeg.org/)

