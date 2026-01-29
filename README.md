# Video Dubber AI

## Giới Thiệu

Video Dubber AI là ứng dụng web tự động chuyển đổi giọng nói tiếng Anh trong video sang tiếng Việt. Hệ thống kết hợp các công nghệ AI tiên tiến để cung cấp giải pháp thuyết minh video chuyên nghiệp.

## Tính Năng Chính

- **Nhận diện giọng nói**: Sử dụng Whisper để trích xuất text từ audio tiếng Anh
- **Dịch tự động**: Chuyển đổi text sang tiếng Việt bằng MarianMT
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

### 3. Cài Đặt FFmpeg

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

### 4. Cấu Hình FFmpeg Path (Nếu cần)

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
3. Nhấn "Bắt đầu Dubbing"
4. Chờ hệ thống xử lý (thời gian tùy theo độ dài video)
5. Tải video đã thuyết minh về máy

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
│   ├── index.html            # Trang upload video
│   └── result.html           # Trang hiển thị kết quả
├── static/
│   └── js/
│       └── progress.js       # Xử lý giao diện người dùng
├── uploads/                  # Thư mục chứa video upload
└── outputs/                  # Thư mục chứa video đã xử lý
```

## Quy Trình Xử Lý

```
Video đầu vào
    ↓
Nhận diện giọng nói (Whisper)
    ↓
Tối ưu segments (merge + filter)
    ↓
Bảo vệ thuật ngữ ML + Dịch sang Việt (MarianMT)
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
| Dịch | MarianMT | Dịch Anh → Việt |
| Tổng hợp giọng | Edge TTS | Tạo audio Việt |
| Xử lý video | FFmpeg + ffprobe | Ghép audio/video, đo thời lượng |
| Backend | Python | Logic chính |
| Frontend | HTML/CSS/JS | Giao diện người dùng |

## Cấu Hình

### File `config.py`

```python
UPLOAD_FOLDER = 'uploads'      # Nơi lưu video upload
OUTPUT_FOLDER = 'outputs'      # Nơi lưu video đã xử lý
```

## Tối Ưu Hóa Audio Timing

Hệ thống tự động đo độ dài audio TTS bằng **ffprobe** và so sánh với thời gian segment gốc:

- **Nếu TTS < segment gốc**: Giữ nguyên
- **Nếu TTS > segment gốc**: Tăng tốc audio bằng FFmpeg `atempo` filter (giữ pitch)
- **Buffer**: Sử dụng 95% thời gian gốc để tránh overlap sát nút

```python
# Ví dụ: Nếu segment = 5s, TTS = 6s
tempo = 5 * 0.95 / 6 = 0.79  # Tăng tốc ~21%
```

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

