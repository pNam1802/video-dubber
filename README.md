# Nora — Lồng tiếng video & đọc kịch bản bằng AI

Tải lên một video, nhận lại video **đã lồng tiếng** (Việt, Anh, Nhật, Trung, Hàn) kèm phụ đề —
giọng đọc liền mạch kiểu video review trên TikTok, có thể chọn giọng miễn phí, giọng AI có cảm
xúc hoặc clone giọng của chính bạn. Hoặc dán một kịch bản, nhận lại file đọc mp3 hay **video
Manhwa recap** có tranh minh hoạ vẽ tự động.

**Dùng thử:** https://pnam1802--video-dubber-web.modal.run

## Sản phẩm làm được gì

### Lồng tiếng video

- **Lồng tiếng tự động:** nhận dạng lời nói, dịch, tạo giọng đọc và ghép lại vào video gốc.
  Video 11 phút xử lý xong trong khoảng 4–5 phút với giọng miễn phí; giọng Gemini lâu hơn
  (khoảng 10 phút) do giới hạn 10 request/phút của Gemini TTS.
- **Nhiều cặp ngôn ngữ:** video nói tiếng Anh, Việt, Nhật, Trung, Hàn (hoặc tự nhận biết); dịch
  sang tiếng Việt, Anh, Nhật, Trung, Hàn — gồm cả chiều **Việt → Anh**.
- **Dịch tự nhiên cho mọi chủ đề** (phim, truyện, vlog, bài giảng...): trước khi dịch, AI đọc
  hiểu cả transcript (tóm tắt nội dung + bảng từ lóng/tên riêng) nên xưng hô và tên riêng nhất
  quán cả video. Hai kiểu dịch: **Tự nhiên** hoặc **Bản địa hoá** (đổi câu đùa, tiếng lóng sang
  meme quen thuộc của nước đích).
- **Duyệt lời thoại trước khi tạo giọng:** sửa câu nghe nhầm/dịch chưa ưng, hoặc bấm dịch lại
  từng câu (vẫn hiểu ngữ cảnh cả video) — giọng đọc và video được tạo đúng theo bản đã duyệt.
- **Nhiều giọng đọc, nghe thử ngay trong form:**
  - Tiếng Việt mặc định (VieNeu-TTS, mã nguồn mở): giọng **Nora** và 6 giọng Việt có sẵn.
  - Tiếng Anh mặc định (Chatterbox-Turbo, mã nguồn mở): giọng **Narrator** kể chuyện và 1 giọng
    nữ có sẵn.
  - **Clone giọng của bạn** (VieNeu cho tiếng Việt, Chatterbox cho tiếng Anh) từ 3–15 giây ghi
    âm — tải file lên hoặc **thu bằng mic** ngay trên trang.
  - Miễn phí (edge-tts): HoaiMy, NamMinh, Vivienne, Seraphina (nữ), Florian (nam); đích tiếng
    Anh dùng Aria/Guy.
  - Có cảm xúc (Gemini TTS): Leda (nữ), Charon (nam) — đọc hồi hộp, nhấn nhá theo nội dung.
- **Đọc liền mạch:** đọc trọn câu, nghỉ ngắn và đều giữa các câu, không có khoảng lặng dài; câu
  cuối luôn được đọc hết (giữ khung hình cuối nếu cần).
- **Tua nhanh hình ở khoảng lặng** (bật sẵn, tối đa 2×): giọng Việt thường ngắn hơn lời gốc, nên
  thay vì để khoảng nghỉ, hình được tua cho khớp giọng — giọng vẫn đọc ở tốc độ tự nhiên.
- **Âm lượng chuẩn mạng xã hội:** đầu ra chuẩn hoá về −14 LUFS như YouTube/TikTok.
- **Phụ đề** song ngữ, chỉ ngôn ngữ đích, chỉ ngôn ngữ gốc hoặc không phụ đề; tải về dạng `.srt`.

### Đọc kịch bản

- Dán văn bản (tối đa 20.000 ký tự, tiếng Việt hoặc Anh), chọn giọng (kể cả giọng clone), nhận
  file **mp3 đọc liền mạch + phụ đề SRT** — đọc nguyên văn, không dịch.
- **Manhwa recap:** AI đọc cả kịch bản, dựng storyboard với nhân vật cố định, vẽ tranh màu nước
  kiểu manhwa (Z-Image-Turbo) khoảng 6 giây một tranh, ghép thành video có zoom chậm trên nền mờ.
  Chọn kiểu nhân vật **Cô gái hướng dương** (mặc định) hoặc **Mơ màng**, có ảnh mẫu để xem trước.

### Tài khoản & tiện ích

- Đăng nhập (kể cả Google), đổi mật khẩu, lịch sử có ảnh thu nhỏ, email báo khi job cần duyệt /
  xong / lỗi, hạn mức theo người dùng, trang thống kê và quản lý người dùng cho quản trị viên.

## Demo

### Giọng mặc định Nora + tua nhanh hình

Đoạn 2,5 phút bên dưới (cùng đoạn với phần so sánh giọng), giọng **Nora** (VieNeu-TTS), tua
nhanh hình ở khoảng lặng: video còn **2:01**, **0/26** chỗ nghỉ ≥ 0,5 giây, giọng đọc tốc độ tự
nhiên (không câu nào bị kéo giãn). Không tua hình thì cùng đoạn này có 21/26 chỗ nghỉ ≥ 0,5 giây.

https://github.com/user-attachments/assets/567150fb-4758-4072-bada-550afae19ce2

File: [5_VieNeu_Nora_tua_hinh.mp4](demo/voices/5_VieNeu_Nora_tua_hinh.mp4)

### So sánh giọng đọc

Cùng một đoạn 2,5 phút, dịch bằng DeepSeek, xếp giọng liền mạch — chỉ khác giọng đọc
(mỗi job dịch riêng nên câu chữ có thể lệch nhau đôi chút). Xem bản gốc tiếng Anh trước để
so sánh:

| Video | Giọng | Khoảng lặng ≥ 0,8 giây | Chi phí giọng đọc |
|---|---|---|---|
| [0_Goc_tieng_Anh.mp4](demo/voices/0_Goc_tieng_Anh.mp4) | **Bản gốc tiếng Anh** | — | — |
| [1_Gemini_Leda_nu.mp4](demo/voices/1_Gemini_Leda_nu.mp4) | Gemini Leda (nữ) | 0 | ≈ $0,035 |
| [2_Gemini_Charon_nam.mp4](demo/voices/2_Gemini_Charon_nam.mp4) | Gemini Charon (nam) | 4 | ≈ $0,033 |
| [3_Edge_HoaiMy_nu.mp4](demo/voices/3_Edge_HoaiMy_nu.mp4) | edge-tts HoaiMy (nữ) | 2 | Miễn phí |
| [4_Edge_Florian_nam.mp4](demo/voices/4_Edge_Florian_nam.mp4) | edge-tts Florian (nam) | 4 | Miễn phí |

#### Bản gốc (tiếng Anh)

[▶ Xem video gốc](demo/voices/0_Goc_tieng_Anh.mp4)

#### Gemini Leda (nữ)

https://github.com/user-attachments/assets/7828d5d9-df73-46ab-b3ad-b27781d11006

#### Gemini Charon (nam)

https://github.com/user-attachments/assets/d5559380-e327-43c0-ba26-4cd37f28e76a

#### edge-tts HoaiMy (nữ)

https://github.com/user-attachments/assets/1e4878cf-9b97-4537-b8cb-7934fdbe94c7

#### edge-tts Florian (nam)

https://github.com/user-attachments/assets/44b4492c-fca9-4ca0-a7fc-d7be3dc0026b

> **Nguồn:** trích đoạn 7:30–10:00 của video *"Chap 6 — The Apocalypse Has Come"* thuộc kênh
> **Manhwa Academy**. Bản quyền nội dung gốc thuộc về kênh và tác giả truyện. Đoạn trích chỉ dùng
> để minh hoạ chất lượng lồng tiếng cho mục đích học tập/đồ án, không dùng thương mại; sẽ gỡ nếu
> chủ sở hữu yêu cầu.

## Chi phí mỗi video

Chỉ tính tiền gọi API (dịch + giọng đọc), cho video 11 phút (~14.000 ký tự tiếng Việt):

| Trường hợp | Dịch | Giọng đọc | Tổng | Theo phút video |
|---|---|---|---|---|
| **Rẻ nhất** — DeepSeek ngoài giờ cao điểm + edge-tts | ≈ $0,004 | $0 | **≈ $0,004** | ≈ $0,0004 |
| DeepSeek giờ cao điểm + edge-tts | ≈ $0,0085 | $0 | **≈ $0,009** | ≈ $0,0008 |
| DeepSeek + Gemini TTS (Flash-Lite) | ≈ $0,0085 | ≈ $0,09 | **≈ $0,10** | ≈ $0,009 |
| **Đắt nhất** — DeepSeek lỗi, dịch bằng Gemini + Gemini TTS | ≈ $0,024 | ≈ $0,09 | **≈ $0,11** | ≈ $0,01 |

- **Máy chủ** (Modal: GPU nhận dạng + CPU tạo giọng/ghép video) thêm khoảng **$0,015 / video**,
  chưa kể container web chạy 24/7.
- Giờ cao điểm DeepSeek: 01:00–04:00 và 06:00–10:00 UTC (8–11h, 13–17h giờ Việt Nam), thứ Hai–Sáu;
  ngoài giờ đó giá một nửa.
- **Từ 01/01/2027** Gemini TTS và `gemini-3.6-flash` tăng giá gấp đôi → trường hợp đắt nhất
  khoảng $0,23/video. DeepSeek + edge-tts không đổi.
- Chi phí thật của từng job hiện trên trang job ("Chi phí máy chủ", "Chi phí dịch"). Hoá đơn:
  DeepSeek platform, [Google AI Studio](https://aistudio.google.com) → Billing, Modal → Usage.

## Cách dùng

**Lồng tiếng video**

1. Đăng nhập, chọn video (MP4, MOV, AVI, MKV, WEBM).
2. **Ngôn ngữ:** kiểm tra "Video đang nói" (mặc định tự nhận biết), bấm thẻ ngôn ngữ ở "Dịch sang".
3. **Giọng đọc:** chọn một thẻ giọng (bấm ▶ để nghe thử). Chọn "Giọng của bạn" thì tải hoặc thu
   giọng mẫu ngay tại đó.
4. "Tuỳ chọn thêm" (không bắt buộc): phụ đề, âm thanh gốc (mặc định 0%), độ chính xác, kiểu dịch,
   tua hình; Whisper, công cụ dịch và API key riêng nằm trong "Thiết lập kỹ thuật".
5. Bấm **Bắt đầu lồng tiếng**. Khi dịch xong, job dừng ở bước **Duyệt lời thoại** — sửa câu
   nếu cần rồi bấm **Xác nhận, tạo giọng đọc**.
6. Tải video và phụ đề ở trang kết quả; khung lời thoại bên cạnh tô câu đang đọc, bấm để tua.

**Đọc kịch bản:** vào mục **Đọc kịch bản** trên sidebar, dán văn bản, chọn ngôn ngữ, giọng và
kiểu nhân vật, bấm bắt đầu. Kết quả (video recap hoặc mp3 + SRT) nằm trong Lịch sử như job video.

---

# Phần kỹ thuật

## Pipeline

1. Tách âm thanh bằng ffmpeg.
2. Nhận dạng giọng nói bằng Whisper (faster-whisper, dự phòng openai-whisper), lấy mốc thời gian
   từng từ và tách thành câu trọn vẹn. Video nói tiếng Việt luôn dùng `large-v3-turbo` (model nhỏ
   nghe "bà chị" thành "3 chị" rồi dịch sai theo).
3. **Đọc hiểu:** LLM đọc cả transcript, viết tóm tắt + bảng từ lóng/tên riêng, lưu vào job
   (`translation_notes`) để cả lúc "dịch lại câu này" vẫn có ngữ cảnh. Prompt dịch cũng được báo
   transcript là bản nhận dạng tự động, có thể nghe nhầm.
4. Dịch theo lô bằng DeepSeek (mặc định), lỗi thì lùi sang Gemini rồi OpenAI; mỗi lô kèm các câu
   đã dịch ngay trước làm ngữ cảnh. Kiểu dịch `natural` hoặc `localized` (bản địa hoá).
5. Dừng chờ người dùng duyệt lời thoại.
6. Tạo giọng đọc (VieNeu / Chatterbox / edge-tts / Gemini TTS / gTTS) và xếp liền mạch lên dòng
   thời gian.
7. Tua nhanh hình ở khoảng lặng (nếu bật), trộn giọng vào video bằng ffmpeg, chuẩn hoá −14 LUFS,
   xuất video + SRT.

Bước 1–4 chạy trên container GPU (T4), bước 6–7 chạy trên container CPU (VieNeu/Chatterbox gọi
sang container GPU riêng) — xem "Triển khai Modal".

## Giọng đọc

Hai thứ quyết định bản lồng tiếng nghe có "TikTok" hay không: đọc **trọn câu** và **liền mạch**.

- **Tách câu theo từng từ**: Whisper trả mốc thời gian từng từ, câu được cắt đúng tại dấu
  chấm (quá 12 giây mới cắt ở dấu phẩy). Trước đây ghép theo đoạn Whisper, 46/66 câu bị cắt
  giữa chừng và giọng đọc hạ giọng như hết câu ngay giữa ý.
- **Xếp giọng liền mạch** ([`core/voice_timeline.py`](core/voice_timeline.py)): câu sau vào
  ngay khi câu trước đọc xong (+0,25 giây), được vào sớm tối đa 2 giây; một tốc độ chung cho cả
  video để tổng giọng Việt khớp thời gian nói gốc, câu còn hở thì đọc chậm thêm tối đa 10%.
  Không bao giờ chồng giọng. Đo trên video 11 phút: khoảng lặng ≥ 0,8 giây từ 57 còn 8.
- **Đổi tốc độ không làm méo giọng**: edge-tts tổng hợp lại với tham số `rate`; VieNeu/Gemini/gTTS
  dùng `atempo` của ffmpeg trên bản đã có (không gọi API lần nữa). Riêng VieNeu **không bao giờ
  bị làm chậm** (atempo giãn chậm tới ~0,79× nghe lê), chỉ tăng tốc khi câu dài quá khung hình.
- **Tua nhanh hình ở khoảng lặng** ([`core/video_retime.py`](core/video_retime.py), bật sẵn): mỗi
  câu ứng với một đoạn hình — từ lúc câu đó bắt đầu tới lúc câu sau bắt đầu — đoạn đó được tua
  nhanh vừa đủ cho câu tiếng Việt + 0,25 giây nghỉ, tối đa 2×. Không bao giờ làm chậm hình; đoạn
  intro trước câu đầu giữ nguyên. Hình đổi tốc độ bằng **một** bộ lọc `setpts` theo hàm từng khúc
  (giải mã một lần, giữ fps gốc); âm thanh gốc cắt từng khúc + `atempo` (giữ cao độ) rồi nối lại;
  phụ đề SRT theo đúng lúc giọng đọc. Đoạn Chap 6 (2:30): còn 2:01, chỗ nghỉ ≥ 0,5 giây từ 21/26
  còn 0/26, ghép video lâu thêm chưa tới 1 giây. Video ra có `+faststart` (phát ngay khi đang tải).
  Biểu thức `setpts` dựng thành cây `if()` lồng nhau — dạng tổng phẳng làm ffmpeg 5.1 trên Modal
  báo "Cannot allocate memory" từ khoảng 100 khúc.
- **Không cắt cụt câu cuối** ([`core/video_composer.py`](core/video_composer.py)): giọng đọc xong
  muộn hơn cuối hình thì `tpad` giữ khung hình cuối, `apad` kéo dài tiếng, thêm 0,6 giây
  (`END_HOLD_SEC`).
- **Âm lượng −14 LUFS** (`OUTPUT_LOUDNORM`): video lồng tiếng và mp3 đọc kịch bản đều qua
  `loudnorm`, rồi chốt `aformat` stereo vì ffmpeg 5.1 không tự chọn được channel layout.
- **Chặn âm thanh hỏng**: file TTS ngắn bất thường so với số âm tiết bị tạo lại (edge-tts đôi khi
  trả file có dung lượng nhưng không có tiếng).

| Công cụ | Giọng | Chi phí | Ghi chú |
|---|---|---|---|
| **VieNeu-TTS** (mặc định, [mã nguồn mở Apache 2.0](https://github.com/pnnbao97/VieNeu-TTS)) | **Nora** (nữ, mặc định), Ngọc Huyền, Quỳnh Anh, Mỹ Duyên, Kim Thanh (nữ), Thiện Minh, Đức Trí (nam), **Giọng của bạn** (clone) | Miễn phí (chỉ tốn GPU ≈ $0,01 / video) | Chạy trên GPU T4 riêng (`VoiceCloner`), đọc cả video trong một lượt: ≈ 25 giây cho 10 phút giọng đọc. Âm thanh có watermark AI |
| **Chatterbox-Turbo** (mặc định cho tiếng Anh, [mã nguồn mở MIT](https://github.com/resemble-ai/chatterbox)) | **Narrator** (kể chuyện, mặc định), giọng nữ có sẵn, **Giọng của bạn** (clone) | Miễn phí (chỉ tốn GPU) | Chỉ đọc tiếng Anh. Container GPU T4 riêng (`EnglishCloner`, torch 2.6 khác Whisper/VieNeu); mẫu ngắn hơn 5 giây được lặp cho đủ |
| edge-tts | HoaiMy, NamMinh, Vivienne, Seraphina (nữ), Florian (nam); Aria, Guy khi đích tiếng Anh | Miễn phí | Vivienne/Seraphina/Florian là giọng *Multilingual* đọc được tiếng Việt |
| Gemini TTS (`gemini-3.8-flash-lite-tts`) | Leda (nữ), Charon (nam) | ≈ $0,09 / 10 phút giọng đọc (giá tới hết 2026) | Có cảm xúc theo chỉ dẫn `GEMINI_TTS_STYLE` (có bản riêng cho từng ngôn ngữ); giới hạn 10 request/phút ở Tier 1 |
| gTTS | — | Miễn phí | Dự phòng cuối cùng |

VieNeu chỉ đọc tiếng Việt, Chatterbox chỉ đọc tiếng Anh (`TTS_ENGINE_LANGUAGES`) — form và API
chặn engine không hợp ngôn ngữ đích. Gemini, VieNeu hoặc Chatterbox lỗi thì câu đó rơi về
edge-tts (giọng đúng ngôn ngữ đích), rồi gTTS.

Mỗi giọng có sẵn có file nghe thử trong `static/voices/<engine>/`, tạo một lần bằng
`python scripts/make_voice_previews.py` (chạy trên Modal, nhận tên engine để chỉ tạo lại giọng của engine đó).

**Clone giọng:** vào **Cài đặt → Giọng của bạn** (hoặc chọn "Giọng của bạn" ngay trong form),
tải lên hoặc thu bằng mic 3–15 giây ghi âm (đọc rõ, không nhạc nền) và xác nhận đó là giọng của
bạn hoặc đã được chủ giọng đồng ý. Mỗi người có **một giọng mẫu cho mỗi ngôn ngữ** — clone giữ cả
ngữ điệu của mẫu, nên mẫu tiếng Việt đọc bản tiếng Anh nghe lơ lớ: VieNeu clone từ mẫu tiếng Việt,
Chatterbox từ mẫu tiếng Anh (`CLONE_SAMPLE_LANGUAGE`).

Giọng mặc định **Nora** cũng là giọng clone, từ một đoạn giọng do chính VieNeu tạo ra
([`core/voices/nora.wav`](core/voices/nora.wav)), không phải giọng người thật. Giọng **Narrator**
tiếng Anh clone từ mẫu 13,7 giây ([`core/voices/narrator_en.wav`](core/voices/narrator_en.wav)),
chủ dự án có quyền sử dụng.

## Đọc kịch bản

- [`core/script_reader.py`](core/script_reader.py): tách câu theo đoạn/dấu câu (câu dài hơn 250
  ký tự được chia nhỏ), đọc rồi nối với khoảng nghỉ 0,35 giây (0,9 giây khi xuống đoạn), xuất mp3
  −14 LUFS + SRT. Giới hạn 20.000 ký tự (`SCRIPT_MAX_CHARS`) để không chạm timeout 1800 giây của
  `EnglishCloner`.
- [`core/script_images.py`](core/script_images.py): Gemini đọc cả kịch bản, dựng storyboard với
  ngoại hình nhân vật viết **một lần** rồi chèn nguyên văn vào mọi prompt để nhân vật đồng nhất;
  ảnh chỉ đổi ở ranh giới câu. Gemini lỗi thì vẽ thẳng theo lời đọc.
- [`core/recap.py`](core/recap.py) — chế độ **Manhwa recap** (mặc định trên form): tranh khổ dọc
  768×960 giữa nền mờ, khoảng 6 giây một tranh (tối đa 150), zoom chậm, 25 fps. Sau khi vẽ có
  bước hậu kỳ `grade()` — giảm sáng, tăng bão hoà, nhuộm bóng tím / vùng sáng hồng đào, làm nét,
  bloom và hạt sáng — để khớp tông ảnh mẫu, không tốn thêm GPU.
- [`core/image_client.py`](core/image_client.py) gọi `ImageGenerator` trên Modal (GPU L40S):
  Z-Image-Turbo (Apache 2.0), khoảng 3,2 giây/ảnh.
- Các chế độ ảnh cũ (`none`, `few`, `scenes`, `manga`) vẫn nhận qua API/CLI nếu gửi tường minh.
  Vẽ ảnh hỏng thì job vẫn xong với mp3 và báo lý do.

## Công nghệ

- Python 3.12, ffmpeg/ffprobe
- faster-whisper (dự phòng openai-whisper) + torch
- DeepSeek / OpenAI (openai SDK), Gemini (google-genai)
- VieNeu-TTS, Chatterbox-Turbo, edge-tts, Gemini TTS, gTTS, pydub
- Z-Image-Turbo (vẽ tranh), Pillow (dựng khung recap/manga)
- Flask, SQLAlchemy, Alembic; Postgres (Neon) khi chạy trên Modal
- Modal (GPU T4 + L40S + CPU), Tailwind CSS

## Cấu trúc thư mục

	app/                Flask application package
	  __init__.py       create_app() factory
	  models.py         User, Job, TranscriptSegment
	  auth.py, oauth.py đăng nhập / đăng ký / Google / đổi mật khẩu
	  main.py           trang chính, phục vụ file, /healthz
	  api.py            upload, đọc kịch bản, tiến trình, duyệt lời thoại, giọng mẫu
	  jobs.py           điều phối job (thread local / Modal), ghi tiến trình vào DB
	  voices.py         lưu / kiểm tra giọng mẫu clone của người dùng
	  admin.py          thống kê, quản lý người dùng, trạng thái hệ thống
	  quota.py          hạn mức theo người dùng
	  mailer.py         email báo trạng thái job
	  cli.py            flask create-admin, flask set-role
	  errors.py         chuẩn hoá lỗi JSON/HTML
	core/               pipeline (không phụ thuộc Flask)
	  pipeline.py       điều phối hai giai đoạn lồng tiếng + đọc kịch bản
	  transcriber.py    Whisper + tách câu theo mốc thời gian từng từ
	  translator/       DeepSeek, Gemini, OpenAI (dùng chung llm_common.py: đọc hiểu, kiểu dịch)
	  tts_engine.py     VieNeu, Chatterbox, edge-tts, Gemini TTS, gTTS + đổi tốc độ đọc
	  vieneu_client.py  gọi VieNeu-TTS (container GPU VoiceCloner)
	  chatterbox_client.py gọi Chatterbox-Turbo (container GPU EnglishCloner)
	  voice_timeline.py xếp giọng đọc liền mạch lên dòng thời gian
	  video_retime.py   tua nhanh hình ở khoảng lặng cho khớp giọng đọc
	  video_composer.py trộn giọng vào video bằng ffmpeg, chuẩn hoá âm lượng
	  script_reader.py  đọc kịch bản thành mp3 + SRT
	  script_images.py  storyboard + ảnh minh hoạ cho kịch bản
	  recap.py, manga.py dựng video Manhwa recap / trang manga
	  image_client.py   gọi Z-Image-Turbo (container GPU ImageGenerator)
	  voices/           file mẫu của giọng Nora và Narrator
	scripts/            make_voice_previews.py — tạo file nghe thử giọng
	static/voices/      file nghe thử từng giọng
	assets/             nguồn Tailwind, file Blender của hình hero trang giới thiệu
	demo/               video demo (xem mục Demo)
	migrations/         Alembic
	modal_app.py        khai báo triển khai Modal
	wsgi.py             điểm vào WSGI

## Cài đặt & chạy local

Yêu cầu: Python 3.10+, ffmpeg và ffprobe có trong PATH.

	pip install -r requirements.txt
	copy .env.example .env
	flask --app wsgi db upgrade
	flask --app wsgi create-admin

Đặt `DEEPSEEK_API_KEY` trong `.env` (công cụ dịch mặc định). `GEMINI_API_KEY` bật dịch dự phòng
bằng Gemini và các giọng Gemini TTS; `OPENAI_API_KEY` là dự phòng cuối. Database mặc định là
SQLite `data/database.db`; đặt `DATABASE_URL` trỏ tới Postgres cho production (bắt buộc trên Modal).

Chạy web (dev):

	python wsgi.py

Production (waitress):

	set APP_ENV=production
	set FLASK_SECRET_KEY=<random hex>
	waitress-serve --listen=0.0.0.0:8000 wsgi:app

`APP_ENV=production` bắt buộc có `FLASK_SECRET_KEY`. Tạo bằng
`python -c "import secrets; print(secrets.token_hex(32))"`. `GET /healthz` trả 200 khi database,
ffmpeg và ffprobe đều sẵn sàng, 503 nếu không.

CLI (mặc định dịch bằng DeepSeek, key lấy từ `DEEPSEEK_API_KEY`):

	python cli.py path/to/video.mp4
	python cli.py path/to/video.mp4 --translator gemini --gemini-api-key AIza...
	python cli.py path/to/video.mp4 --translator openai --openai-api-key sk-...

Kết quả nằm ở `data/outputs/*_dubbed.mp4` và `data/outputs/*_dubbed.srt`. Lần chạy đầu chậm hơn
vì phải tải model Whisper.

Chạy test (không cần torch/whisper):

	pip install -r requirements-dev.txt
	pytest

## Triển khai Modal

```
   Trình duyệt
        │  upload / poll
        ▼
┌──────────────────────┐ spawn() ┌──────────────────────────┐
│  Web · CPU           │────────▶│  Dubber · GPU T4         │  giai đoạn 1:
│  @modal.wsgi_app()   │         │  Whisper nạp 1 lần       │  nhận dạng + dịch
│  min_containers=1    │         └────────────┬─────────────┘
│                      │ spawn() ┌──────────────────────────┐
│                      │────────▶│  Composer · CPU          │  giai đoạn 2:
└──────────┬───────────┘         │  region="us"             │  giọng đọc + ghép
           │                     └────────────┬─────────────┘
           ▼                                  ▼ ghi tiến trình / file
   ┌───────────────────┐              ┌────────────────────────┐
   │ Postgres (Neon)   │◀─────────────│ Modal Volume           │
   │ user · job · quota│              │ video, srt, model cache│
   └───────────────────┘              └────────────────────────┘
```

Toàn bộ khai báo nằm trong [`modal_app.py`](modal_app.py): container web (CPU) bọc Flask
bằng `@modal.wsgi_app()`; giai đoạn 1 (nhận dạng + dịch) chạy trên `Dubber` —
`@app.cls(gpu="T4")` với `@modal.enter()` nạp model một lần; giai đoạn 2 (giọng đọc + ghép
video) chạy trên `Composer` — container **CPU** (`cpu=1.0`, `region="us"`), vì bước này không
dùng GPU (chờ mạng + ffmpeg). Tách ra giảm chi phí máy chủ mỗi video 11 phút từ ≈ $0,051 xuống
≈ $0,015. Tất cả dùng chung một Volume cho file và chung Postgres cho tiến trình.

Các model nặng còn lại mỗi thứ một container GPU riêng (môi trường Python khác nhau, chỉ bật khi
cần, `max_containers=1`, tắt sau 60 giây rảnh):

| Container | GPU | Việc |
|---|---|---|
| `Dubber` | T4 | Whisper + dịch (giai đoạn 1) |
| `Composer` | CPU | giọng đọc, ghép video, đọc kịch bản (giai đoạn 2) |
| `VoiceCloner` | T4 | VieNeu-TTS (tiếng Việt, clone giọng) |
| `EnglishCloner` | T4 | Chatterbox-Turbo (tiếng Anh, clone giọng) |
| `ImageGenerator` | L40S | Z-Image-Turbo vẽ tranh cho Đọc kịch bản |

Ba ràng buộc của Modal quyết định toàn bộ thiết kế:

| Ràng buộc | Hệ quả |
|---|---|
| Container ephemeral, chạy song song | Không dùng được SQLite và `data/outputs` trên đĩa → Postgres hosted + Volume |
| Web endpoint bị cắt ở **150 giây** | Không dùng SSE. Pattern chuẩn: `spawn()` trả call id → client poll |
| Tính tiền theo giây container (kể cả cold start) | `max_containers`, quota theo user, và tối ưu cold start là bắt buộc |

Các bước, chạy một lần:

	pip install modal
	modal setup
	modal volume create dubber-data
	modal volume create dubber-models
	modal secret create video-dubber \
	    APP_ENV=production \
	    FLASK_SECRET_KEY=<random hex> \
	    DATABASE_URL=postgresql://... \
	    JOB_RUNNER=modal \
	    DATA_DIR=/data \
	    DEEPSEEK_API_KEY=... \
	    GEMINI_API_KEY=...

Mỗi lần deploy:

	modal run modal_app.py::migrate     # flask db upgrade trên Postgres
	modal deploy modal_app.py

Xem log: `modal app logs video-dubber`.

### Cách web và container xử lý nói chuyện với nhau

| | Cơ chế |
|---|---|
| Gửi việc | Web gọi `Dubber().run.spawn(...)` (giai đoạn 1) / `Composer().run_phase2.spawn(...)` (giai đoạn 2), lưu `call.object_id` vào `Job.modal_call_id` |
| Truyền file | Tất cả mount chung Volume `dubber-data` tại `/data`; bên ghi `commit()`, bên đọc `reload()` |
| Báo tiến trình | Container xử lý ghi thẳng vào Postgres, web chỉ đọc DB — không cần Redis |
| Huỷ job | `modal.FunctionCall.from_id(call_id).cancel()` |
| Theo dõi | Client poll `GET /api/progress/<id>`; request HTTP không bao giờ chạm giới hạn 150 giây |

Chạy local vẫn như cũ: `JOB_RUNNER` mặc định là `thread`, job chạy trong thread nền
của tiến trình web, không đụng gì tới Modal.

## Chi phí & tài khoản Modal

- Tài khoản sinh viên/học thuật dùng tốt cho dự án này. Modal có chương trình cấp credit cho
  academics (xem `modal.com/academics`), ngoài ra tài khoản mới có credit miễn phí ban đầu.
- Dùng GPU cần thêm phương thức thanh toán, kể cả khi còn credit.
- Bắt buộc đặt `max_containers` và trần chi tiêu **trước** khi mở cho người ngoài dùng — một job lỗi
  lặp lại hoặc một người upload hàng loạt là đủ đốt sạch credit.
- Nếu định dùng cho mục đích ngoài học tập, đọc kỹ điều khoản của chương trình credit học thuật.

## Lộ trình phát triển

### Phase 0 — Chốt an toàn tối thiểu ✅ (đã xong)

- [x] Bắt buộc `FLASK_SECRET_KEY` khi `APP_ENV=production`, fail fast lúc khởi động
- [x] `SESSION_COOKIE_SECURE / HTTPONLY / SAMESITE` + hạn session 7 ngày
- [x] `CSRFProtect` cho toàn bộ form (login, register, logout, upload)
- [x] Error handler trả JSON cho `/api/*`, xử lý 413 quá dung lượng
- [x] `GET /healthz` kiểm tra database + ffmpeg + ffprobe
- [x] Pin toàn bộ `requirements.txt`, thêm `Flask-WTF` và `waitress`

### Phase 1 — Nền dữ liệu ✅ (đã xong)

- [x] Tách `app.py` thành package `app/` với `create_app()` factory và 3 blueprint
- [x] Flask-Migrate + migration baseline (`migrations/versions/e05f1aec0495_*.py`)
- [x] `User`: thêm `role`, `is_active`, `created_at`
- [x] `Job`: thêm `status, progress, message, error, source_filename, file_size, translator_engine, tts_engine, whisper_model, duration_sec, elapsed_sec, segment_count, started_at, finished_at`
- [x] Bỏ dict `JOBS` trong RAM — trạng thái đọc/ghi thẳng từ DB; job dở dang được đánh dấu `interrupted` khi khởi động lại
- [x] CLI `flask create-admin` và `flask set-role`
- [x] Hỗ trợ Postgres qua `DATABASE_URL` (tự đổi scheme `postgres://` → `postgresql+psycopg://`)
- [x] Thêm cột `modal_call_id` vào `Job`

### Phase 2 — Đưa pipeline lên Modal GPU (đang làm)

- [x] `modal_app.py`: image (apt `ffmpeg` + requirements), `modal.Secret`, 2 `modal.Volume`
- [x] Bọc Flask bằng `@modal.wsgi_app()` trên container CPU
- [x] Pipeline thành `@app.cls(gpu="T4")` + `@modal.enter()` nạp model một lần
- [x] Model cache trỏ vào Volume qua `HF_HOME` / `XDG_CACHE_HOME`
- [x] `spawn()` → lưu `modal_call_id`; GPU ghi tiến trình vào Postgres; huỷ bằng `FunctionCall.cancel()`
- [x] `DATA_DIR` cấu hình được để trỏ vào Volume mount
- [x] **Độ trễ:** TTS chạy song song (`ThreadPoolExecutor`, `TTS_CONCURRENCY=8`) — 68,5s → 6,1s
- [x] **Độ trễ:** `faster-whisper` làm backend mặc định, openai-whisper giữ làm dự phòng — nhanh hơn 1,6–1,95×
- [x] `timeout=` cho mọi lệnh ffmpeg/ffprobe qua `utils/proc.py` (`FFMPEG_TIMEOUT`, `FFPROBE_TIMEOUT`)
- [x] Retry có backoff cho batch dịch OpenAI/Gemini (`LLM_MAX_RETRIES`), chỉ thử lại lỗi tạm thời
- [x] **Tự lùi sang engine dịch khác** (DeepSeek → Gemini → OpenAI, engine đầu tiên có key) khi một bên
      hỏng — một lần đổi model hay rate limit không còn làm hỏng cả job. (MarianMT từng là điểm lùi trước khi hỗ trợ đa ngôn ngữ; nay đã bị khoá
      khỏi lựa chọn của người dùng vì chỉ dịch được EN→VI.)
- [x] **Đo thời gian từng bước** (`extract/transcribe/translate/tts/compose_sec`) — thay việc bấm giờ thủ công
- [x] **Vị trí hàng đợi** trong `/api/progress` khi job đang chờ container GPU
- [ ] Đẩy output lên R2 + presigned URL (hiện đang dùng chung Volume, đủ dùng nhưng mọi lượt xem đều qua container web)
- [x] **Đã deploy và kiểm chứng thật** — https://pnam1802--video-dubber-web.modal.run
- [x] Đồng bộ Volume hai chiều: web `commit()` sau khi lưu upload, `reload()` khi phục vụ file kết quả (`app/storage.py`)

Kết quả chạy thật trên production (video 86 giây, Whisper base + MarianMT + edge-tts):

| | GTX 1650 (local) | **T4 (Modal, container ấm)** |
|---|---|---|
| Thời gian pipeline | 63,2s | **13,1s** |
| Tổng kể cả upload | — | 30,0s |

### Phase 3 — API v1 (2 ngày)

- [ ] Blueprint `/api/v1` trả JSON thuần, template Jinja vẫn chạy song song
- [ ] `POST /auth/login|logout|register`, `GET /me`
- [ ] `POST /jobs` (202 + call id), `GET /jobs` phân trang + lọc, `GET /jobs/<id>`, `DELETE`, `POST /jobs/<id>/cancel`
- [ ] Polling có backoff thay cho SSE
- [ ] `GET /config` trả engine/model khả dụng (bỏ hard-code trùng ở `app.py` và `templates/index.html`)
- [ ] `GET /stats` cho dashboard
- [ ] Chuẩn hoá error shape `{error, code, detail}`

### Phase 4 — Phân quyền & hạn mức ✅ (đã xong)

- [x] `require_role()` trong `app/permissions.py`, áp lên mọi endpoint quản trị
- [x] Admin API: `GET /api/admin/users`, `PATCH /api/admin/users/<id>` (đổi role, khoá, đặt hạn mức riêng), `GET /api/admin/stats`
- [x] `GET /api/usage` cho user xem mức tiêu thụ của chính mình
- [x] Quota theo user: số job/ngày, giây GPU/ngày, dung lượng lưu trữ — admin không bị giới hạn
- [x] Kiểm tra hạn mức **trước** khi ghi file, dùng `Content-Length`
- [x] Flask-Limiter cho login / register / upload
- [x] `ProxyFix` để rate limit đếm theo IP thật, không gộp tất cả vào IP của proxy Modal
- [x] `Job.estimated_cost_usd` + `User.quota_jobs_per_day`, migration `07861866bba4`

Hạn mức mặc định (chỉnh bằng biến môi trường):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `QUOTA_JOBS_PER_DAY` | 10 | Số job trong 24 giờ |
| `QUOTA_GPU_SECONDS_PER_DAY` | 1800 | Tổng giây GPU trong 24 giờ |
| `QUOTA_STORAGE_MB` | 2048 | Dung lượng video kết quả đang giữ |
| `GPU_COST_PER_SECOND` | 0.000164 | Đơn giá T4 để ước tính chi phí |

> **`estimated_cost_usd` là cận dưới, không phải hoá đơn.** Nó chỉ tính thời gian
> pipeline chạy. Modal còn tính thời gian container nằm không sau job, cold start,
> CPU, RAM và container web — thực tế cao hơn nhiều lần. Dùng để so sánh giữa user
> và giữa các video; muốn số thật thì chạy `modal environment billing report`.

### Phase 5 — Frontend React (2–3 tuần)

- [ ] Vite + React + TypeScript + TanStack Query, build tĩnh cho Flask serve (cùng origin, không CORS)
- [ ] Luồng auth + route được bảo vệ
- [ ] Upload kéo thả có progress thật (XHR)
- [ ] Trang job: polling, xem trước video, tải bằng presigned URL, huỷ job
- [ ] Lịch sử: lọc, phân trang, xoá
- [ ] Dashboard: job theo ngày, tỉ lệ thành công, thời gian xử lý, engine hay dùng, GPU-giây đã tiêu
- [ ] Trang admin, rồi xoá `templates/` và `static/` cũ

### Phase 6 — Deploy & vận hành (2 ngày)

- [ ] Hai Modal app dev/prod, mỗi bên một bộ Secret
- [ ] `flask db upgrade` chạy như một Modal function trước mỗi lần deploy
- [ ] Custom domain, `min_containers=1` cho web nếu cold start làm phiền
- [ ] `max_containers` + trần chi tiêu + cảnh báo
- [ ] Logging có cấu trúc + request id thay `print()`, gắn Sentry
- [ ] Vòng đời file trên R2, backup Postgres, gỡ 2 file `.mp4` khỏi git

### Phase 7 — Test & CI ✅ (đã xong)

- [x] **399 test pytest** trong `tests/`
- [x] API test với Flask test client, mỗi test một DB SQLite sạch
- [x] Unit test phần lõi không cần Flask: fallback dịch, timeout lệnh ngoài, retry API
- [x] GitHub Actions: chạy migration trên DB trống rồi chạy test, mỗi lần push
- [x] `requirements-dev.txt` đủ để chạy test mà **không cần torch/whisper** — CI xong trong khoảng một phút
- [ ] Smoke test `modal run` với video 5 giây (nightly, tốn GPU nên không chạy mỗi PR)

Chạy test:

	pytest

Bộ test bao gồm cả những trường hợp bảo mật dễ bỏ sót: người dùng khác không xem,
không huỷ, không xoá được job của mình; email Google chưa xác minh không chiếm được
tài khoản; hạn mức chặn trước khi ghi file xuống đĩa.

## Tối ưu độ trễ & số đo

Xếp theo mức cải thiện trên mỗi công sử dụng. Hai mục đầu **không phụ thuộc Modal** — làm được ngay:

| Nút thắt | Cách xử lý | Ghi chú |
|---|---|---|
| ~~Whisper chậm~~ ✅ **đã làm** | `core/transcriber.py` giờ có 2 backend, mặc định `WHISPER_BACKEND=auto` ưu tiên faster-whisper (CUDA `float16`, CPU `int8`), tự lùi về openai-whisper nếu không khởi tạo được | Đo thật **1,6–1,95×** ở bước nhận dạng (không đạt 3–4× như dự án công bố, có thể do GPU 4 GB) |
| ~~TTS gọi tuần tự~~ ✅ **đã làm** | `synthesize_all()` giờ chạy song song bằng `ThreadPoolExecutor`, mặc định 8 luồng (`TTS_CONCURRENCY`) | Đo thật: **68,5s → 6,1s**. Tổng thời gian cả pipeline 114,2s → 49,4s |
| Cold start container GPU | `@modal.enter()` load model + Volume cache; `scaledown_window` dài hơn để container còn ấm giữa các job | Thời gian load model cũng bị tính tiền |
| Cold start web | `min_containers=1` cho container web (CPU rẻ) | Bỏ được độ trễ ở lần truy cập đầu |
| Import + khởi tạo nặng | Bật memory snapshot của Modal cho phần import/nạp weights vào RAM | Kiểm tra tài liệu xem phiên bản hiện tại đã snapshot được state trên GPU chưa |
| Dịch theo batch tuần tự | Chạy song song vài batch dịch (Gemini/OpenAI đều chịu được) | Marian chạy local thì giới hạn ở GPU, không song song được nhiều |

### Số đo thực tế (29.08.2026)

Video 86 giây, Whisper `base`, MarianMT, edge-tts, máy dev: **GTX 1650 laptop, 4 GB VRAM**.

**TTS song song** — đo hai lần liên tiếp trên cùng máy:

| Bước | Tuần tự | Song song 8 luồng |
|---|---|---|
| Whisper | 26,3s | 26,3s |
| Dịch (Marian) | 16,1s | 14,1s |
| TTS (11 segment) | **68,5s** | **6,1s** |
| Ghép ffmpeg | 2,0s | 2,0s |
| **Tổng** | **114,2s** | **49,4s** |

**faster-whisper vs openai-whisper** — chỉ riêng bước nhận dạng, ba lần đo:

| Điều kiện | openai-whisper | faster-whisper | Tỉ lệ |
|---|---|---|---|
| Đo riêng, model vừa tải | 14,2s | 7,3s | 1,95× |
| Đo riêng, model đã cache | 11,1s | 6,4s | 1,73× |
| Trong pipeline, máy đang nóng | 65,9s | 40,5s | 1,63× |

> **Cảnh báo về số liệu:** máy dev là GPU laptop 4 GB và bị throttle nhiệt rõ rệt sau
> vài lần chạy liên tiếp — cùng một cấu hình cho ra 49,4s lúc máy nguội và 125,5s sau
> nhiều lần chạy. Vì vậy **chỉ nên tin các so sánh đo liền kề nhau**, đừng so số tổng
> giữa các lần chạy cách xa nhau. Con số đáng tin cho việc chọn GPU sẽ đến từ Phase 2,
> khi mỗi job chạy trong một container Modal riêng.
>
> Mức cải thiện của TTS song song cũng phụ thuộc độ dài segment: với vài segment rất
> ngắn (~1 giây) thì song song không nhanh hơn, thậm chí chậm hơn chút vì chi phí luồng.

faster-whisper cũng cắt segment khác đi một chút (12 segment thay vì 11 trên cùng video),
nội dung nhận dạng tương đương.

Chọn GPU: Whisper `base`/`small` chạy thoải mái trên **T4** hoặc **L4**. Chỉ lên A10G/A100 khi thật sự
cần `large` — đắt hơn nhiều mà với model nhỏ thì không nhanh hơn tương ứng.

## Demo cũ (MarianMT, trước khi dùng LLM)

Bản lồng tiếng đời đầu: dịch bằng MarianMT (model EN→VI fine-tune riêng), giọng edge-tts đặt
đúng thời điểm từng đoạn Whisper. MarianMT nay đã bị khoá khỏi lựa chọn của người dùng — nó dịch
sai trong im lặng khi nguồn không phải tiếng Anh và không dịch được sang Nhật/Trung/Hàn; code
vẫn giữ để tham khảo.

### Video gốc

https://github.com/user-attachments/assets/458dfa22-e69a-4aa6-b86b-cb6ddb3e952f

### Bản lồng tiếng (MarianMT)

https://github.com/user-attachments/assets/3c90c685-9d6f-4107-a983-cd66ad8e9376

## Ghi chú về video demo trong repo

- Video lớn hơn 100 MB thì dùng Git LFS hoặc đính kèm ở GitHub Releases.
- Muốn phát trực tiếp trong README: kéo thả video vào ô soạn thảo của GitHub để lấy link
  `user-attachments`, đặt link trên một dòng riêng.
- `demo/` không được đóng vào image Modal (xem `SOURCE_IGNORE` trong `modal_app.py`).
