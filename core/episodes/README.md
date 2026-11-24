# Cảnh viết tay cho từng tập

Mỗi tập có thể có hình vẽ riêng thay cho 14 bố cục chung của `core/explainer_scene.py`.

- `<module>_vi.txt`: kịch bản đúng như trong Xưởng (mỗi đoạn một dòng). Số thứ tự câu trong `BOARD` tính theo file này.
- `<module>.py`:
  - `BOARD = [(câu bắt đầu, "tên_hàm"), ...]`
  - mỗi cảnh là một hàm `def tên_hàm(sc, spec, k)`:
    - `sc.at(i)` cho biết lúc câu thứ i của cảnh bắt đầu được đọc;
    - `sc.wait_until(t)` chờ tới mốc `t`; `sc.run(...)` chạy hiệu ứng;
    - `k` gồm màu (`k.ACCENT`, `k.GOLD`...), `k.label(...)`, `k.FRAME_W`.

Mảnh ghép vẽ dùng chung (thẻ, mũi tên, khối code, hình trụ cơ sở dữ liệu...) ở `_kit.py`: `sys.path.insert(0, thư mục này); from _kit import ...`.

Xem trước trên máy (không tốn Modal), mỗi cảnh ra một mp4 và một ảnh 4 khung:

    PYTHONUTF8=1 python scripts/preview_episode.py s1e01 [--only tên_hàm]

Thumbnail: thêm hàm `thumbnail(sc, k, lang)` (vẽ tĩnh, khung 1280x720). Lúc đăng YouTube app tự vẽ và đặt thumbnail.
Xem trước: `python scripts/preview_episode.py s1e01 --thumb --lang en`.

Dùng trong video: `python scripts/preview_episode.py s1e01 --board` in ra bảng phân cảnh dạng
`{"layout": "custom", "module": ..., "scene": ...}`. Dán bảng đó vào Xưởng (hoặc PATCH `storyboard_vi`), deploy, rồi tạo video.
