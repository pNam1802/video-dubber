"""
app/legal.py
Chính sách bảo mật và điều khoản sử dụng.

Google bắt buộc phải có URL chính sách bảo mật thì mới cho chuyển ứng dụng
OAuth sang chế độ production. Nội dung dưới đây mô tả đúng những gì hệ thống
thực sự làm — sai một chỗ là vừa vô ích vừa gây hiểu nhầm.
"""
from __future__ import annotations

UPDATED = "31 tháng 8, 2026"

PRIVACY = [
    {
        "heading": "Chúng tôi thu thập gì",
        "paragraphs": [
            "Đây là đồ án tốt nghiệp, không phải dịch vụ thương mại. Hệ thống chỉ "
            "lưu những dữ liệu cần thiết để chạy được việc lồng tiếng.",
        ],
        "bullets": [
            "Tài khoản: tên đăng nhập, và mật khẩu đã băm nếu bạn đăng ký bằng mật khẩu.",
            "Nếu đăng nhập bằng Google: địa chỉ email, tên hiển thị và ảnh đại diện. "
            "Chúng tôi không xin quyền truy cập Gmail, Drive hay bất kỳ dữ liệu nào khác.",
            "Video bạn tải lên, cùng file âm thanh và phụ đề sinh ra từ video đó.",
            "Thông tin về mỗi lần xử lý: thời lượng video, công cụ dịch đã dùng, "
            "thời gian xử lý từng bước, trạng thái thành công hay thất bại.",
        ],
    },
    {
        "heading": "Dữ liệu được gửi đi đâu",
        "paragraphs": [
            "Việc lồng tiếng cần gọi tới một vài dịch vụ bên ngoài. Bạn nên biết "
            "chính xác phần nào của video rời khỏi hệ thống:",
        ],
        "bullets": [
            "Nhận dạng giọng nói chạy bằng Whisper ngay trên máy chủ GPU của chúng tôi. "
            "Âm thanh không gửi đi đâu cả.",
            "Việc dịch dùng Google Gemini hoặc OpenAI tuỳ bạn chọn, nên phần lời thoại đã "
            "nhận dạng được gửi tới dịch vụ đó để dịch.",
            "Việc tổng hợp giọng nói tiếng Việt dùng dịch vụ của Microsoft (edge-tts) hoặc "
            "Google (gTTS), nên câu tiếng Việt đã dịch được gửi tới đó để tạo giọng đọc.",
        ],
    },
    {
        "heading": "Dữ liệu lưu ở đâu",
        "paragraphs": [
            "Máy chủ xử lý đặt tại Hoa Kỳ (Modal), cơ sở dữ liệu đặt tại Hoa Kỳ (Neon). "
            "Video và phụ đề nằm trên ổ lưu trữ của Modal, chỉ tài khoản đã tạo ra chúng "
            "mới xem được — hệ thống kiểm tra quyền sở hữu ở từng lượt tải.",
        ],
    },
    {
        "heading": "Giữ trong bao lâu",
        "paragraphs": [
            "Video và phụ đề được giữ cho tới khi bạn xoá, hoặc tới khi đồ án kết thúc "
            "và hệ thống ngừng hoạt động. Chúng tôi không dùng video của bạn để huấn "
            "luyện mô hình, không phân tích nội dung, không chia sẻ cho bên thứ ba nào "
            "ngoài các dịch vụ dịch và tổng hợp giọng nói đã nêu ở trên.",
        ],
    },
    {
        "heading": "Quyền của bạn",
        "paragraphs": [
            "Bạn có thể yêu cầu xoá tài khoản và toàn bộ dữ liệu bất cứ lúc nào bằng "
            "cách liên hệ theo địa chỉ ở cuối trang. Nếu đăng nhập bằng Google, bạn có "
            "thể thu hồi quyền truy cập tại trang Tài khoản Google của mình, mục "
            "Bảo mật, phần Ứng dụng của bên thứ ba.",
        ],
    },
    {
        "heading": "Liên hệ",
        "paragraphs": [
            "Mọi câu hỏi về dữ liệu, gửi tới nguyenphuongnam22114@gmail.com.",
        ],
    },
]

TERMS = [
    {
        "heading": "Đây là gì",
        "paragraphs": [
            "Nora là công cụ lồng tiếng Việt cho video bài giảng tiếng Anh, "
            "được xây dựng trong khuôn khổ một đồ án tốt nghiệp. Hệ thống cung cấp "
            "miễn phí, không cam kết luôn sẵn sàng, và có thể ngừng hoạt động khi "
            "đồ án kết thúc.",
        ],
    },
    {
        "heading": "Bạn chịu trách nhiệm về video mình tải lên",
        "paragraphs": [
            "Chỉ tải lên video mà bạn có quyền sử dụng. Đừng tải nội dung vi phạm bản "
            "quyền của người khác, nội dung riêng tư của người khác, hay nội dung trái "
            "pháp luật.",
        ],
    },
    {
        "heading": "Hạn mức",
        "paragraphs": [
            "Mỗi tài khoản có hạn mức số lượt xử lý, thời gian GPU và dung lượng lưu "
            "trữ trong 24 giờ. Mỗi lần bấm bắt đầu đều tính một lượt, kể cả khi job "
            "thất bại. Hạn mức tồn tại vì mỗi lần chạy đều tốn chi phí máy chủ thật.",
        ],
    },
    {
        "heading": "Chất lượng bản dịch",
        "paragraphs": [
            "Bản dịch và giọng đọc do máy tạo ra, có thể sai — nhất là với thuật ngữ "
            "chuyên ngành, tên riêng và số liệu. Hãy tự kiểm tra lại trước khi dùng "
            "cho việc quan trọng. Chúng tôi không chịu trách nhiệm cho hậu quả của "
            "bản dịch sai.",
        ],
    },
]
