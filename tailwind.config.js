/**
 * Cấu hình Tailwind.
 *
 * Trước đây khối này nằm trong thẻ <script> của templates/_head.html và được
 * cdn.tailwindcss.com diễn giải ngay trên trình duyệt — 417 KB JavaScript phải
 * tải về rồi biên dịch CSS lại từ đầu ở MỖI lần mở trang. Nay biên dịch một lần
 * bằng `npm run build:css`, trình duyệt chỉ nhận một file CSS tĩnh.
 *
 * Đổi gì ở đây thì phải chạy lại build và commit static/css/app.css — image của
 * Modal không có Node, nó chỉ chép file đã build vào.
 */
module.exports = {
  darkMode: "class",

  // Tailwind quét đúng những file này để biết class nào được dùng thật. Class
  // ghép chuỗi trong <script> vẫn bắt được vì nó quét văn bản thô, miễn là tên
  // class xuất hiện nguyên vẹn (ví dụ "bg-error" trong classList.replace).
  // landing.html có CSS riêng (xem đầu file đó), bỏ qua để chữ trong trang
  // không vô tình sinh thêm class vào app.css.
  content: ["./templates/**/*.html", "!./templates/landing.html"],

  theme: {
    extend: {
      // Nhóm primary đọc từ biến CSS để người dùng đổi được màu nhấn — bảng
      // màu và các lựa chọn nằm ở templates/_accent.html.
      colors: {
        "background": "#0b0b0b",
        "surface": "#0b0b0b",
        "surface-dim": "#0b0b0b",
        "surface-bright": "#333333",
        "surface-container-lowest": "#070707",
        "surface-container-low": "#111111",
        "surface-container": "#151515",
        "surface-container-high": "#1e1e1e",
        "surface-container-highest": "#2a2a2a",
        "surface-variant": "#2a2a2a",
        "on-surface": "#ececec",
        "on-surface-variant": "#a3a3a3",
        "on-background": "#ececec",
        "primary": "rgb(var(--c-primary) / <alpha-value>)",
        "primary-container": "rgb(var(--c-primary-container) / <alpha-value>)",
        "primary-fixed": "rgb(var(--c-primary-fixed) / <alpha-value>)",
        "primary-fixed-dim": "rgb(var(--c-primary-fixed-dim) / <alpha-value>)",
        "on-primary": "rgb(var(--c-on-primary) / <alpha-value>)",
        "on-primary-container": "rgb(var(--c-on-primary-container) / <alpha-value>)",
        "secondary": "#d1d1d1",
        "tertiary": "#ffe4dd",
        "info": "#8ab4f8",
        "error": "#ffb4ab",
        "on-error": "#690005",
        "error-container": "#93000a",
        "on-error-container": "#ffdad6",
        "outline": "#5d5d5d",
        "outline-variant": "#2a2a2a",
      },
      borderRadius: {
        "md": "5px",
        "lg": "6px",
        "xl": "8px",
        "2xl": "12px",
        "3xl": "16px",
      },
      spacing: {
        "xs": "4px",
        "sm": "8px",
        "md": "16px",
        "gutter": "16px",
        "lg": "24px",
        "xl": "32px",
      },
      fontFamily: {
        "body-lg": ["Inter", "sans-serif"],
        "body-md": ["Inter", "sans-serif"],
        "label-sm": ["Inter", "sans-serif"],
        "headline-md": ["Inter", "sans-serif"],
        "headline-lg": ["Inter", "sans-serif"],
        "headline-xl": ["Inter", "sans-serif"],
        "data-tabular": ["DM Mono", "monospace"],
      },
      fontSize: {
        "body-lg": ["16px", { lineHeight: "24px", fontWeight: "400" }],
        "body-md": ["14px", { lineHeight: "20px", fontWeight: "400" }],
        "label-sm": ["12px", { lineHeight: "16px", letterSpacing: "0", fontWeight: "500" }],
        "headline-md": ["19px", { lineHeight: "28px", letterSpacing: "-0.01em", fontWeight: "500" }],
        "headline-lg": ["24px", { lineHeight: "32px", letterSpacing: "-0.015em", fontWeight: "500" }],
        "headline-xl": ["32px", { lineHeight: "40px", letterSpacing: "-0.025em", fontWeight: "500" }],
        "data-tabular": ["14px", { lineHeight: "20px", fontWeight: "400" }],
      },
    },
  },

  plugins: [require("@tailwindcss/forms")],
};
