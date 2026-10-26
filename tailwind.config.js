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
      colors: {
        "background": "#101416",
        "surface": "#101416",
        "surface-dim": "#101416",
        "surface-bright": "#363a3c",
        "surface-container-lowest": "#0b0f11",
        "surface-container-low": "#181c1e",
        "surface-container": "#1c2022",
        "surface-container-high": "#272b2d",
        "surface-container-highest": "#313538",
        "surface-variant": "#313538",
        "on-surface": "#e0e3e5",
        "on-surface-variant": "#c2c9b1",
        "on-background": "#e0e3e5",
        "primary": "#bdfd5d",
        "primary-container": "#a2e043",
        "primary-fixed": "#b6f657",
        "primary-fixed-dim": "#9bd93c",
        "on-primary": "#213600",
        "on-primary-container": "#3e6100",
        "secondary": "#c2c7cb",
        "tertiary": "#ffe4dd",
        "error": "#ffb4ab",
        "on-error": "#690005",
        "error-container": "#93000a",
        "on-error-container": "#ffdad6",
        "outline": "#8c947d",
        "outline-variant": "#424937",
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
        "body-lg": ["Be Vietnam Pro"],
        "body-md": ["Be Vietnam Pro"],
        "label-sm": ["Be Vietnam Pro"],
        "headline-md": ["Be Vietnam Pro"],
        "headline-lg": ["Be Vietnam Pro"],
        "headline-xl": ["Be Vietnam Pro"],
        "data-tabular": ["JetBrains Mono"],
      },
      fontSize: {
        "body-lg": ["16px", { lineHeight: "24px", fontWeight: "400" }],
        "body-md": ["14px", { lineHeight: "20px", fontWeight: "400" }],
        "label-sm": ["12px", { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "500" }],
        "headline-md": ["20px", { lineHeight: "28px", fontWeight: "600" }],
        "headline-lg": ["24px", { lineHeight: "32px", fontWeight: "600" }],
        "headline-xl": ["32px", { lineHeight: "40px", letterSpacing: "-0.02em", fontWeight: "700" }],
        "data-tabular": ["14px", { lineHeight: "20px", fontWeight: "400" }],
      },
    },
  },

  plugins: [require("@tailwindcss/forms")],
};
