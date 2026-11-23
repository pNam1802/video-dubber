"""Script của trang Xưởng kịch bản phải parse được: trang dựng hoàn toàn bằng JS, một lỗi cú
pháp (vd xuống dòng thật lọt vào chuỗi) làm trắng cả trang mà server vẫn trả 200."""
from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="cần Node.js")
def test_studio_page_script_parses(as_admin):

    html = as_admin.get("/quan-tri/xuong-kich-ban").get_data(as_text=True)
    scripts = [part.split("</script>")[0] for part in html.split("<script>")[1:]]
    code = "\n;\n".join(scripts)
    result = subprocess.run(["node", "-e", "new Function(require('fs').readFileSync(0, 'utf8'))"],
                            input=code, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr[-800:]
