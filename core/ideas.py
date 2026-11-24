"""
core/ideas.py
"Ý tưởng mỗi ngày" cho Xưởng kịch bản: agent đọc tin trên web (ưu tiên nguồn tiếng Trung, nơi có
nhiều nội dung AI hay nhưng ít người Việt tiếp cận), rồi chọn vài ý tưởng hợp với series.

Hai bước:
  1. search  — Gemini có Google Search viết ghi chú các bài/video đáng chú ý, kèm nguồn web thật.
  2. pick    — một lượt LLM nữa chọn 3-6 ý tưởng hợp lộ trình series, tránh trùng tập đã làm và ý
               tưởng đã gợi ý gần đây. Mỗi ý tưởng chỉ được trỏ tới nguồn trong danh sách của bước 1
               (theo số thứ tự), nên link luôn là link thật, không do LLM tự bịa.

Chỉ lấy KIẾN THỨC và Ý TƯỞNG làm nguồn tham khảo; kịch bản, hình và giọng là của kênh.
Module không đụng database: nhận/trả dict thuần để test được không cần Flask.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

from core.studio import Source, _ask_json, _strings, _text, ask_gemini_search

MAX_IDEAS = 6
#: Chủ đề người dùng gõ khi bấm "Tìm ngay" dài tối đa bấy nhiêu ký tự.
MAX_FOCUS = 200
#: Nguồn tiếng Trung hay viết/làm video về AI, gợi ý cho bước tìm kiếm.
CHINESE_SOURCES = ("机器之心 (jiqizhixin.com)", "量子位 (qbitai.com)", "知乎 Zhihu", "Bilibili B站", "36氪 36kr",
                   "InfoQ 中文", "CSDN", "掘金 juejin.cn")
CHINESE_LABS = ("DeepSeek", "Qwen / Alibaba", "Moonshot Kimi", "Zhipu GLM", "MiniMax", "ByteDance Doubao")


def search_request(topic: str, days: int = 7, focus: str = "") -> str:
    if focus:
        # Chủ đề cụ thể: lấy cả bài giảng giải cũ hơn, không chỉ tin trong tuần.
        scope = (f"Search the web for explanations, tutorials, papers and recent discussion about: {focus} "
                 f"(in the context of {topic}). Recent items are a plus but solid explainers from the last year "
                 "are welcome too. Translate this focus into Chinese search terms and search with them as well. ")
    else:
        scope = (f"Search the web for what was published in the last {days} days about: {topic} "
                 "(LLM agents, generative AI, tool use, memory, planning, multi-agent systems, RAG, evaluation). ")
    return (
        scope +
        "Prioritise CHINESE-language sources and creators — search in Chinese too, e.g. "
        "\"AI Agent 原理\", \"大模型 智能体 教程\", \"Agent 架构 解读\", \"RAG 实战\" — on: "
        f"{', '.join(CHINESE_SOURCES)}; and notable releases, papers or open-source projects from Chinese labs "
        f"({', '.join(CHINESE_LABS)}). Add at most two or three important international items.\n"
        "For each notable item write, in Vietnamese: what it is, the core idea explained simply with the "
        "intuition behind it, and why it matters to someone learning to build agents. Prefer explanations "
        "and teaching content over funding or business news. Mention the source of every item."
    )


def pick_request(context: str, notes: str, sources: List[Source], avoid: List[str], focus: str = "") -> str:
    numbered = "\n".join(f"[{i}] {s.get('title')} — {s.get('url')}" for i, s in enumerate(sources))
    avoided = "\n".join(f"- {t}" for t in avoid) or "(none)"
    return (
        f"{context}\n\nResearch notes from today's web search:\n{notes}\n\nSources (index, title, url):\n{numbered}\n\n"
        f"Ideas already suggested recently (do NOT repeat them):\n{avoided}\n\n"
        + (f"The author asked for this focus: {focus}. Every idea must be about it.\n" if focus else "")
        + "You help the author pick topics for the next educational videos of this series. From the notes, "
        f"choose 3-{MAX_IDEAS} ideas that teach something real (an idea, a technique, how something works), "
        "fit the series audience, and are not already covered by earlier episodes. Use the material as a "
        "source of knowledge only: the video will be the author's own explanation.\n"
        "Return JSON: {\"ideas\": [{\"title\": short Vietnamese video title, \"summary\": 2-4 Vietnamese "
        "sentences explaining the core idea simply, \"angle\": 1-2 Vietnamese sentences on how to teach it "
        "(a concrete example or a visual), \"key_points\": 3-5 short Vietnamese points, \"sources\": list of "
        "source indices from the list above that support it (at least one)}]}"
    )


def clean_ideas(data: dict, sources: List[Source]) -> List[dict]:
    """Giữ ý tưởng có tên và ít nhất một nguồn hợp lệ; nguồn lấy theo chỉ số trong danh sách thật."""
    out = []
    raw = data.get("ideas") if isinstance(data, dict) else None
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not str(item.get("title", "")).strip():
            continue
        picked = []
        for idx in item.get("sources") if isinstance(item.get("sources"), list) else []:
            try:
                src = sources[int(idx)]
            except (TypeError, ValueError, IndexError):
                continue
            if src not in picked:
                picked.append(src)
        if not picked:
            continue
        out.append({"title": _text(item.get("title"), 200), "summary": _text(item.get("summary"), 1500),
                    "angle": _text(item.get("angle"), 800), "key_points": _strings(item.get("key_points"), 6),
                    "sources": picked[:5]})
    return out[:MAX_IDEAS]


def find_ideas(context: str, topic: str, avoid: List[str],
               search: Callable[[str], Tuple[str, List[Source]]] = ask_gemini_search,
               ask: Callable[[str], dict] = _ask_json, focus: str = "") -> List[dict]:
    focus = _text(focus, MAX_FOCUS)
    notes, sources = search(search_request(topic, focus=focus))
    if not sources:
        raise RuntimeError("Tìm kiếm không trả về nguồn nào.")
    return clean_ideas(ask(pick_request(context, _text(notes, 12000), sources, avoid, focus)), sources)
