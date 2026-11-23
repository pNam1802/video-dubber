"""
core/studio.py
"Xưởng kịch bản": agent lên ý tưởng và viết kịch bản cho một series video giải thích
(ví dụ "LLM & AI Agent"), để đọc bằng chế độ Video giải thích (core/explainer.py).

Các bước, người dùng duyệt/sửa giữa mỗi bước:
  plan_series   — chủ đề -> tên series, văn phong, lộ trình N tập (mỗi tập: tên, mục tiêu, ý chính).
  research      — Gemini + Google Search: ghi chú kiến thức kèm NGUỒN THẬT cho một tập.
  write_script  — dàn ý + kịch bản tiếng Việt + câu hỏi tự kiểm tra.
  review_script — tự soát: sai kỹ thuật, lặp lại tập trước, quá dài/ngắn, câu khó đọc thành tiếng;
                  trả về danh sách lỗi đã sửa và bản đã sửa.
  translate     — sang tiếng Anh, giữ đúng bảng thuật ngữ.
  wrap_up       — chốt tập: tóm tắt + thuật ngữ mới, ghi vào trí nhớ series.

TRÍ NHỚ: mọi bước viết đều nhận memory_context() — văn phong, bảng thuật ngữ và tóm tắt các
tập đã chốt — nên tập 7 biết tập 3 đã giải thích embedding bằng ví dụ nào, dịch "attention"
ra sao. Trí nhớ nằm trong database (app/models.Series/Episode), không phụ thuộc phiên chat.
Series vài chục tập vẫn vừa ngữ cảnh của Gemini, nên chưa cần tìm kiếm vector trên trí nhớ.

Module này không đụng tới database: nhận/trả dict thuần để test được không cần Flask.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Tuple

from core.script_images import ask_gemini_json

#: Kịch bản ~8-10 phút đọc: tiếng Việt đọc khoảng 150 từ/phút.
TARGET_WORDS = (1100, 1500)
MAX_EPISODES = 30
Source = Dict[str, str]


def _ask_json(prompt: str) -> dict:
    """Gemini, lỗi thì DeepSeek; trả dict. Cả hai lỗi thì ném ra."""
    from core.stock_footage import ask_deepseek_json

    errors = []
    for name, ask in (("Gemini", ask_gemini_json), ("DeepSeek", ask_deepseek_json)):
        try:
            data = json.loads(ask(prompt))
            if isinstance(data, dict):
                return data
            errors.append(f"{name}: không trả về JSON object")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("; ".join(errors))


def ask_gemini_search(prompt: str) -> Tuple[str, List[Source]]:
    """Gemini có công cụ Google Search: trả lời + danh sách nguồn web đã dùng."""
    from config.settings import GEMINI_API_KEY, GEMINI_MODEL

    if not GEMINI_API_KEY:
        raise RuntimeError("chưa có GEMINI_API_KEY")
    from google import genai
    from google.genai import types

    from core.translator.llm_common import call_with_retry

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(temperature=0.3, tools=[types.Tool(google_search=types.GoogleSearch())])
    response = call_with_retry(
        lambda: client.models.generate_content(model=GEMINI_MODEL, config=config, contents=prompt),
        what="Gemini (nghiên cứu)",
    )
    sources: List[Source] = []
    for candidate in response.candidates or []:
        meta = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is not None and getattr(web, "uri", None):
                sources.append({"title": getattr(web, "title", "") or web.uri, "url": web.uri})
    return response.text or "", dedupe_sources(sources)


def dedupe_sources(sources: List[Source]) -> List[Source]:
    seen, out = set(), []
    for src in sources:
        key = src.get("url") or src.get("title")
        if key and key not in seen:
            seen.add(key)
            out.append({"title": str(src.get("title") or key)[:200], "url": str(src.get("url") or "")[:1000]})
    return out


def _strings(value, limit: int = 20, chars: int = 300) -> List[str]:
    if not isinstance(value, list):
        return []
    return [" ".join(str(v).split())[:chars] for v in value if str(v).strip()][:limit]


def _text(value, chars: int = 4000) -> str:
    return str(value or "").strip()[:chars]


# ── Trí nhớ series ───────────────────────────────────────────
def memory_context(series: dict, episodes: List[dict], current_number: int | None = None) -> str:
    """Mọi thứ agent cần nhớ về series, dạng văn bản đưa thẳng vào prompt."""
    lines = [f"SERIES: {series.get('title')}", f"Topic: {series.get('topic')}",
             f"Audience: {series.get('audience')}"]
    if series.get("style_guide"):
        lines.append(f"Style guide:\n{series['style_guide']}")
    glossary = series.get("glossary") or []
    if glossary:
        lines.append("Glossary (ALWAYS use these Vietnamese/English terms):")
        lines += [f"- {g.get('vi')} = {g.get('en')}: {g.get('definition', '')}" for g in glossary]
    lines.append("Roadmap:")
    for ep in episodes:
        mark = "  <- CURRENT" if ep.get("number") == current_number else ""
        lines.append(f"{ep.get('number')}. {ep.get('title')} — {ep.get('goal')}{mark}")
    done = [ep for ep in episodes if ep.get("summary") and ep.get("number") != current_number]
    if done:
        lines.append("What earlier episodes ALREADY covered (do not re-explain in depth, refer back instead):")
        lines += [f"[Ep {ep['number']}] {ep['summary']}" for ep in done]
    return "\n".join(lines)


# ── Các bước ─────────────────────────────────────────────────
def plan_series(topic: str, audience: str, count: int, ask: Callable[[str], dict] = _ask_json) -> dict:
    count = max(1, min(MAX_EPISODES, count))
    data = ask(
        "You plan an educational YouTube series of animated explainer videos (3Blue1Brown style) in "
        "VIETNAMESE, later also published in English. The author's first goal is to LEARN the topic deeply "
        "by teaching it, so the roadmap must build knowledge step by step: each episode relies only on "
        "earlier ones, from fundamentals to advanced, ending with practical applications.\n"
        f"Topic: {topic}\nAudience: {audience or 'Vietnamese developers and students'}\n"
        f"Number of episodes: {count}\n"
        "Return JSON: {\"title\": series title in Vietnamese, \"description\": 2-3 sentences, "
        "\"style_guide\": 4-6 short rules for voice and structure (hook in the first 15 seconds, one concrete "
        "running example, recap at the end...), \"glossary\": [{\"vi\", \"en\", \"definition\"}] for the "
        f"10-20 core terms, \"episodes\": [{{\"title\", \"goal\": one sentence on what the viewer will "
        f"understand, \"key_points\": 3-5 points}}] — exactly {count} episodes}}. All text in Vietnamese "
        "except the English column of the glossary."
    )
    episodes = []
    for raw in data.get("episodes", []) if isinstance(data.get("episodes"), list) else []:
        if isinstance(raw, dict) and str(raw.get("title", "")).strip():
            episodes.append({"title": _text(raw.get("title"), 200), "goal": _text(raw.get("goal"), 500),
                             "key_points": _strings(raw.get("key_points"), 6)})
    if not episodes:
        raise RuntimeError("Lộ trình không có tập nào.")
    return {"title": _text(data.get("title"), 200) or topic[:200], "description": _text(data.get("description"), 1000),
            "style_guide": _text(data.get("style_guide") if isinstance(data.get("style_guide"), str)
                                 else "\n".join(_strings(data.get("style_guide"), 10)), 2000),
            "glossary": clean_glossary(data.get("glossary")), "episodes": episodes[:count]}


def clean_glossary(value) -> List[dict]:
    out = []
    for g in value if isinstance(value, list) else []:
        if isinstance(g, dict) and str(g.get("vi", "")).strip() and str(g.get("en", "")).strip():
            out.append({"vi": _text(g["vi"], 80), "en": _text(g["en"], 80), "definition": _text(g.get("definition"), 300)})
    return out[:200]


def merge_glossary(old: List[dict], new: List[dict]) -> List[dict]:
    """Thêm thuật ngữ mới; thuật ngữ đã có (theo tiếng Anh) giữ nguyên cách dịch cũ."""
    known = {g["en"].lower() for g in old}
    return old + [g for g in new if g["en"].lower() not in known]


def research(context: str, episode: dict,
             search: Callable[[str], Tuple[str, List[Source]]] = ask_gemini_search) -> Tuple[str, List[Source]]:
    notes, sources = search(
        f"{context}\n\n"
        f"Research episode {episode['number']}: \"{episode['title']}\" — {episode['goal']}\n"
        f"Key points: {'; '.join(episode.get('key_points') or [])}\n"
        "Search the web for accurate, current, authoritative information (original papers, official docs, "
        "well-known courses). Write research notes in Vietnamese for the scriptwriter: precise definitions, "
        "how it works step by step, one concrete running example with real numbers, common misconceptions, "
        "and 2-3 interesting facts. Mention the source for each important claim. Do not write the script."
    )
    return _text(notes, 12000), sources


def write_script(context: str, episode: dict, notes: str, ask: Callable[[str], dict] = _ask_json) -> dict:
    lo, hi = TARGET_WORDS
    data = ask(
        f"{context}\n\nResearch notes for this episode:\n{notes}\n\n"
        f"Write episode {episode['number']}: \"{episode['title']}\" — {episode['goal']}\n"
        f"Key points: {'; '.join(episode.get('key_points') or [])}\n"
        f"Rules: Vietnamese narration of {lo}-{hi} words, read aloud by a TTS voice over an animated "
        "explainer, so: plain spoken sentences, no markdown, no bullet symbols, no emoji, no code, spell out "
        "symbols; one idea per paragraph (a paragraph = one line); short sentences. Hook the viewer in the "
        "first two sentences, use ONE running example throughout, briefly refer back to earlier episodes "
        "instead of re-explaining them, end with a recap and a teaser for the next episode. Follow the "
        "glossary exactly. Only state facts supported by the research notes.\n"
        "Return JSON: {\"outline\": [6-10 section titles], \"script\": the full narration with paragraphs "
        "separated by newlines, \"quiz\": [{\"q\", \"a\"}] 3-5 self-check questions with short answers, "
        "\"new_terms\": [{\"vi\", \"en\", \"definition\"}] important terms not in the glossary yet}"
    )
    script = clean_script(data.get("script"))
    if not script:
        raise RuntimeError("Kịch bản rỗng.")
    return {"outline": _strings(data.get("outline"), 12), "script": script, "quiz": clean_quiz(data.get("quiz")),
            "new_terms": clean_glossary(data.get("new_terms"))}


def clean_script(value) -> str:
    """Một đoạn một dòng, bỏ dòng trống và ký hiệu markdown mà LLM hay chèn vào."""
    lines = []
    for line in str(value or "").splitlines():
        line = line.strip().lstrip("#*->•").strip()
        if line:
            lines.append(" ".join(line.split()))
    return "\n".join(lines)[:20000]


def clean_quiz(value) -> List[dict]:
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and str(item.get("q", "")).strip():
            out.append({"q": _text(item["q"], 300), "a": _text(item.get("a"), 600)})
    return out[:8]


def review_script(context: str, episode: dict, notes: str, script: str,
                  ask: Callable[[str], dict] = _ask_json) -> dict:
    """Bước tự soát (một vòng phê bình + sửa). Trả {"issues": [...], "script": bản đã sửa}."""
    lo, hi = TARGET_WORDS
    data = ask(
        f"{context}\n\nResearch notes:\n{notes}\n\nDraft script of episode {episode['number']} "
        f"(\"{episode['title']}\"):\n{script}\n\n"
        "You are a strict technical reviewer and script editor. Check: (1) technical errors or claims not "
        "supported by the notes, (2) content that repeats what earlier episodes already covered, "
        f"(3) length outside {lo}-{hi} words, (4) sentences hard to read aloud (symbols, code, very long "
        "sentences), (5) glossary terms used inconsistently, (6) missing hook, running example or recap. "
        "Fix every problem you find while keeping the author's structure.\n"
        "Return JSON: {\"issues\": [short Vietnamese descriptions of what you fixed], \"script\": the full "
        "corrected script, one paragraph per line}"
    )
    fixed = clean_script(data.get("script"))
    return {"issues": _strings(data.get("issues"), 20), "script": fixed or script}


def translate(context: str, script: str, ask: Callable[[str], dict] = _ask_json) -> str:
    data = ask(
        f"{context}\n\nTranslate this Vietnamese narration into natural spoken English for the English "
        "version of the video. Keep one paragraph per line, the same structure and the running example; "
        "use the glossary's English terms exactly; adapt Vietnamese-specific jokes or phrases naturally.\n"
        f"Script:\n{script}\n\nReturn JSON: {{\"script\": the English narration}}"
    )
    english = clean_script(data.get("script"))
    if not english:
        raise RuntimeError("Bản dịch rỗng.")
    return english


def wrap_up(script: str, ask: Callable[[str], dict] = _ask_json) -> dict:
    data = ask(
        "Summarise this finished episode for the series memory, so later episodes can refer back to it "
        "without repeating it.\n"
        f"Script:\n{script}\n\n"
        "Return JSON: {\"summary\": 3-5 Vietnamese sentences: concepts explained, the running example used, "
        "key analogies, \"new_terms\": [{\"vi\", \"en\", \"definition\"}] terms this episode defined}"
    )
    summary = _text(data.get("summary"), 2000)
    if not summary:
        raise RuntimeError("Tóm tắt rỗng.")
    return {"summary": summary, "new_terms": clean_glossary(data.get("new_terms"))}


def word_count(text: str) -> int:
    return len(str(text or "").split())
