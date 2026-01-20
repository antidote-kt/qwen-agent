import json
import re
import os
from flask import Flask, request, jsonify, Response, stream_with_context

import json5

from assistant_video_gen import (
    _try_parse_json_text,
    _extract_shots_from_free_text,
    _ensure_keyframe_and_video_scripts,
    MyKeyframePlanner,
    ExportShotsToJson,
    AssignRefImages,
    MyVideoGen,
    VideoConcatenate,
    init_agent_service,
)
from enhanced_prompt import PromptEnhancer
from video_downloader import (
    process_url_and_call_model,
    call_multimodal_local_video,
    PROMPT_ANALYSIS_TEMPLATE,
)
from t2i import BatchStoryboardPainter
from music_gen import MusicPromptRefiner, MyMusicGen

app = Flask(__name__)

# Lazy init to avoid repeated model/tool loading
_agent_bot = None
_session_memory = {}
_MAX_SESSION_MESSAGES = 50


def _get_agent():
    global _agent_bot
    if _agent_bot is None:
        _agent_bot = init_agent_service()
    return _agent_bot


def _json_response(status: str, data=None, message: str = ""):
    return jsonify({"status": status, "data": data, "message": message})


def _parse_enhanced_text(raw):
    """Best-effort parse for prompt_enhance output.

    Handles cases where upstream returns JSON-ish text without quotes around keys.
    Falls back to returning the raw string under {'raw': raw}.
    """

    if raw is None:
        return None

    # If already dict, return as-is
    if isinstance(raw, dict):
        return raw

    # Non-string, wrap as raw
    if not isinstance(raw, str):
        return {"raw": raw}

    text = raw.strip()
    if not text:
        return {"raw": raw}

    # 1) strict JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) JSON5 tolerant parse
    try:
        return json5.loads(text)
    except Exception:
        pass

    # 3) Heuristic: add quotes to keys like enhanced_prompt: xxx
    try:
        fixed = re.sub(r'([,{]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', text)
        return json5.loads(fixed)
    except Exception:
        return {"raw": raw}


def _normalize_messages_for_agent(messages):
    """Normalize incoming messages to qwen_agent schema.

    qwen_agent expects each message as {role, content}, where content is either a string
    or a list of dict items compatible with ContentItem(**item) and each item contains
    exactly one of: text/image/file/audio/video.

    This API also accepts an OpenAI-like format where each content item is:
      {"type": "text", "text": "..."}
      {"type": "image_url", "image_url": {"url": "http(s)://..." or "file:///..."}}
      {"type": "video_url", "video_url": {"url": "http(s)://..." or "file:///..."}}
    """

    if not isinstance(messages, list):
        return []

    def _convert_item(item):
        if not isinstance(item, dict):
            return {"text": str(item)}

        # Already qwen_agent style?
        for k in ("text", "image", "file", "audio", "video"):
            if k in item and item.get(k) is not None:
                return {k: item.get(k)}

        t = item.get("type")
        if not t:
            # fallback: treat as text-ish
            if "text" in item:
                return {"text": str(item.get("text") or "")}
            return {"text": json.dumps(item, ensure_ascii=False)}

        t = str(t)
        if t == "text":
            return {"text": str(item.get("text") or "")}
        if t in ("image", "image_url"):
            payload = item.get("image_url") if t == "image_url" else item
            url = None
            if isinstance(payload, dict):
                url = payload.get("url") or payload.get("image")
            if url is None:
                url = item.get("url")
            return {"image": url}
        if t in ("video", "video_url"):
            payload = item.get("video_url") if t == "video_url" else item
            url = None
            if isinstance(payload, dict):
                url = payload.get("url") or payload.get("video")
            if url is None:
                url = item.get("url")
            return {"video": url}
        if t in ("file", "file_url"):
            payload = item.get("file_url") if t == "file_url" else item
            url = None
            if isinstance(payload, dict):
                url = payload.get("url") or payload.get("file")
            if url is None:
                url = item.get("url")
            return {"file": url}
        if t in ("audio", "audio_url"):
            payload = item.get("audio_url") if t == "audio_url" else item
            url = None
            if isinstance(payload, dict):
                url = payload.get("url") or payload.get("audio")
            if url is None:
                url = item.get("url")
            return {"audio": url}

        # Unknown type -> keep as text to avoid hard failure
        return {"text": json.dumps(item, ensure_ascii=False)}

    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            norm = []
            for it in content:
                ci = _convert_item(it)
                # drop invalid empty items
                if isinstance(ci, dict) and len(ci) == 1 and list(ci.values())[0] is not None:
                    norm.append(ci)
            out.append({"role": role, "content": norm})
        else:
            out.append({"role": role, "content": content})
    return out


@app.route("/analyze_video_url", methods=["POST"])
def analyze_video_url():
    body = request.get_json(force=True, silent=True) or {}
    url = body.get("url")
    fps = int(body.get("fps", 2) or 2)
    if not url:
        return _json_response("error", None, "url is required")
    try:
        result = process_url_and_call_model(url, fps=fps)
        parsed = _try_parse_json_text(result)
        if not isinstance(parsed, dict):
            shots = _extract_shots_from_free_text(result)
            if shots:
                parsed = {"detailed_script": shots}
        if isinstance(parsed, dict):
            parsed = _ensure_keyframe_and_video_scripts(parsed)
        return _json_response("success", {"raw": result, "analysis": parsed})
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/analyze_video_local", methods=["POST"])
def analyze_video_local():
    body = request.get_json(force=True, silent=True) or {}
    file_path = body.get("file_path")
    fps = int(body.get("fps", 2) or 2)
    if not file_path:
        return _json_response("error", None, "file_path is required")
    try:
        result = call_multimodal_local_video(
            file_path,
            text=PROMPT_ANALYSIS_TEMPLATE,
            fps=fps,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen3-vl-plus",
        )
        parsed = _try_parse_json_text(result)
        if not isinstance(parsed, dict):
            shots = _extract_shots_from_free_text(result)
            if shots:
                parsed = {"detailed_script": shots}
        if isinstance(parsed, dict):
            parsed = _ensure_keyframe_and_video_scripts(parsed)
        return _json_response("success", {"raw": result, "analysis": parsed})
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/prompt_enhance", methods=["POST"])
def prompt_enhance():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tool = PromptEnhancer()
        res = tool.call(json.dumps(body))
        parsed = _parse_enhanced_text(res)
        return _json_response("success", parsed)
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/plan_shots", methods=["POST"])
def plan_shots():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tool = MyKeyframePlanner()
        res = tool.call(json.dumps(body))
        return _json_response("success", json.loads(res))
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/export_shots", methods=["POST"])
def export_shots():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tool = ExportShotsToJson()
        res = tool.call(json.dumps(body))
        return _json_response("success", json.loads(res))
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/assign_refs", methods=["POST"])
def assign_refs():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tool = AssignRefImages()
        res = tool.call(json.dumps(body))
        return _json_response("success", json.loads(res))
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/paint_storyboard", methods=["POST"])
def paint_storyboard():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tool = BatchStoryboardPainter()
        res = tool.call(json.dumps(body))
        # call 返回通常是 JSON 字符串
        try:
            parsed = json.loads(res)
        except Exception:
            parsed = res
        return _json_response("success", parsed)
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/gen_music", methods=["POST"])
def gen_music():
    body = request.get_json(force=True, silent=True) or {}
    try:
        refine_tool = MusicPromptRefiner()
        refine_res = refine_tool.call(json.dumps(body))
        try:
            refine = json.loads(refine_res)
        except Exception:
            refine = refine_res

        music_tool = MyMusicGen()
        music_res = music_tool.call(json.dumps(refine if isinstance(refine, dict) else body))
        try:
            music = json.loads(music_res)
        except Exception:
            music = music_res

        return _json_response("success", {"refined": refine, "music": music})
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/gen_video", methods=["POST"])
def gen_video():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tool = MyVideoGen()
        res = tool.call(json.dumps(body))
        return _json_response("success", json.loads(res))
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/concat", methods=["POST"])
def concat_videos():
    body = request.get_json(force=True, silent=True) or {}
    try:
        tool = VideoConcatenate()
        res = tool.call(json.dumps(body))
        return _json_response("success", json.loads(res))
    except Exception as e:
        return _json_response("error", None, str(e))


@app.route("/health", methods=["GET", "POST"])
def health():
    return _json_response("ok", {"message": "healthy"})


@app.route("/agent_chat", methods=["POST"])
def agent_chat():
    """统一多模态消息入口，转给视频 Agent（Assistant）。

    请求体示例：
    {
        "session_id": "可选",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述/需求/指令"},
                    {"type": "image_url", "image_url": {"url": "http://... 或 file:///..."}},
                    {"type": "video_url", "video_url": {"url": "http://... 或 file:///..."}}
                ]
            }
        ]
    }
    返回：{status, data: {replies, session_id}, message}
    """

    body = request.get_json(force=True, silent=True) or {}
    messages = body.get("messages") or []
    session_id = body.get("session_id")

    stream = body.get("stream")
    if stream is None:
        stream = request.args.get("stream")
    stream = str(stream).lower() in ("1", "true", "yes", "y", "on")

    if not messages:
        return _json_response("error", None, "messages is required")

    # Normalize message schema for qwen_agent
    messages = _normalize_messages_for_agent(messages)

    # Merge历史+本次消息；简单内存会话（非持久）
    merged = messages
    if session_id:
        history = _session_memory.get(session_id, [])
        merged = history + messages
        if len(merged) > _MAX_SESSION_MESSAGES:
            merged = merged[-_MAX_SESSION_MESSAGES:]

    try:
        bot = _get_agent()

        if not stream:
            replies = bot.run_nonstream(messages=merged, session_id=session_id)
            if session_id:
                new_hist = merged + replies
                if len(new_hist) > _MAX_SESSION_MESSAGES:
                    new_hist = new_hist[-_MAX_SESSION_MESSAGES:]
                _session_memory[session_id] = new_hist
            return _json_response("success", {"replies": replies, "session_id": session_id})

        def _find_last_assistant_text(replies_chunk):
            if not isinstance(replies_chunk, list):
                return ""
            for m in reversed(replies_chunk):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, str):
                        return c
                    if c is None:
                        return ""
                    return str(c)
            return ""

        @stream_with_context
        def generate_sse():
            last_full_text = ""
            final_replies = None
            try:
                # Send a meta event first so the client can confirm the stream opened.
                yield "event: meta\n" + "data: " + json.dumps(
                    {"status": "start", "session_id": session_id}, ensure_ascii=False
                ) + "\n\n"

                for replies_chunk in bot.run(messages=merged, session_id=session_id):
                    final_replies = replies_chunk
                    full_text = _find_last_assistant_text(replies_chunk)
                    if full_text.startswith(last_full_text):
                        delta = full_text[len(last_full_text) :]
                    else:
                        delta = full_text
                    last_full_text = full_text

                    payload = {
                        "type": "delta",
                        "delta": delta,
                        "content": full_text,
                        "replies": replies_chunk,
                        "session_id": session_id,
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

                # Update memory with the final replies only (keep history small).
                if session_id and isinstance(final_replies, list):
                    new_hist = merged + final_replies
                    if len(new_hist) > _MAX_SESSION_MESSAGES:
                        new_hist = new_hist[-_MAX_SESSION_MESSAGES:]
                    _session_memory[session_id] = new_hist

                yield "event: done\n" + "data: " + json.dumps(
                    {"status": "done", "session_id": session_id}, ensure_ascii=False
                ) + "\n\n"
            except GeneratorExit:
                # Client disconnected.
                return
            except Exception as e:
                yield "event: error\n" + "data: " + json.dumps(
                    {"status": "error", "message": str(e), "session_id": session_id}, ensure_ascii=False
                ) + "\n\n"

        resp = Response(generate_sse(), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp
    except Exception as e:
        return _json_response("error", None, str(e))


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000") or 8000)
    app.run(host="0.0.0.0", port=port, debug=True)
