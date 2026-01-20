"""A simple multi-step video generation agent demo.

Workflow:
1) Help the user refine / expand a detailed English video prompt.
2) Generate a sequence of keyframes (a storyboard) from the refined prompt.
3) Based on the refined prompt and keyframes, call a custom video generation tool to get a video URL.

The actual video generation backend is abstracted as `my_keyframe_planner` and `my_video_gen`.
You can replace the implementation of their ``call`` methods with your real API calls.
"""

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from http import HTTPStatus

import dashscope
from dashscope.aigc.image_synthesis import ImageSynthesis

# Make sure DashScope key is available for all downstream tools/modules.
# 优先级：配置文件 > 环境变量
try:
    _dk = (getattr(dashscope, "api_key", None) or "").strip()
except Exception:
    _dk = ""
if not _dk:
    try:
        from config_loader import get_dashscope_api_key
        dashscope.api_key = get_dashscope_api_key()
    except (ImportError, FileNotFoundError, ValueError, KeyError):
        # 如果配置文件不存在或读取失败，回退到环境变量
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

from video_generator import sample_async_call
from video_downloader import (
    process_url_and_call_model,
    call_multimodal_local_video,
    PROMPT_ANALYSIS_TEMPLATE,
)
from video_scorer import score_video
# 导入 video_cut 以便注册 `video_editor` 工具（装饰器在导入时生效）
import video_cut  # noqa: F401
# 导入 enhanced_prompt 以便注册 `my_PromptEnhancer` 提示词增强工具
import enhanced_prompt  # noqa: F401
from enhanced_prompt import image_file_to_data_uri
import t2i  # noqa: F401
# 导入 music_gen 以便注册 `music_prompt_refiner` / `my_music_gen` 工具
import music_gen  # noqa: F401

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")

def _try_parse_json_text(s: str):
    if not isinstance(s, str):
        return None
    s2 = s.strip()
    if not s2:
        return None
    if s2.startswith("ERROR_JSON_PARSE"):
        return None
    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_shots_from_free_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    out: list[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 1) Prefer explicit "镜头X:" / "Shot X:" lines.
    for line in lines:
        m = re.match(r"^\s*(?:镜头|Shot)\s*([0-9]{1,3})\s*[:：]\s*(.+?)\s*$", line)
        if not m:
            continue
        sid = m.group(1).strip()
        desc = m.group(2).strip()
        if desc:
            out.append(f"{sid}. {desc}")
    if out:
        return out

    # 2) Fallback: treat each non-empty line that looks like a shot as one item.
    # Common pattern: contains "建议时长" and "镜头" keywords in parentheses.
    shot_like = [ln for ln in lines if ("建议时长" in ln) or ("时长" in ln and "镜头" in ln)]
    if len(shot_like) >= 2:
        return [f"{i+1}. {ln}" for i, ln in enumerate(shot_like)]

    return []


def _ensure_keyframe_and_video_scripts(analysis: dict) -> dict:
    """Ensure analysis dict contains both keyframe/video variants.

    Fallbacks:
    - If keyframe/video lists are missing, derive them from `detailed_script`.
    """
    if not isinstance(analysis, dict):
        return analysis
    if "detailed_script_keyframe" in analysis and "detailed_script_video" in analysis:
        return analysis
    detailed = analysis.get("detailed_script")
    if not isinstance(detailed, list):
        return analysis

    def _kf_line(x):
        t = str(x).strip()
        if not t:
            return t
        return (
            t
            + "（关键帧版：定格单帧画面；强调人物外观一致性、数量约束、环境/光照、构图与关键道具；弱化连续动作过程）"
        )

    def _vid_line(x):
        t = str(x).strip()
        if not t:
            return t
        return (
            t
            + "（视频分镜版：强调动作推进（起势→过程→结果）与镜头运动；严格遵守数量约束与环境一致性；不加额外角色/字幕水印）"
        )

    analysis["detailed_script_keyframe"] = [_kf_line(x) for x in detailed]
    analysis["detailed_script_video"] = [_vid_line(x) for x in detailed]
    return analysis


def _download_http_file(url: str, dst_path: str, max_bytes: int) -> int:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )
    total = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(dst_path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"download too large: {total} bytes > {max_bytes} bytes")
            f.write(chunk)
    return total


@register_tool("video_script_analyzer")
class VideoScriptAnalyzer(BaseTool):
    """A tool that downloads a reference video from URL and analyzes it
    with a multimodal model to produce a structured script / suggestions.

    It internally reuses `process_url_and_call_model` defined in
    `video_downloader.py` and returns the raw text result (usually a JSON
    string following PROMPT_ANALYSIS_TEMPLATE), wrapped in a small status
    object so the agent can further refine the prompt.
    """

    description = (
        "Analyze a reference short video given by URL using a multimodal "
        "model, and return a JSON-like analysis including script summary, "
        "detailed shots, key scenes, and adaptation suggestions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "description": "The URL of the reference video to analyze.",
                "type": "string",
            },
            "fps": {
                "description": "Frame sampling rate for video analysis (default 2).",
                "type": "integer",
                "default": 2,
            },
        },
        "required": ["url"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        url = args["url"]
        fps = int(args.get("fps", 2) or 2)

        try:
            # 复用 video_downloader 的端到端流程：下载 -> video.mp4 -> 多模态分析
            result = process_url_and_call_model(url, fps=fps)
            if result is None:
                # 通常是缺少 API Key 或分析失败
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Video analysis failed or API key missing.",
                        "result": None,
                    },
                    ensure_ascii=False,
                )

            parsed = _try_parse_json_text(result)
            if not isinstance(parsed, dict):
                extracted = _extract_shots_from_free_text(result)
                if extracted:
                    parsed = {"detailed_script": extracted}
            if isinstance(parsed, dict):
                parsed = _ensure_keyframe_and_video_scripts(parsed)

            return json.dumps(
                {
                    "status": "success",
                    "result": result,
                    "analysis": parsed,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Exception during video analysis: {e}",
                    "result": None,
                },
                ensure_ascii=False,
            )


@register_tool("local_video_script_analyzer")
class LocalVideoScriptAnalyzer(BaseTool):
    """Analyze a locally uploaded reference video file using a multimodal model.

    This tool is intended for cases where the user uploads a video file via
    the Web UI (file:// path) instead of providing a public URL.
    """

    description = (
        "Analyze a locally uploaded short video file using a multimodal "
        "model, and return a JSON-like analysis including script summary, "
        "detailed shots, key scenes, and adaptation suggestions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "description": (
                    "The local path or file:// URI of the reference video "
                    "to analyze (as provided by the Web UI)."
                ),
                "type": "string",
            },
            "fps": {
                "description": "Frame sampling rate for video analysis (default 2).",
                "type": "integer",
                "default": 2,
            },
        },
        "required": ["file_path"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        raw_path = args["file_path"]
        fps = int(args.get("fps", 2) or 2)

        # 支持 file:// 开头的 URI，也支持普通本地路径
        if isinstance(raw_path, str) and raw_path.startswith("file://"):
            local_path = raw_path[len("file://") :]
        else:
            local_path = raw_path

        try:
            # 优先从配置文件读取 API Key
            try:
                from config_loader import get_dashscope_api_key
                api_key = get_dashscope_api_key()
            except (ImportError, FileNotFoundError, ValueError, KeyError):
                api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
            
            result = call_multimodal_local_video(
                local_path,
                text=PROMPT_ANALYSIS_TEMPLATE,
                fps=fps,
                api_key=api_key,
                model="qwen3-vl-plus",
            )
            if not result:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Video analysis returned empty result.",
                        "result": None,
                    },
                    ensure_ascii=False,
                )

            parsed = _try_parse_json_text(result)
            if not isinstance(parsed, dict):
                extracted = _extract_shots_from_free_text(result)
                if extracted:
                    parsed = {"detailed_script": extracted}
            if isinstance(parsed, dict):
                parsed = _ensure_keyframe_and_video_scripts(parsed)

            return json.dumps(
                {
                    "status": "success",
                    "result": result,
                    "analysis": parsed,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Exception during local video analysis: {e}",
                    "result": None,
                },
                ensure_ascii=False,
            )


@register_tool("my_keyframe_planner")
class MyKeyframePlanner(BaseTool):
    """A custom tool for shot planning and keyframe generation.

    It takes a global video prompt, splits it into several shots
    (storyboard segments), and for each shot generates a keyframe
    image as the first frame reference.
    """

    description = (
        "Shot planning service. It takes a video prompt (CN/EN) and returns a JSON list of shots. "
        "Each shot includes: shot_prompt (dynamic, for video), keyframe_prompt (static first frame, for image), "
        "duration, and id."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "description": (
                    "A detailed description of the video idea (Chinese/English is OK), including scene, "
                    "characters, style, and overall story beats."
                ),
                "type": "string",
            },
            "num_shots": {
                "description": (
                    "How many shots / storyboard segments to design. "
                    "If omitted, defaults to 3. Typically between 1 and 10."
                ),
                "type": "integer",
                "default": 3,
            },
        },
        "required": ["prompt"],
    }

    def call(self, params: str, **kwargs) -> str:
        # Parse JSON arguments
        args = json.loads(params)
        prompt = args["prompt"]
        # Allow caller / agent to control how many shots to design
        try:
            num_shots = int(args.get("num_shots", 3) or 3)
        except Exception:
            num_shots = 3
        # Clamp to a reasonable range
        if num_shots < 1:
            num_shots = 1
        if num_shots > 10:
            num_shots = 10

        # Ensure DashScope API key is set
        if not dashscope.api_key:
            try:
                from config_loader import get_dashscope_api_key
                dashscope.api_key = get_dashscope_api_key()
            except (ImportError, FileNotFoundError, ValueError, KeyError):
                dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

        # Call DashScope text generation to plan shots.
        # We ask the model to directly output a JSON array of shots.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional film director. "
                    f"Given a video prompt, you must design exactly {num_shots} shots (storyboard segments). "
                    "Output ONLY a valid JSON array, no extra text. "
                    "Hard requirements for EACH shot object:"
                    "\n- id: integer starting from 1"
                    "\n- duration: integer seconds (>0)"
                    "\n- shot_prompt: a DYNAMIC shot description focusing on visual changes over time, actions, and camera movement (for video generation)"
                    "\n- keyframe_prompt: a STATIC first-frame description (single moment), describing ONLY what is visible in one still image (for keyframe image generation)"
                    "\nGlobal constraints:"
                    "\n- NO subtitles/captions/on-screen text/overlays/watermarks/logos/UI elements."
                    "\n- NO voiceover/narration/dialogue instructions (audio handled separately)."
                    "\n- IMPORTANT: keyframe_prompt MUST NOT mention music/audio/BGM/soundtrack/ambient sound at all."
                    "\n  (Music may be mentioned in shot_prompt only if needed.)"
                    "\nJSON schema: [ {""id"": <int>, ""duration"": <int>, ""shot_prompt"": <string>, ""keyframe_prompt"": <string>}, ... ]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Design exactly {num_shots} shots for the following video idea, "
                    "covering the beginning, middle, and end of the story. "
                    "The total video duration is about 10-15 seconds.\n\n" + prompt
                ),
            },
        ]

        response = dashscope.Generation.call(
            "qwen-max",
            messages=messages,
            result_format="message",
            stream=False,
        )

        if response.status_code != HTTPStatus.OK:
            # You can customize error handling here.
            raise RuntimeError(f"DashScope keyframe planning failed: {response.code} {response.message}")

        content = response.output.choices[0].message.content

        # The model is instructed to return a pure JSON array of shots.
        try:
            shots = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: wrap raw content as a single-entry shot.
            shots = [
                {
                    "id": 1,
                    "shot_prompt": content,
                    "keyframe_prompt": content,
                    "description": content,
                    "duration": 10,
                }
            ]

        # Normalize fields for downstream compatibility
        if isinstance(shots, list):
            def _strip_audio_lines(text: str) -> str:
                if not isinstance(text, str) or not text:
                    return ""
                lines = [ln.strip() for ln in text.splitlines()]
                drop_keywords = (
                    "背景音乐",
                    "配乐",
                    "音乐",
                    "bgm",
                    "soundtrack",
                    "music",
                    "audio",
                    "ambient sound",
                    "环境音",
                    "音轨",
                )
                kept: list[str] = []
                for ln in lines:
                    if not ln:
                        continue
                    lnl = ln.lower()
                    if any(k in ln for k in drop_keywords) or any(k in lnl for k in drop_keywords):
                        continue
                    kept.append(ln)
                cleaned = "\n".join(kept).strip()
                return cleaned or text.strip()

            for s in shots:
                if not isinstance(s, dict):
                    continue
                sp = s.get("shot_prompt") or s.get("description") or s.get("prompt")
                kp = s.get("keyframe_prompt") or s.get("t2i_prompt")
                if sp is not None:
                    s["shot_prompt"] = str(sp)
                    # Back-compat: keep "description" as dynamic shot prompt
                    s["description"] = str(sp)
                if kp is not None:
                    # Keyframe prompt must not mention music/audio.
                    s["keyframe_prompt"] = _strip_audio_lines(str(kp))
                    # Back-compat for older exporters/painters
                    if "t2i_prompt" not in s:
                        s["t2i_prompt"] = s["keyframe_prompt"]

        # Only keep the first `num_shots` shots at most
        if isinstance(shots, list) and len(shots) > num_shots:
            shots = shots[:num_shots]

        # Return pure shot metadata; image generation is handled by
        # downstream tools such as `batch_storyboard_painter`.
        return json.dumps({"shots": shots}, ensure_ascii=False)


@register_tool("export_shots_to_json")
class ExportShotsToJson(BaseTool):
    """Export a list of shots into a JSON file compatible with
    `batch_storyboard_painter` in t2i.py.

    This tool is intended to be called after `my_keyframe_planner`.
    """

    description = (
        "Export a list of shots into a prompts JSON file that can be "
        "consumed by the `batch_storyboard_painter` tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "shots": {
                "description": (
                    "The shots to export. Can be either a JSON string or a "
                    "list/dict matching the output of `my_keyframe_planner`."
                ),
                "type": "string",
            },
            "global_prompt": {
                "description": (
                    "Optional refined global video prompt. When provided, it "
                    "will be prepended to each per-shot description so that "
                    "the painter can see the full context (characters, style, "
                    "story background, etc.)."
                ),
                "type": "string",
            },
            "prepend_global_prompt": {
                "description": (
                    "Whether to prepend `global_prompt` to each per-shot prompt. "
                    "Default false so the prompt sent to the painter matches the user-confirmed shot text exactly."
                ),
                "type": "boolean",
                "default": False,
            },
            "append_duration": {
                "description": (
                    "Whether to append the shot duration to each prompt (e.g., '时长：3秒'). "
                    "Default true for consistency with user-visible shot plans."
                ),
                "type": "boolean",
                "default": True,
            },
            "output_dir": {
                "description": (
                    "Optional output directory for the prompts JSON file. "
                    "Defaults to ./storyboard_output."
                ),
                "type": "string",
                "default": "./storyboard_output",
            },
            "ref_image_dir": {
                "description": (
                    "Optional reference image directory. If provided, each exported shot will be marked as ref=true so the painter can use reference images."
                ),
                "type": "string",
            },
            "ref_shot_ids": {
                "description": (
                    "Optional shot ids that should use the provided reference image(s), e.g. \"01,02\". "
                    "If omitted and only a single reference image is provided, it will be applied to the first shot only."
                ),
                "type": "string",
            },
            "ref_label": {
                "description": (
                    "Optional role/entity name that the provided reference image corresponds to (e.g. \"猴子\", \"Bobo\"). "
                    "When provided together with `ref_image_dir` (a single image file, or a directory containing only 1 image), "
                    "the tool will automatically enable ref=true for any shot whose text mentions this label (or its aliases), "
                    "so that all shots containing that character will consistently use the same reference image."
                ),
                "type": "string",
            },
            "ref_label_aliases": {
                "description": (
                    "Optional aliases for `ref_label`, to improve matching. "
                    "Can be a comma/space separated string (e.g. \"小猴,monkey\") or a JSON array string "
                    "(e.g. [\"小猴\",\"monkey\"])."
                ),
                "type": "string",
            },
            "save_json": {
                "description": (
                    "Whether to persist the prompts JSON to disk. "
                    "Default false to avoid writing local files; set true only if you need a file path for other tools."
                ),
                "type": "boolean",
                "default": False,
            },
        },
        "required": ["shots"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        shots_raw = args.get("shots")
        global_prompt = args.get("global_prompt", "")
        prepend_global_prompt = args.get("prepend_global_prompt", False)
        append_duration = args.get("append_duration", True)
        output_dir = args.get("output_dir", "./storyboard_output")
        save_json = args.get("save_json", False)
        try:
            save_json = bool(save_json)
        except Exception:
            save_json = False
        try:
            prepend_global_prompt = bool(prepend_global_prompt)
        except Exception:
            prepend_global_prompt = False
        try:
            append_duration = bool(append_duration)
        except Exception:
            append_duration = True

        # Parse shots from string or structured object
        shots = []
        if isinstance(shots_raw, str):
            try:
                parsed = json.loads(shots_raw)
            except json.JSONDecodeError:
                parsed = []
        else:
            parsed = shots_raw

        if isinstance(parsed, dict) and "shots" in parsed:
            shots = parsed["shots"]
        else:
            shots = parsed or []

        def _strip_text_overlay_lines(text: str) -> str:
            if not isinstance(text, str) or not text:
                return ""
            lines = [ln.strip() for ln in text.splitlines()]
            drop_keywords = (
                "旁白",
                "配音",
                "解说",
                "口播",
                "台词",
                "对白",
                "字幕",
                "文字贴片",
                "水印",
                "字幕：",
                "旁白：",
                "narration",
                "voiceover",
                "subtitle",
                "subtitles",
                "caption",
                "captions",
                "on-screen text",
                "overlay text",
                "watermark",
                "logo",
                "ui element",
            )
            kept: list[str] = []
            for ln in lines:
                if not ln:
                    continue
                lnl = ln.lower()
                if any(k in ln for k in drop_keywords) or any(k in lnl for k in drop_keywords):
                    continue
                kept.append(ln)
            cleaned = "\n".join(kept).strip()
            return cleaned or text.strip()

        def _strip_audio_lines(text: str) -> str:
            # Keyframe image prompts should not include any music/audio descriptions.
            if not isinstance(text, str) or not text:
                return ""
            lines = [ln.strip() for ln in text.splitlines()]
            drop_keywords = (
                "背景音乐",
                "配乐",
                "音乐",
                "bgm",
                "soundtrack",
                "music",
                "audio",
                "ambient sound",
                "环境音",
                "音轨",
                "配音",
                "旁白",
                "解说",
                "voiceover",
                "narration",
            )
            kept: list[str] = []
            for ln in lines:
                if not ln:
                    continue
                lnl = ln.lower()
                if any(k in ln for k in drop_keywords) or any(k in lnl for k in drop_keywords):
                    continue
                kept.append(ln)
            cleaned = "\n".join(kept).strip()
            return cleaned or text.strip()

        prompts = []
        # 如果调用方提供了 ref_image_dir，说明希望对所有镜头启用参考图（导出时打上 ref 标记）
        ref_image_dir = args.get("ref_image_dir")
        ref_shot_ids = args.get("ref_shot_ids")
        ref_label = args.get("ref_label")
        ref_label_aliases = args.get("ref_label_aliases")
        ref_apply_ids = None  # None means "all"; set({"__FIRST__"}) means "first shot only"
        ref_warnings = []

        def _norm_sid(v):
            try:
                return f"{int(v):02d}"
            except Exception:
                s = str(v).strip()
                return s if len(s) != 1 else ("0" + s)

        def _parse_aliases(raw) -> list[str]:
            if raw is None:
                return []
            if isinstance(raw, list):
                return [str(x).strip() for x in raw if str(x).strip()]
            s = str(raw).strip()
            if not s:
                return []
            # Try JSON array first
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        return [str(x).strip() for x in arr if str(x).strip()]
                except Exception:
                    pass
            # Fallback: split by comma/space
            items = re.split(r"[,\s]+", s)
            return [x.strip() for x in items if x.strip()]

        def _keywords_for_label(label: str, aliases: list[str]) -> list[str]:
            label = (label or "").strip()
            if not label and not aliases:
                return []

            toks = []
            if label:
                toks += [t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", label) if t]
            toks += [t for t in aliases if t]

            syn = []
            l = label.lower() if label else ""
            if ("猴" in label) or ("monkey" in l) or any("monkey" in a.lower() for a in aliases):
                syn += ["猴子", "小猴", "monkey"]
            if ("猫" in label) or ("cat" in l) or ("bobo" in l) or ("toto" in l) or any(
                any(k in a.lower() for k in ("cat", "bobo", "toto")) for a in aliases
            ):
                syn += ["猫", "猫咪", "cat", "cats", "bobo", "toto"]

            # unique while preserving order
            out = []
            for x in [*toks, *syn]:
                xs = str(x).strip()
                if xs and xs not in out:
                    out.append(xs)
            return out

        # Explicit ref shot ids override heuristics
        if ref_shot_ids:
            try:
                items = re.split(r"[,\s]+", str(ref_shot_ids).strip())
                items = [x for x in items if x]
                ref_apply_ids = set(_norm_sid(x) for x in items)
            except Exception:
                ref_apply_ids = None

        # If user specifies which role/entity this reference image belongs to, apply it to all shots mentioning it.
        # This overrides the "single ref -> first shot only" heuristic, but still respects explicit `ref_shot_ids`.
        if (ref_apply_ids is None) and ref_label and ref_image_dir and os.path.exists(ref_image_dir):
            # Only safe when there is exactly one reference image (file, or dir with <=1 image).
            ref_has_single_image = False
            if os.path.isfile(ref_image_dir):
                ref_has_single_image = True
            elif os.path.isdir(ref_image_dir):
                try:
                    files = [
                        f
                        for f in os.listdir(ref_image_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ]
                except Exception:
                    files = []
                ref_has_single_image = len(files) <= 1

            if ref_has_single_image:
                aliases = _parse_aliases(ref_label_aliases)
                keywords = _keywords_for_label(str(ref_label), aliases)
                matched = set()
                if keywords and isinstance(shots, list):
                    for idx, shot in enumerate(shots):
                        if not isinstance(shot, dict):
                            continue
                        sid = shot.get("id") or shot.get("shot_id") or (idx + 1)
                        sid_str = _norm_sid(sid)
                        text = (
                            shot.get("t2i_prompt")
                            or shot.get("midjourney_prompt")
                            or shot.get("description")
                            or shot.get("prompt")
                            or ""
                        )
                        text_l = str(text).lower()
                        if any(str(kw).lower() in text_l for kw in keywords):
                            matched.add(sid_str)
                if matched:
                    ref_apply_ids = matched
                    ref_warnings.append(
                        f"参考图角色匹配：检测到角色“{ref_label}”出现在镜头 {','.join(sorted(matched))}，将对这些镜头启用参考图。"
                    )
                else:
                    ref_warnings.append(
                        f"参考图角色匹配：未在镜头描述中匹配到角色“{ref_label}”，将使用默认策略（仅第 1 镜头或 ref_shot_ids）。"
                    )
            else:
                ref_warnings.append(
                    "参考图角色匹配：检测到 ref_image_dir 目录中包含多张图片，无法自动判断每个角色对应哪张图；"
                    "请改用 `assign_ref_images`（多参考图映射）或显式传入 `ref_shot_ids`。"
                )

        # Heuristic: if only one ref image is provided, do NOT apply it to all shots by default.
        if (ref_apply_ids is None) and ref_image_dir and os.path.exists(ref_image_dir):
            if os.path.isfile(ref_image_dir):
                ref_apply_ids = {"__FIRST__"}
                ref_warnings.append("仅提供了 1 张参考图（文件），默认只用于第 1 个镜头。")
            elif os.path.isdir(ref_image_dir):
                try:
                    files = [
                        f
                        for f in os.listdir(ref_image_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ]
                except Exception:
                    files = []
                if len(files) <= 1:
                    ref_apply_ids = {"__FIRST__"}
                    ref_warnings.append("参考图目录中仅检测到 1 张图片，默认只用于第 1 个镜头。")
        if isinstance(shots, list):
            for idx, shot in enumerate(shots):
                if not isinstance(shot, dict):
                    continue
                sid = shot.get("id") or shot.get("shot_id") or (idx + 1)
                try:
                    sid_str = f"{int(sid):02d}"
                except Exception:
                    sid_str = str(sid)

                # For keyframe image generation we MUST use static keyframe prompt (first frame),
                # not the dynamic shot prompt (video motion description).
                base_desc = (
                    shot.get("keyframe_prompt")
                    or shot.get("keyframe_description")
                    or shot.get("t2i_prompt")
                    or shot.get("midjourney_prompt")
                    or shot.get("prompt")
                    or shot.get("description")
                    or ""
                )
                base_desc = _strip_text_overlay_lines(base_desc)
                base_desc = _strip_audio_lines(base_desc)
                if not base_desc:
                    continue

                duration = shot.get("duration")
                duration_int = None
                try:
                    if duration is not None:
                        di = int(duration)
                        if di > 0:
                            duration_int = di
                except Exception:
                    duration_int = None

                # Build the exact prompt that will be sent to the painter.
                # By default, keep it identical to the user-confirmed per-shot text (no hidden prefix).
                full_prompt = base_desc
                if prepend_global_prompt and global_prompt:
                    gp = _strip_audio_lines(_strip_text_overlay_lines(str(global_prompt)))
                    full_prompt = f"{gp}\n{base_desc}".strip() if gp else base_desc

                # Keyframe prompts describe a single still moment; duration is not relevant and can confuse image generation.
                if (
                    append_duration
                    and duration_int is not None
                    and isinstance(full_prompt, str)
                    and ("时长" not in full_prompt)
                    and ("duration" not in full_prompt.lower())
                ):
                    if ("keyframe_prompt" not in shot) and ("keyframe_description" not in shot):
                        full_prompt = f"{full_prompt}\n时长：{duration_int}秒"

                out_shot = {
                    "shot_id": sid_str,
                    "t2i_prompt": full_prompt,
                }

                # Preserve any upstream per-shot reference bindings when provided explicitly.
                # This enables workflows like:
                # - assigning different keyframe_url values to different characters/shots (via assign_ref_images), then
                # - exporting and painting without losing those per-shot mappings.
                keyframe_url = shot.get("keyframe_url")
                if isinstance(keyframe_url, str) and keyframe_url.strip():
                    out_shot["keyframe_url"] = keyframe_url.strip()
                    # When a keyframe_url is present, the painter will treat it as reference input.
                    out_shot["ref"] = True

                # If upstream already marked this shot as ref, keep it.
                if bool(shot.get("ref", False)):
                    out_shot["ref"] = True

                # IMPORTANT: Never embed base64/data-URI into tool outputs. It will explode LLM context length.
                # If the user provided a local ref image directory, just mark ref=true and pass ref_image_dir
                # to `batch_storyboard_painter`; the painter tool will read/encode the local file internally.
                if (
                    ("ref" not in out_shot)
                    and ref_image_dir
                    and (
                        ref_apply_ids is None
                        or (("__FIRST__" in ref_apply_ids) and idx == 0)
                        or (sid_str in ref_apply_ids)
                    )
                ):
                    out_shot["ref"] = True

                prompts.append(out_shot)

        json_path = None
        if save_json:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(output_dir, f"prompts_{timestamp}.json")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)

        result = {
            "status": "success",
            "json_path": os.path.abspath(json_path) if json_path else None,
            "shot_count": len(prompts),
            "json_content": prompts if not save_json else None,
            "warnings": ref_warnings,
        }
        return json.dumps(result, ensure_ascii=False)


@register_tool("assign_ref_images")
class AssignRefImages(BaseTool):
    """Assign reference images to shots based on user-provided labels.

    This tool is designed to prevent the common failure mode where a single ref image
    is accidentally applied to all shots. It annotates each shot with:
    - ref=true
    - keyframe_url=<file://... or http(s)://...>
    for only the matched shots.
    """

    description = (
        "Assign uploaded reference images to specific shots automatically based on labels "
        "(e.g., '猴子参考图', '猫参考图') and shot descriptions."
    )

    parameters = {
        "type": "object",
        "properties": {
            "shots": {
                "description": "Shots list (JSON string or list/dict). Each shot should contain id/shot_id and description/t2i_prompt.",
                "type": "string",
            },
            "ref_images": {
                "description": (
                    "Reference images with labels. JSON string of either: "
                    "1) list of {label, path} items, or 2) dict {label: path}. "
                    "path can be http(s) URL, file:// URL, or local Windows path."
                ),
                "type": "string",
            },
        },
        "required": ["shots", "ref_images"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        shots_raw = args.get("shots")
        ref_raw = args.get("ref_images")

        # Parse shots
        if isinstance(shots_raw, str):
            try:
                parsed_shots = json.loads(shots_raw)
            except Exception:
                parsed_shots = []
        else:
            parsed_shots = shots_raw

        if isinstance(parsed_shots, dict) and "shots" in parsed_shots:
            shots = parsed_shots["shots"]
        else:
            shots = parsed_shots or []

        # Parse refs
        if isinstance(ref_raw, str):
            try:
                ref_parsed = json.loads(ref_raw)
            except Exception:
                ref_parsed = {}
        else:
            ref_parsed = ref_raw

        refs = []
        if isinstance(ref_parsed, dict):
            for k, v in ref_parsed.items():
                refs.append({"label": str(k), "path": v})
        elif isinstance(ref_parsed, list):
            for item in ref_parsed:
                if isinstance(item, dict) and item.get("label") and item.get("path"):
                    refs.append({"label": str(item["label"]), "path": item["path"]})

        def _to_file_url(p: str) -> str:
            if not isinstance(p, str):
                return ""
            p = p.strip()
            if not p:
                return ""
            if p.startswith(("http://", "https://", "file://", "data:")):
                return p
            # Windows path
            if re.match(r"^[A-Za-z]:[\\\\/]", p):
                return "file://" + p.replace("\\", "/")
            return p

        def _keywords_for_label(label: str):
            label = (label or "").strip()
            if not label:
                return []
            # split into tokens; keep Chinese chunks as tokens too
            toks = [t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", label) if t]

            # add lightweight built-in synonyms
            syn = []
            l = label.lower()
            if ("猴" in label) or ("monkey" in l):
                syn += ["猴子", "小猴", "monkey"]
            if ("猫" in label) or ("cat" in l) or ("bobo" in l) or ("toto" in l):
                syn += ["猫", "cat", "cats", "bobo", "toto"]
            return list(dict.fromkeys([*toks, *syn]))

        ref_entries = []
        for r in refs:
            label = r.get("label", "")
            path = _to_file_url(str(r.get("path", "")).strip())
            if not label or not path:
                continue
            ref_entries.append(
                {
                    "label": label,
                    "path": path,
                    "keywords": _keywords_for_label(label),
                }
            )

        assignments = {}
        unmatched_shots = []

        for idx, shot in enumerate(shots):
            if not isinstance(shot, dict):
                continue

            sid = shot.get("shot_id") or shot.get("id") or (idx + 1)
            try:
                sid_str = f"{int(sid):02d}"
            except Exception:
                sid_str = str(sid)

            text = (
                shot.get("t2i_prompt")
                or shot.get("description")
                or shot.get("prompt")
                or ""
            )
            text_l = str(text).lower()

            best = None
            best_score = 0
            for r in ref_entries:
                score = 0
                for kw in r["keywords"]:
                    kw_l = str(kw).lower()
                    if not kw_l:
                        continue
                    if kw_l in text_l:
                        score += 1
                if score > best_score:
                    best_score = score
                    best = r

            if best and best_score > 0:
                shot["ref"] = True
                shot["keyframe_url"] = best["path"]
                assignments[sid_str] = {"label": best["label"], "path": best["path"], "score": best_score}
            else:
                unmatched_shots.append(sid_str)

        return json.dumps(
            {
                "status": "success",
                "shots": shots,
                "assignments": assignments,
                "unmatched_shots": unmatched_shots,
                "ref_count": len(ref_entries),
            },
            ensure_ascii=False,
        )


@register_tool("my_video_gen")
class MyVideoGen(BaseTool):
    """A custom tool for final video generation.

    This is a minimal example. It assumes there exists some video
    generation service that accepts the refined text prompt plus
    keyframe plan and returns a video URL.
    Replace the implementation of ``call`` with your real backend logic.
    """

    description = (
        "Video generation service for shot-based workflow. "
        "It takes a detailed video prompt (CN/EN) and a list of shots, "
        "then generates a short video clip for each shot."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "description": (
                    "The final refined description of the video to generate (CN/EN)."
                ),
                "type": "string",
            },
            "shots": {
                "description": (
                    "A JSON-serializable list or string that describes the planned shots / "
                    "storyboard for the video. Prefer fields: id, duration, shot_prompt (dynamic), keyframe_prompt (static), keyframe_url."
                ),
                "type": "string",
            },
            "auto_download": {
                "description": (
                    "Whether to download generated clips to local files automatically. "
                    "Default false: users can manually download via the returned video_url in WebUI/browser."
                ),
                "type": "boolean",
                "default": False,
            },
            "audio_url": {
                "description": (
                    "Optional background audio track URL/path/file:// to condition the video generation. "
                    "If provided, it will be passed to DashScope VideoSynthesis as audio_url. "
                    "Per-shot audio_url can also be provided inside each shot object to override this."
                ),
                "type": "string",
            },
            "max_workers": {
                "description": (
                    "Optional: maximum parallel workers for per-shot video generation inside the tool. "
                    "Default 3. Higher values may hit provider rate limits/timeouts."
                ),
                "type": "integer",
                "default": 3,
            },
        },
        "required": ["prompt", "shots"],
    }

    def call(self, params: str, **kwargs) -> str:
        def _parse_tool_args(raw):
            # qwen-agent normally passes a JSON string, but sometimes it can contain
            # extra trailing content (leading to JSONDecodeError: Extra data).
            if isinstance(raw, dict):
                return raw
            if raw is None:
                return {}
            if not isinstance(raw, str):
                raw = str(raw)

            s = raw.strip()
            if not s:
                return {}

            # 1) Strict JSON
            try:
                return json.loads(s)
            except Exception:
                pass

            # 2) Parse the first JSON value and ignore trailing garbage.
            try:
                obj, _end = json.JSONDecoder().raw_decode(s)
                return obj if isinstance(obj, dict) else {"_": obj}
            except Exception:
                pass

            # 3) JSON5 tolerant parse (optional dependency).
            try:
                import json5  # type: ignore

                obj = json5.loads(s)
                return obj if isinstance(obj, dict) else {"_": obj}
            except Exception:
                pass

            # 4) Heuristic: find first '{' and parse from there.
            try:
                start = s.find("{")
                if start != -1:
                    obj, _end = json.JSONDecoder().raw_decode(s[start:])
                    return obj if isinstance(obj, dict) else {"_": obj}
            except Exception:
                pass

            raise ValueError("Invalid tool args: not valid JSON object")

        # Parse JSON arguments (tolerant)
        try:
            args = _parse_tool_args(params)
        except Exception as e:
            return json.dumps(
                {
                    "clips": [],
                    "failed_clips": [
                        {
                            "id": None,
                            "attempt": None,
                            "status": "error",
                            "message": f"Invalid my_video_gen params: {e}",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        prompt = args["prompt"]
        shots_raw = args["shots"]
        audio_url_global = args.get("audio_url")
        max_workers_arg = args.get("max_workers", None)
        auto_download_arg = args.get("auto_download", None)

        # Parse shots (may be a JSON string or a list)
        if isinstance(shots_raw, str):
            try:
                shots = json.loads(shots_raw)
            except json.JSONDecodeError:
                # Some upstream callers may embed literal newlines inside the JSON string;
                # try a tolerant pass that escapes newlines.
                try:
                    fixed = (
                        shots_raw.replace("\r\n", "\\n")
                        .replace("\n", "\\n")
                        .replace("\r", "\\n")
                    )
                    shots = json.loads(fixed)
                except Exception:
                    shots = []
        else:
            shots = shots_raw

        # Normalize: allow both {"shots":[...]} and plain list.
        if isinstance(shots, dict) and "shots" in shots:
            shots = shots.get("shots") or []

        try:
            print(f"[MyVideoGen] Received shots={len(shots) if isinstance(shots, list) else type(shots)} max_workers_arg={max_workers_arg!r}")
        except Exception:
            pass

        # Parallelism: param > env > default(1)
        try:
            if max_workers_arg is None:
                max_workers = int(os.getenv("VIDEO_MAX_WORKERS", "3") or 3)
            else:
                max_workers = int(max_workers_arg)
        except Exception:
            max_workers = 3
        if max_workers < 1:
            max_workers = 1
        max_workers = min(max_workers, 6)

        clips = []
        # If enabled, download each generated clip to a local file so the user can click a stable `/file=...` link in WebUI.
        # Default OFF: users can manually download the remote URL in the browser.
        env_auto = os.getenv("VIDEO_AUTO_DOWNLOAD", "0").strip().lower() not in ("0", "false", "no", "off")
        if auto_download_arg is None:
            auto_download = env_auto
        else:
            try:
                auto_download = bool(auto_download_arg)
            except Exception:
                auto_download = env_auto
        download_dir = os.getenv("VIDEO_DOWNLOAD_DIR", os.path.join("workspace", "downloads")).strip() or os.path.join(
            "workspace", "downloads"
        )
        try:
            max_mb = int(os.getenv("VIDEO_MAX_DOWNLOAD_MB", "200") or 200)
        except Exception:
            max_mb = 200
        max_bytes = max(1, max_mb) * 1024 * 1024

        # 评分阈值与最大重试次数：若评分低于该阈值，则自动尝试重新生成一次
        QUALITY_THRESHOLD = 60.0
        MAX_ATTEMPTS = 2

        # 对每个分镜单独调用 sample_async_call：
        # - 如果有 keyframe_url，则作为图生视频参考图
        # - 否则走文生视频
        def _looks_like_chinese(s: str) -> bool:
            if not isinstance(s, str) or not s:
                return False
            cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
            return cjk >= 4

        def _no_voice_constraints() -> str:
            # User requested: disable voiceover feature entirely.
            return (
                "\n\nAudio constraints (NO narration):"
                "\n- 全程不要配音/旁白/解说/对白（无口播/人声）。"
                "\n- If audio is supported, use ONLY instrumental background music / ambient sound (NO speech, NO lyrics)."
                "\n- 画面中不要出现任何字幕/台词文字/文字贴片/水印/Logo/UI界面元素。"
                "\n- Do NOT add any subtitles/captions/on-screen text/overlays/watermarks/logos/UI elements."
            )

        def _sanitize_shot_desc(desc_text: str) -> str:
            # Remove any narration/subtitle/dialogue lines so the model won't burn text into frames.
            if not isinstance(desc_text, str) or not desc_text:
                return ""
            lines = [ln.strip() for ln in desc_text.splitlines()]
            drop_keywords = (
                "旁白",
                "配音",
                "解说",
                "口播",
                "台词",
                "对白",
                "字幕",
                "文字贴片",
                "水印",
                "字幕：",
                "旁白：",
                "narration",
                "voiceover",
                "subtitle",
                "subtitles",
                "caption",
                "captions",
                "on-screen text",
                "overlay text",
                "watermark",
                "logo",
            )
            kept: list[str] = []
            for ln in lines:
                if not ln:
                    continue
                lnl = ln.lower()
                if any(k in ln for k in drop_keywords) or any(k in lnl for k in drop_keywords):
                    continue
                kept.append(ln)
            cleaned = "\n".join(kept).strip()
            return cleaned or desc_text.strip()

        def _wants_background_music(text: str) -> bool:
            if not isinstance(text, str) or not text:
                return False
            t = text.lower()
            if any(k in text for k in ("背景音乐", "bgm", "配乐", "音乐", "氛围音乐", "环境音")):
                return True
            if any(k in t for k in ("background music", "bgm", "soundtrack", "ambient sound")):
                return True
            return False

        def _bgm_hint_constraints(global_prompt: str) -> str:
            # If user wants BGM but we don't have an explicit audio_url, at least nudge the model.
            # Note: actual audio generation capability depends on the backend model.
            if not _wants_background_music(global_prompt):
                return ""
            return (
                "\n\nAudio/Soundtrack preference:" \
                "\n- 如果支持音频轨：请加入与氛围匹配的背景音乐/环境音（纯音乐，无歌词，无人声）。" \
                "\n- If an audio track is supported: add suitable instrumental background music / ambient sound (NO lyrics, NO speech)." \
                "\n- Do NOT add any subtitles/captions/on-screen text."
            )

        def _run_one_shot(shot: dict, idx: int):
            if not isinstance(shot, dict):
                return idx, None, None

            # Compatibility: accept different upstream field names.
            sid = (
                shot.get("id")
                or shot.get("shot_id")
                or shot.get("shotId")
                or shot.get("shotID")
                # seen in some upstream/LLM outputs (typo)
                or shot.get("shot_iid")
            )
            desc = (
                shot.get("shot_prompt")
                or shot.get("video_prompt")
                or shot.get("description")
                or shot.get("prompt")
                # Fallbacks: if caller accidentally passes keyframe/t2i prompts to video tool.
                or shot.get("t2i_prompt")
                or shot.get("keyframe_prompt")
                or ""
            )
            desc = _sanitize_shot_desc(desc)

            # 1) 优先使用结构化字段 duration
            duration_val = shot.get("duration")
            duration = None
            if duration_val is not None:
                try:
                    duration = int(duration_val)
                except Exception:
                    duration = None

            # 2) 若缺少 duration 字段，则尝试从中文描述中解析“时长：X秒”
            if duration is None and isinstance(desc, str):
                m = re.search(r"时长[:：]?\s*(\d+)\s*秒", desc)
                if m:
                    try:
                        duration = int(m.group(1))
                    except Exception:
                        duration = None

            # 3) 仍未获得合法时长时，安全回退到默认 5 秒
            if duration is None:
                duration = 5
            if duration <= 0:
                duration = 5

            # Reference image / keyframe for I2V (optional). Accept a few common aliases.
            keyframe_url = (
                shot.get("keyframe_url")
                or shot.get("image_url")
                or shot.get("keyframe")
                or ""
            )

            # Optional per-shot audio override
            audio_url = (
                shot.get("audio_url")
                or shot.get("bgm_url")
                or audio_url_global
                or os.getenv("VIDEO_AUDIO_URL", "").strip()
                or None
            )

            if not desc:
                try:
                    keys = ",".join(sorted(list(shot.keys()))[:40])
                    print(f"[MyVideoGen] ⚠️ Skip shot (missing description). sid={sid!r}, keys={keys}")
                except Exception:
                    pass
                return idx, None
            if not sid:
                sid = shot.get("shot_id") or shot.get("id") or str(idx + 1)

            # Keep the wrapper language aligned with the user's prompt language.
            if _looks_like_chinese(prompt) or _looks_like_chinese(desc):
                shot_intro = f"\n\n这是第 {sid} 镜头："
                dur_line = f" 本镜头时长约 {duration} 秒。"
                hard = (
                    "\n\n硬性约束："
                    "\n- 严格遵守上文的角色数量约束；不要新增人物/动物/重复角色。"
                    "\n- 环境/地点/光照需与该镜头描述及（如提供）关键帧参考图保持一致。"
                    "\n- 画面中绝对不要出现任何字幕/文字/贴片/水印/Logo/UI界面元素。"
                )
            else:
                shot_intro = f"\n\nThis is shot {sid}: "
                dur_line = f" The clip duration should be about {duration} seconds."
                hard = (
                    "\n\nHard constraints:"
                    "\n- Respect all character-count constraints stated above; do NOT add extra people/animals/duplicate characters."
                    "\n- Keep the environment/location/lighting consistent with this shot description and (if provided) the keyframe image."
                    "\n- Do NOT add ANY on-screen text/overlays/subtitles/captions/watermarks/logos/UI elements."
                )

            shot_prompt = (
                prompt
                + shot_intro
                + desc
                + dur_line
                + hard
                + _bgm_hint_constraints(prompt)
                + _no_voice_constraints()
            )
            input_img_url = keyframe_url if keyframe_url else None

            video_url = None
            quality_score = None
            last_fail = None

            for attempt in range(1, MAX_ATTEMPTS + 1):
                result = sample_async_call(
                    shot_prompt,
                    input_img_url=input_img_url,
                    duration=duration,
                    audio_url=audio_url,
                )

                if not isinstance(result, dict):
                    try:
                        print(f"[MyVideoGen] Shot {sid} attempt {attempt} returned non-dict result: {result}")
                    except Exception:
                        pass
                    continue
                if result.get("status") != "success" or not result.get("video_url"):
                    try:
                        print(
                            f"[MyVideoGen] Shot {sid} attempt {attempt} failed: "
                            f"status={result.get('status')}, message={result.get('message')}"
                        )
                    except Exception:
                        pass
                    last_fail = {
                        "id": sid,
                        "attempt": attempt,
                        "status": result.get("status"),
                        "message": result.get("message"),
                    }
                    continue

                video_url = result["video_url"]

                quality_score = None
                try:
                    score_result = score_video(video_url)
                    if isinstance(score_result, dict):
                        quality_score = score_result.get("overall_score")
                except Exception:
                    quality_score = None

                try:
                    print(
                        f"[MyVideoGen] Shot {sid} attempt {attempt} quality_score = "
                        f"{quality_score if quality_score is not None else 'N/A'}"
                    )
                except Exception:
                    pass

                if quality_score is None or quality_score >= QUALITY_THRESHOLD:
                    break

            if not video_url:
                if last_fail is None:
                    last_fail = {
                        "id": sid,
                        "attempt": None,
                        "status": "error",
                        "message": "No video_url returned (unknown reason).",
                    }
                return idx, None, last_fail

            clip = {
                "id": sid,
                # Keep tool outputs small to avoid LLM input-length limits.
                # Full prompts/descriptions can be extremely long and are not needed for downstream tools.
                "duration": duration,
                "video_url": video_url,
                "quality_score": quality_score,
            }

            try:
                include_desc = os.getenv("VIDEO_INCLUDE_DESC", "0").strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                include_desc = False
            if include_desc:
                clip["description"] = desc
            else:
                if isinstance(desc, str) and desc:
                    clip["description_preview"] = (desc[:400] + "…") if len(desc) > 400 else desc

            if auto_download and isinstance(video_url, str) and video_url.startswith(("http://", "https://")):
                try:
                    os.makedirs(download_dir, exist_ok=True)
                    safe_id = re.sub(r"[^0-9A-Za-z._-]+", "_", str(sid))
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{safe_id}_{ts}.mp4"
                    local_rel = os.path.join(download_dir, filename)
                    local_abs = os.path.abspath(local_rel)
                    size = _download_http_file(video_url, local_abs, max_bytes=max_bytes)

                    local_url = "/file=" + local_rel.replace("\\", "/")
                    clip["download_path"] = local_abs
                    clip["download_url"] = local_url
                    clip["download_bytes"] = size
                    try:
                        print(f"[MyVideoGen] Download ready: {local_abs} ({size} bytes)")
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        print(f"[MyVideoGen] Download failed for {sid}: {e}")
                    except Exception:
                        pass

            # WebUI-friendly preview helpers
            try:
                preview_url = clip.get("download_url") or clip.get("video_url")
                if isinstance(preview_url, str) and preview_url:
                    clip["preview_url"] = preview_url
            except Exception:
                pass

            return idx, clip, None

        if isinstance(shots, list):
            failed_clips = []
            if max_workers <= 1 or len(shots) <= 1:
                for idx, shot in enumerate(shots):
                    _, clip, fail = _run_one_shot(shot, idx)
                    if clip:
                        clips.append(clip)
                    elif fail:
                        failed_clips.append(fail)
            else:
                try:
                    print(f"[MyVideoGen] Parallel mode: max_workers={max_workers}, shots={len(shots)}")
                except Exception:
                    pass
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [ex.submit(_run_one_shot, shot, idx) for idx, shot in enumerate(shots)]
                    results = []
                    for fut in as_completed(futs):
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            try:
                                print(f"[MyVideoGen] worker exception: {e}")
                            except Exception:
                                pass
                    for _, clip, fail in sorted(results, key=lambda x: x[0]):
                        if clip:
                            clips.append(clip)
                        elif fail:
                            failed_clips.append(fail)

        # 保持原有工具返回格式不变：始终返回 {"clips": [...]} 结构
        out = {"clips": clips, "failed_clips": []}
        try:
            if "failed_clips" in locals() and failed_clips:
                out["failed_clips"] = failed_clips
        except Exception:
            pass
        return json.dumps(out, ensure_ascii=False)


@register_tool("video_concatenate")
class VideoConcatenate(BaseTool):
    """A simple tool to concatenate multiple video clips into one file.

    It downloads each clip from the given URLs, uses ffmpeg to
    concatenate them in order, and (optionally) mixes in a background
    music track (first N seconds). Returns a local path and a WebUI
    download link (/file=...) to the merged video.
    """

    description = (
        "Concatenate multiple video clips in order into a single video file "
        "using ffmpeg. Optionally add background music (first N seconds). "
        "Input is a list of clip URLs or objects with video_url."
    )
    parameters = {
        "type": "object",
        "properties": {
            "clips": {
                "description": (
                    "A JSON-serializable list of clips. Each clip can be either "
                    "a string URL or an object with a 'video_url' field. The "
                    "order in the list is the concatenation order."
                ),
                "type": "string",
            },
            "music_url": {
                "description": (
                    "Optional background music URL/path/file://. "
                    "If provided, the merged output will include this audio (trimmed to music_seconds)."
                ),
                "type": "string",
            },
            "music_seconds": {
                "description": "Optional: use only the first N seconds of the music (default 10).",
                "type": "number",
                "default": 10,
            },
            "output_path": {
                "description": (
                    "Optional output mp4 path. Defaults to workspace/edits/concat_<timestamp>.mp4 "
                    "so it can be downloaded in WebUI."
                ),
                "type": "string",
            },
        },
        "required": ["clips"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        clips_raw = args.get("clips", [])
        music_url = (args.get("music_url") or "").strip()
        music_seconds = args.get("music_seconds", 10)
        out_path_arg = (args.get("output_path") or "").strip()

        try:
            max_mb = int(os.getenv("CONCAT_MAX_DOWNLOAD_MB", os.getenv("VIDEO_MAX_DOWNLOAD_MB", "500")) or 500)
        except Exception:
            max_mb = 500
        max_bytes = max(1, max_mb) * 1024 * 1024
        try:
            ffmpeg_timeout = int(os.getenv("FFMPEG_CONCAT_TIMEOUT_SECONDS", "900") or 900)
        except Exception:
            ffmpeg_timeout = 900

        # Parse clips list (stringified JSON or list)
        if isinstance(clips_raw, str):
            try:
                clips = json.loads(clips_raw)
            except json.JSONDecodeError:
                clips = []
        else:
            clips = clips_raw

        normalized = []
        if isinstance(clips, list):
            for item in clips:
                if isinstance(item, str):
                    url = item
                elif isinstance(item, dict):
                    url = item.get("video_url") or item.get("url") or ""
                else:
                    url = ""
                url = (url or "").strip()
                if url:
                    normalized.append(url)

        if not normalized:
            raise RuntimeError("No valid clip URLs provided to video_concatenate.")

        def _file_url_to_local(u: str) -> str:
            local_path = urllib.parse.unquote(urllib.parse.urlparse(u).path)
            if re.match(r"^/[A-Za-z]:/", local_path):
                local_path = local_path[1:]
            return local_path

        def _webui_file_to_local(u: str) -> str | None:
            # WebUI serves local files under allowed_paths as /file=<relpath>.
            if not isinstance(u, str):
                return None
            if u.startswith("/file="):
                rel = u[len("/file=") :].lstrip("/")
                return os.path.abspath(rel.replace("/", os.sep))
            if u.startswith("file="):
                rel = u[len("file=") :].lstrip("/")
                return os.path.abspath(rel.replace("/", os.sep))
            return None

        tmp_dir = tempfile.mkdtemp(prefix="video_concat_")
        part_files = []

        # Download each clip to a local temporary file
        for idx, url in enumerate(normalized):
            part_path = os.path.join(tmp_dir, f"part_{idx}.mp4")
            try:
                print(f"[video_concatenate] Fetch clip {idx + 1}/{len(normalized)}: {url[:120]}")
                local_path = None
                if url.startswith("file://"):
                    local_path = _file_url_to_local(url)
                else:
                    local_path = _webui_file_to_local(url)
                    if local_path is None and re.match(r"^[A-Za-z]:[\\\\/]", url):
                        local_path = url

                if local_path is not None:
                    if not os.path.exists(local_path):
                        raise RuntimeError(f"Local clip not found: {local_path}")
                    # Copy local file into temp dir for concat stability
                    with open(local_path, "rb") as src, open(part_path, "wb") as dst:
                        dst.write(src.read())
                else:
                    # http(s) download with timeout/size guard
                    if not url.startswith(("http://", "https://")):
                        raise RuntimeError(f"Unsupported clip URL (expected http(s), file://, /file=, or local path): {url}")
                    _download_http_file(url, part_path, max_bytes=max_bytes)
            except Exception as e:
                raise RuntimeError(f"Failed to download clip {idx + 1}: {e}")
            part_files.append(part_path)

        # Prepare ffmpeg concat input file
        list_file = os.path.join(tmp_dir, "inputs.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in part_files:
                safe_path = p.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        # Default output inside workspace/ so WebUI can serve it via /file=...
        if out_path_arg:
            output_path = os.path.abspath(out_path_arg)
        else:
            os.makedirs(os.path.join("workspace", "edits"), exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.abspath(os.path.join("workspace", "edits", f"concat_{ts}.mp4"))
        concat_path = os.path.join(tmp_dir, "final_concat.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            concat_path,
        ]

        try:
            print(f"[video_concatenate] ffmpeg concat (stream copy) ...")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=ffmpeg_timeout)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH. Please install ffmpeg to use video_concatenate."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg concat timed out after {ffmpeg_timeout}s")

        if proc.returncode != 0:
            # Fallback: re-encode when stream-copy concat fails
            cmd2 = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                concat_path,
            ]
            print(f"[video_concatenate] ffmpeg concat fallback (re-encode) ...")
            proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=ffmpeg_timeout)
            if proc2.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {proc.stderr or proc2.stderr}")

        # Optional: add background music (first N seconds)
        if music_url:
            ms = 10.0
            try:
                ms = float(music_seconds)
            except Exception:
                ms = 10.0
            if ms <= 0:
                ms = 10.0

            music_local = music_url
            if music_local.startswith("file://"):
                music_local = urllib.parse.unquote(urllib.parse.urlparse(music_local).path)
                if re.match(r"^/[A-Za-z]:/", music_local):
                    music_local = music_local[1:]
            elif music_local.startswith(("http://", "https://")):
                music_dst = os.path.join(tmp_dir, "bgm.mp3")
                try:
                    _download_http_file(music_local, music_dst, max_bytes=max_bytes)
                except Exception as e:
                    raise RuntimeError(f"Failed to download music: {e}")
                music_local = music_dst

            if not os.path.exists(music_local):
                raise RuntimeError(f"Music file not found: {music_local}")

            cmd3 = [
                "ffmpeg",
                "-y",
                "-i",
                concat_path,
                "-i",
                music_local,
                "-filter_complex",
                f"[1:a]atrim=0:{ms},asetpts=N/SR/TB[a]",
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                output_path,
            ]
            print(f"[video_concatenate] ffmpeg mux music ...")
            proc3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=ffmpeg_timeout)
            if proc3.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed: {proc3.stderr}")
        else:
            # No music: move concat result to final output path
            try:
                with open(concat_path, "rb") as src, open(output_path, "wb") as dst:
                    dst.write(src.read())
            except Exception as e:
                raise RuntimeError(f"Failed to write output: {e}")

        output_url = "/file=" + os.path.relpath(output_path, os.getcwd()).replace("\\", "/")
        return json.dumps(
            {"output_path": output_path, "output_url": output_url, "clip_count": len(part_files)},
            ensure_ascii=False,
        )


def init_agent_service():
    """Initialize the video generation assistant.

     The agent's workflow (described in system_message):
     1) Understand the user's idea and refine it into a detailed English video prompt.
     2) Clearly show the refined prompt to the user.
     3) Call ``my_keyframe_planner`` with that refined prompt to generate
         a list of shots (storyboard segments), each with description and
         duration. Image generation for these shots is handled by separate
         tools.
     4) Optionally discuss / adjust the shots and (later) keyframes with the user.
     5) Call ``my_video_gen`` with the refined prompt and shots to generate
         a short video clip for each shot.
     6) Return the list of generated clip URLs and a short explanation
         in the final reply.
    """

    llm_cfg = {"model": "qwen-max"}

    system = (
        "You are a helpful video generation assistant. "
        "IMPORTANT OVERRIDE POLICY: Always follow the user's explicit intent about which stage to run. "
        "If the user clearly asks for only one stage (e.g., only analyze a reference video, only plan shots, only "
        "generate storyboard/keyframe images, or only generate video clips), you MUST do only that stage and stop. "
        "Do NOT force the full end-to-end pipeline. Only run the full workflow when the user has not expressed a clear "
        "preference or explicitly asks for an end-to-end result. When the request is ambiguous, ask one short "
        "clarifying question before proceeding. "
        "KEY RULE FOR KEYFRAMES: If the user asks to directly generate storyboard/keyframe images, you SHOULD NOT call "
        "`my_PromptEnhancer` unless the user explicitly asks you to enhance/expand the prompt. Use the user's prompt "
        "as-is to plan shots and generate keyframe images. When calling `batch_storyboard_painter`, you MUST pass "
        "`json_content` as a JSON list of objects with at least `shot_id` and `t2i_prompt` fields (do not use only "
        "`id`/`description`), and if the user uploaded a local reference image, you MUST pass that file path via the "
        "`ref_image_dir` argument (it may be a single image file path). Alternatively, if you have already called "
        "`assign_ref_images` and embedded per-shot `keyframe_url` values (file://.../http(s)://...) together with "
        "`ref=true` for only the intended shots, you MAY omit `ref_image_dir`. "
        "IMPORTANT FOR IMAGE BACKENDS: The per-shot `t2i_prompt` should contain an explicit image-generation instruction "
        "(e.g., '请生成一幅...的图像' / 'Generate a high-quality image of ...') to reduce 'no image returned' failures."
        "MUSIC (OPTIONAL, REQUIRES EXPLICIT USER APPROVAL): If the user wants background music, you MUST ask for explicit "
        "permission before calling any music generation tools, because it may incur external service costs. If the user "
        "approves, call `music_prompt_refiner` with a short description of the video (or the refined prompt) to get a "
        "JSON with `title` and `style`, show the planned style to the user, then (only after the user confirms) call "
        "`my_music_gen` to generate instrumental BGM and obtain an `audio_url`. If the user does not approve, skip music. "
        "If an `audio_url` is available and the user wants the final edited video with music, use `video_editor` and pass "
        "the clip URLs as `video_urls` and the `audio_url` as `music_url` ONLY when all assets are publicly accessible "
        "http(s) URLs (Shotstack cannot read file:// local paths). If the user asks for simple concatenation with music "
        "and any input is a local path/file://, you MUST use `video_concatenate` with `music_url` and `music_seconds` to "
        "do local ffmpeg concatenation+mixing instead of `video_editor`. "
        "Your workflow has six main stages: "
        "(1) First, ask the user for their video idea and, if possible, a reference short video URL or reference image "
        "from the internet. When a valid video URL is provided, call the `video_script_analyzer` tool to download and "
        "analyze that reference video, obtaining a structured script and adaptation suggestions. In particular, when "
        "the user provides both a textual description and a reference video (either as a URL in the message text, or "
        "as a local file attachment whose path ends with common video extensions such as .mp4, .mov, .avi, or .mkv), "
        "you MUST treat that file as a reference video and use `video_script_analyzer` as the primary analysis path, "
        "rather than sending that file to `my_PromptEnhancer`. When no reference video is available (the user only "
        "provides a textual idea and optional reference image or uploaded **image**), you should normally call the "
        "`my_PromptEnhancer` tool to enrich and expand the user's textual description into a stronger prompt. However, "
        "if the user explicitly states that they already have a satisfactory prompt and wants to directly proceed to "
        "shot planning or video generation (for example, they say things like 'do not enhance the prompt', 'just use my "
        "prompt as-is', or 'directly generate the video with this prompt'), you MUST respect the user's intention and "
        "skip calling `my_PromptEnhancer`, using the user's provided prompt directly for the next steps. "
        "If the user has uploaded a local **image** via the Web UI (so that the latest user message contains an image "
        "attachment with a file:// path and the file name ends with a typical image extension like .jpg, .jpeg, .png, "
        ".webp, or .gif), you should directly pass that file path as the `referedImage` argument to `my_PromptEnhancer` "
        "instead of asking the user to upload the image to an external image hosting service or provide a public URL. "
        "If the uploaded local file is a video (for example, the path ends with .mp4, .mov, .avi, or .mkv), you should "
        "NOT send it to `my_PromptEnhancer`. Instead, you should normally call the `local_video_script_analyzer` tool "
        "with the file path provided by the Web UI so that the model can analyze the local file directly without "
        "requiring a public URL. Only if this local analysis fails or the user explicitly prefers to use an online "
        "reference (for example, they say they will upload the video somewhere and give you a link), you may ask the "
        "user to provide a publicly accessible video URL and then call the `video_script_analyzer` tool with that URL. "
        "When "
        "`my_PromptEnhancer` returns a JSON object, it will always contain at least an `enhanced_prompt` field and may "
        "also contain `positive_prompt` and `negative_prompt` fields. You MUST treat `enhanced_prompt` as the refined "
        "text prompt (even if it looks similar to the original text) instead of claiming that the tool failed. If "
        "`positive_prompt` and `negative_prompt` are present, you MUST clearly present BOTH parts to the user: the "
        "positive_prompt as what should appear in the video and the negative_prompt as everything that should be avoided "
        "(such as low resolution, overexposure, anatomical errors, ugly or distorted content). When you later call the "
        "`my_video_gen` tool, if a separate negative_prompt is available, you SHOULD incorporate it into the `prompt` "
        "argument (for example by appending a segment like 'Negative prompt: ...') so that the downstream generation "
        "model can respect these constraints. Only describe the tool as failed if you actually receive an explicit error "
        "message. In both cases, use the analysis or enhanced prompt together with the user's requirements to refine "
        "their idea into a detailed video prompt (describe scenes, characters, actions, camera motion, style, duration, "
        "etc.). When you PRESENT the refined or enhanced prompt to the user, you MUST do so in the same language that "
        "the user is primarily using in the conversation (for example, if the user writes in Chinese, show the prompt in "
        "Chinese). Internally, when calling tools or backend generation models that work better with English prompts, you "
        "MAY translate the refined prompt to English for those calls, but the user-facing explanation and displayed "
        "prompt must stay in the user's language. After obtaining the enhanced or refined prompt, explicitly show this "
        "user-language prompt (and, if available, its positive and negative parts) to the user and ASK the user to "
        "confirm whether they are satisfied or want any changes before proceeding to shot planning. "
        "(2) Second, you should normally call the `my_keyframe_planner` tool using that refined prompt to generate a "
        "list of shots / storyboard segments (each with description and duration). The number "
        "of shots can be decided based on the conversation (for example, matching the number of shots the user or "
        "analysis has already provided). However, if the user explicitly states that they do NOT want you to do shot "
        "planning or keyframe generation (for example, they say things like 'skip the shot planning step', 'do not "
        "generate keyframes', or 'directly generate the video with this prompt or with my own shot list'), you MUST "
        "respect the user's intention and skip calling `my_keyframe_planner`. In that case, you should either use the "
        "shot list provided by the user directly or, if no explicit shot list is provided, construct a simple shot list "
        "yourself (for example, a single shot that covers the whole prompt) and then call `my_video_gen` directly. "
        "Present any shots you generate clearly to the user and adjust if needed. "
        "IMPORTANT CHARACTER COUNT RULE: Whenever YOU generate or refine prompts (global prompt, per-shot descriptions, "
        "or keyframe prompts), you MUST include explicit character count constraints to avoid unintended duplicates. "
        "For example, say 'ONLY ONE monkey (the same single monkey across all shots)' and 'ONLY TWO cats: Bobo and Toto', "
        "and explicitly state 'do NOT add extra monkeys/cats/people' when relevant. If a character should persist across "
        "shots, explicitly say it is the SAME character instance (same appearance/identity) rather than a new one. "
        "IMPORTANT ENVIRONMENT RULE: For EVERY shot description / keyframe prompt you produce, you MUST include enough "
        "environment context to avoid ambiguity (location, background, time-of-day/lighting, key props, and scene mood). "
        "Keep the environment consistent across shots unless the user explicitly requests a location change. If you "
        "introduce a setting in early shots (e.g., 'apartment balcony overlooking a street'), you MUST keep subsequent "
        "shots aligned with it (same balcony, same street layout) and avoid contradictions (e.g., suddenly indoors) "
        "unless explicitly directed. "
        "IMPORTANT NO-TEMPLATE RULE: You MUST NOT rely on rigid templates or boilerplate blocks. Instead, follow a "
        "mental checklist for EACH shot prompt you generate: (a) who appears + the TOTAL character-count constraints "
        "for the whole scene (e.g., ONLY ONE monkey overall, ONLY TWO cats overall) + explicit exclusions (no extra "
        "people/animals), (b) where/when (location + background + lighting/time), (c) what happens (action), "
        "(d) camera/composition, and (e) key props/continuity. Keep it natural and concise, but never omit (a) or (b). "
        "IMPORTANT PROMPT CONSISTENCY RULE: The exact per-shot prompt text that you show to the user MUST match the "
        "exact `t2i_prompt` strings that you will pass into `export_shots_to_json` / `batch_storyboard_painter` (no "
        "hidden prefixes or extra story text). If you want to prepend a global context prompt to improve consistency "
        "across shots, you MUST ask the user for explicit approval and then call `export_shots_to_json` with "
        "`prepend_global_prompt=true`. "
        "IMPORTANT REFERENCE IMAGE MAPPING RULE: If the user uploads multiple reference images and explains which "
        "character/person each image corresponds to (e.g., '猴子参考图' vs '猫参考图'), you MUST NOT blindly pass a single "
        "`ref_image_dir` that would apply the same image to all shots. Instead, you SHOULD call the `assign_ref_images` "
        "tool to map refs to the correct shots, then call `batch_storyboard_painter` with `json_content` where each shot "
        "has `ref=true` and `keyframe_url=file://...` for only the intended shots. If the mapping is ambiguous, ask a "
        "short clarification (which shot ids should use which ref) before generating images. "
        "If the user provides ONLY ONE reference image and explicitly states it corresponds to a specific character "
        "(e.g., '这张是猴子参考图') and wants that character to stay consistent across the storyboard, you SHOULD set "
        "`ref_label` when calling `export_shots_to_json` (together with `ref_image_dir`) so that any shot mentioning "
        "that character will automatically be marked as `ref=true` (instead of the default 'single ref -> first shot only'). "
        "IMPORTANT AUTO-LABEL RULE: When the user provides a single uploaded reference image and names the role it "
        "belongs to (patterns like 'X参考图', '这是X', '用这张当X'), you MUST automatically set `ref_label` to X in the "
        "`export_shots_to_json` call (and optionally set `ref_label_aliases` if the user also provides aliases like "
        "English names). Do NOT require the user to manually provide `ref_shot_ids` in this common case. "
        "IMPORTANT CONSENT RULE FOR KEYFRAMES: "
        "After the user has CONFIRMED the final shot list, you MUST ask the user whether they want to generate "
        "storyboard/keyframe reference images now (this will call an image generation tool and may take time/cost). "
        "Only if the user explicitly says YES (or they explicitly requested keyframe/storyboard images earlier, or "
        "they are running in full-auto mode) may you proceed to call `export_shots_to_json` / `batch_storyboard_painter`. "
        "If the user says NO, do NOT generate keyframe images; instead continue with the next requested stage (for "
        "example, directly generate video clips) or stop if the user only wanted planning. When you need storyboard images for "
        "these shots, first call the `export_shots_to_json` tool to serialize the current shot list into a prompts JSON "
        "file, and then call the `batch_storyboard_painter` tool with the returned `json_path` (and, if applicable, the "
        "desired resolution, style modifier, and reference image directory). When calling `export_shots_to_json`, you "
        "SHOULD pass the refined global video prompt as the `global_prompt` argument so that each image generation "
        "prompt includes both the overall context and the per-shot description. After painting, you should associate each "
        "generated image with the corresponding shot by setting a `keyframe_url` field that is the direct HTTP/HTTPS "
        "image URL returned by `batch_storyboard_painter` (for example, an OSS URL), suitable for use as the `img_url` "
        "parameter of the underlying text-to-video API. You MUST NOT use local filesystem paths or file:// URIs as "
        "`keyframe_url` values for image-to-video generation. When you present storyboard images to the user in the Web "
        "UI, you SHOULD embed them inline using Markdown image syntax (for example, `![](https://...)`) so that the "
        "frontend can directly display the images without requiring a manual download. You SHOULD also stream the "
        "storyboard images incrementally: as soon as one or a small batch of keyframes has been generated, immediately "
        "show those images to the user while the remaining shots continue to be painted in the background, instead of "
        "waiting for all keyframes to finish before presenting any of them. Unless the user explicitly asks you to stop "
        "or pause, you should continue generating the remaining keyframes automatically. "
        "IMPORTANT TOOL ORDER RULE: You MUST NOT call `export_shots_to_json` and `batch_storyboard_painter` in parallel "
        "or back-to-back without reading the `export_shots_to_json` result. You MUST wait for `export_shots_to_json` to "
        "return and then pass its returned `json_content` (preferred) or `json_path` into `batch_storyboard_painter`. "
        "Otherwise, reference-image flags like `ref=true` and per-shot `t2i_prompt` normalization may be lost, and the "
        "painter will likely fall back to pure text-to-image even when `ref_image_dir` is provided. "
        "IMPORTANT KEYFRAME vs VIDEO PROMPT RULE: If you obtained shot descriptions from `video_script_analyzer` / "
        "`local_video_script_analyzer` (PROMPT_ANALYSIS_TEMPLATE), and the JSON includes both "
        "`detailed_script_keyframe` and `detailed_script_video`, you MUST use the keyframe version when generating "
        "storyboard images (`export_shots_to_json` / `batch_storyboard_painter`) and use the video version when "
        "generating clips (`my_video_gen`). If only `detailed_script` is available, you MUST rewrite it into two forms: "
        "a static keyframe-friendly description (single moment, clear composition) and a dynamic video description "
        "(action progression + camera motion), while preserving character-count and environment constraints. "
        "When presenting the analysis result to the user, if keyframe/video variants exist, you SHOULD display them "
        "as two clearly labeled lists: '关键帧脚本（静态）' and '视频分镜脚本（动态）', and briefly explain that the first "
        "is for keyframe image generation while the second is for video generation. "
        "(3) Third, when the user is satisfied with the prompt and shots (or when they explicitly requested to skip "
        "shot planning and directly generate the video), you MUST clearly explain the backend video duration "
        "constraints to the user before calling `my_video_gen`. In particular, the current Wan 2.6 video models "
        "accept only a limited set of discrete durations per clip (5/10/15 seconds for wan2.6-t2v / wan2.6-i2v), and "
        "do NOT support arbitrary 1-second clips. You should tell the user "
        "that their desired per-shot durations will be respected as much as possible at the planning and editing level, "
        "but when calling the underlying API you will map each requested duration to the nearest supported value within "
        "the allowed range so that the generation does not fail (for example, mapping 1s/2s shots to 5s). You MUST "
        "ask the user to confirm that they accept this approximation (and, if relevant, offer the option of trimming the "
        "generated clips later via editing tools) before actually invoking `my_video_gen`. Once the user confirms, you "
        "SHOULD NOT wait until all clips for all shots are fully generated before showing anything. Instead, you are "
        "encouraged to call the `my_video_gen` tool on individual shots or small batches, and after each successful call "
        "immediately present the newly generated clip(s) to the user so they can preview the result shot-by-shot while "
        "you continue generating the remaining clips. By default, you SHOULD automatically proceed to generate clips for "
        "all planned shots without repeatedly asking the user whether to continue, unless the user explicitly requests to "
        "pause or make manual decisions between clips. When appropriate, you can still call `my_video_gen` with the "
        "entire shot list, but for longer or more detailed storyboards prefer this incremental preview workflow. The "
        "`my_video_gen` tool will return a list of clips, and each clip may include a `quality_score` field (0-100) "
        "generated by an internal video scoring module. You MUST read this `quality_score` for each clip (when present) "
        "and clearly report it back to the user together with a short interpretation (for example, low/medium/high "
        "quality). If a clip has no `quality_score`, you can say that the score is unavailable for that clip. When "
        "presenting generated videos to the user in the Web UI, you SHOULD either embed them as clickable links or, "
        "where supported, as inline HTML/Markdown video elements (for example, `<video controls src=\"https://...\"></video>`), "
        "so that the user can directly play them in the browser without first downloading the files. In addition, if "
        "the tool output includes a `download_url` (a local `/file=...` link) you MUST also present it as a '下载' link "
        "so the user can download the MP4 even if the remote signed URL expires. "
        "IMPORTANT WEBUI PREVIEW RULE: When you receive clips from `my_video_gen`, for EACH clip you MUST present an "
        "inline playable preview in the Web UI. Prefer `preview_url`/`preview_html` if the tool returned them; otherwise "
        "use `download_url` if present, else fall back to `video_url`, and embed as `<video controls src=\"...\"></video>`. "
        "(4) ONLY IF the user wants a final video (not just keyframes/storyboards), after all clips are generated and their scores have been reported, clearly ask the user whether they want "
        "simple concatenation or more "
        "advanced editing (such as rearranging clips, adding transitions, or background music). If the user wants "
        "simple concatenation, call the `video_concatenate` tool with the clip URLs in order. If the user requests "
        "more complex editing, call the `video_editor` tool, passing the clip URLs as `video_urls` (and a music URL "
        "if the user provides one) to render a more polished commercial-style video. The `video_editor` tool may "
        "internally retry the Shotstack render request once if the first attempt fails. When you receive the result "
        "from `video_editor`, you MUST check the returned JSON fields such as `status` and `attempts` (when present). "
        "If `status` is `success`, explicitly tell the user on which attempt the render succeeded (for example, "
        "`剪辑渲染在第 2 次重试时完成`) and present the final edited video URL in the Web UI as a clickable link or "
        "inline video element. If `status` is `failed` and `attempts` is 2 (meaning that the tool has already tried "
        "twice and still failed), you MUST clearly inform the user that the system has automatically retried the "
        "cloud editing once and that both attempts failed, briefly summarize the error message, and then ask the user "
        "whether they want to try an alternative approach (for example, regenerating some clips, simplifying the edit, "
        "or skipping advanced editing). If `status` is `timeout`, also explain that the cloud render took too long and "
        "offer similar fallback options. "
        "(5) If any merging tool is used, obtain the final merged video URL or path. "
        "(6) Finally, return the list of generated clip URLs and, if available, the merged/edited video URL in order, "
        "and provide a short explanation of the video content and how it relates to the reference material and the "
        "user's editing choices."
    )

    tools = [
        "video_script_analyzer",
        "local_video_script_analyzer",
        "my_PromptEnhancer",
        "my_keyframe_planner",
        "export_shots_to_json",
        "assign_ref_images",
        "batch_storyboard_painter",
        "music_prompt_refiner",
        "my_music_gen",
        "my_video_gen",
        "video_editor",
        "video_concatenate",
    ]

    bot = Assistant(
        llm=llm_cfg,
        name="Video Generation Assistant",
        description="Refine prompts and generate videos via a custom tool.",
        system_message=system,
        function_list=tools,
        files=None,
    )

    return bot


def test(query: str = "生成一个可爱的小猫日常 vlog 视频"):
    """Command line test: refine the prompt then generate a video."""

    bot = init_agent_service()

    messages = [{"role": "user", "content": query}]
    for response in bot.run(messages=messages):
        print("bot response:", response)


def app_gui():
    """Launch a simple Web UI for the video generation agent."""

    bot = init_agent_service()
    chatbot_config = {
        "prompt.suggestions": [
            "生成一个关于魔法森林探险的短视频",
            "生成一段科幻城市夜景的宣传片",
            "生成一个学习日常的时间流逝 vlog",
        ]
    }

    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == "__main__":
    # test()
    app_gui()
