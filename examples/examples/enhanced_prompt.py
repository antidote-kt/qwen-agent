import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from http import HTTPStatus
from typing import Optional
import uuid

import requests
import dashscope
import json5
import pprint
from apimart_key import get_apimart_api_key
from dashscope.aigc.image_synthesis import ImageSynthesis
from dashscope.aigc.video_synthesis import VideoSynthesis

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool


def upload_local_image_to_public_url(local_path: str) -> Optional[str]:
    """将本地图片上传到可公网访问的图床，并返回 URL。

    优先使用 ImgBB 官方 API（https://api.imgbb.com/1/upload）：
    - 需要 API Key（推荐设置环境变量 IMGBB_API_KEY）。
    - 通过表单字段 image 上传二进制或 Base64，返回 data.url / data.display_url。

    若 ImgBB 上传失败或未配置 API Key，则回退到 transfer.sh 作为兜底匿名图床。
    若全部失败，返回 None，调用方会退回纯文本增强。
    """

    if not os.path.exists(local_path):
        print(f"⚠️ 本地图片不存在，无法上传: {local_path}")
        return None
    print(f"✅ 本地图片存在，准备上传: {local_path}")
    file_name = os.path.basename(local_path) or f"image_{uuid.uuid4().hex}.png"

    # 优先使用 ImgBB 图床：优先从配置文件读取，其次环境变量
    imgbb_key = ""
    try:
        from config_loader import get_imgbb_api_key
        imgbb_key = get_imgbb_api_key()
    except (ImportError, FileNotFoundError, ValueError, KeyError):
        pass
    
    if not imgbb_key:
        imgbb_key = os.getenv("IMGBB_API_KEY", "").strip()

    if not imgbb_key:
        print("⚠️ 未配置有效的 ImgBB API Key，ImgBB 上传不可用，已停止上传流程。")
        return None

    try:
        with open(local_path, "rb") as f:
            # 直接以 multipart/form-data 方式上传文件，ImgBB 会自动处理文件名
            files = {"image": (file_name, f)}
            resp = requests.post(
                f"https://api.imgbb.com/1/upload?key={urllib.parse.quote(imgbb_key)}",
                files=files,
                timeout=60,
                headers={
                    "User-Agent": "qwen-agent-imgbb-uploader/1.0",
                },
            )

        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                d = data.get("data", {}) or {}
                url = d.get("url") or d.get("display_url") or d.get("url_viewer")
                if isinstance(url, str) and url.startswith("http"):
                    print(f"✅ 已将本地图片上传到 ImgBB 图床: {url}")
                    return url
            print(f"⚠️ ImgBB 上传失败或 success=false，响应: {data}")
        else:
            print(f"⚠️ ImgBB 上传 HTTP 状态码异常: {resp.status_code}, text={resp.text[:200]}")
    except Exception as e:
        print(f"⚠️ 调用 ImgBB 图床异常，错误: {e}")

    # 不再使用 transfer.sh 等回退图床，直接返回 None
    return None


def file_url_to_local_path(file_url: str) -> str:
    """将 file:// URL 转换成本地文件路径，并处理 Windows 前导斜杠等问题。"""

    local_path = file_url[len("file://") :]
    # URL 解码，处理空格等转义
    local_path = urllib.parse.unquote(local_path)
    # Windows 上 file:///C:/... 会变成 /C:/...，需要去掉开头的 /
    if os.name == "nt" and local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ":":
        local_path = local_path[1:]
    return local_path


def image_file_to_data_uri(local_path: str) -> Optional[str]:
    """读取本地图片文件并转为符合 APIMart 要求的 Base64 Data URI。

    形如：data:image/jpeg;base64,/9j/4AAQSkZJRg...
    若文件不存在或读取失败，返回 None。
    """

    if not os.path.exists(local_path):
        print(f"⚠️ 本地图片不存在，无法转为 Base64: {local_path}")
        return None

    # 如果传入的是目录（例如 Gradio 上传的临时目录），尝试从中挑选一张图片
    if os.path.isdir(local_path):
        candidates = [
            f
            for f in sorted(os.listdir(local_path))
            if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg", ".webp", ".gif"]
        ]
        if not candidates:
            print(f"⚠️ 目录中未找到可用图片文件: {local_path}")
            return None
        local_path = os.path.join(local_path, candidates[0])

    mime, _ = mimetypes.guess_type(local_path)
    if not mime:
        mime = "image/jpeg"

    # 文件大小限制：避免生成过大的 data URI（默认 5MB）
    MAX_IMAGE_BYTES = 5 * 1024 * 1024
    try:
        size = os.path.getsize(local_path)
    except Exception as e:
        print(f"⚠️ 获取图片大小失败，无法转为 Base64: {local_path}, 错误: {e}")
        return None
    if size > MAX_IMAGE_BYTES:
        print(f"⚠️ 图片过大({size} bytes)，已超过 5MB 限制，跳过 Base64 转换: {local_path}")
        return None

    try:
        with open(local_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ 读取本地图片失败，无法转为 Base64: {local_path}, 错误: {e}")
        return None

    data_uri = f"data:{mime};base64,{b64}"
    # 注意：不要在终端打印完整 data URI（可能非常长），只打印长度与前缀预览
    preview = data_uri[:120] + "..." if len(data_uri) > 120 else data_uri
    print(f"✅ Base64 转换成功: mime={mime}, bytes={size}, data_uri_len={len(data_uri)}")
    print(f"✅ Base64 Data URI 预览: {preview}")
    return data_uri


def _pick_local_image_file(path_or_dir: str) -> Optional[str]:
    """从文件或目录中选取一张可用图片文件路径（用于上传/转 data URI）。"""
    if not path_or_dir:
        return None
    if not os.path.exists(path_or_dir):
        return None
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        for f in sorted(os.listdir(path_or_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                return os.path.join(path_or_dir, f)
    return None


# 辅助函数-调用实际的api，返回结构化的正/负提示词
def CallApi(prompt: str, img_url: str):

    # 指令提示词：要求模型直接返回 JSON，包含 positive_prompt 和 negative_prompt 两个字段
    instruction1 = (
        "假设你是一个提示词增强助手，请你丰富并扩写以下这段用于文字生成图片模型的提示词，以达到更好的生成效果。"
        "你必须以 JSON 对象的形式返回结果，格式如下："
        "{""positive_prompt"": ""..."", ""negative_prompt"": ""...""}。"
        "其中 positive_prompt 用来正向描述需要生成的画面，negative_prompt 用来罗列所有需要避免的内容（例如：低分辨率、过曝、解剖错误、丑陋、畸形等）。"
        "只返回 JSON，不要添加任何解释性文字。"
    )
    # 需要输入多模态信息时的指令
    instruction2 = (
        "假设你是一个提示词增强助手，请你结合参考图片，丰富并扩写以下这段用于文字生成图片模型的提示词，以达到更好的生成效果。"
        "你必须以 JSON 对象的形式返回结果，格式如下："
        "{""positive_prompt"": ""..."", ""negative_prompt"": ""...""}。"
        "其中 positive_prompt 需要同时参考文字和图片内容，negative_prompt 用来罗列所有需要避免的内容（例如：低分辨率、过曝、解剖错误、丑陋、畸形等）。"
        "只返回 JSON，不要添加任何解释性文字。"
    )

    # 统一判断是否需要多模态
    isNeedMultiMod = bool(img_url)
    image_used = False
    image_skip_reason = ""

    # 如果是本地上传的图片（file:// 或直接本地路径），转换为 Base64 Data URI，直接发给 Apimart
    if isNeedMultiMod and isinstance(img_url, str) and (
        img_url.startswith("file://") or os.path.exists(img_url)
    ):
        local_path = file_url_to_local_path(img_url) if img_url.startswith("file://") else img_url
        picked = _pick_local_image_file(local_path)
        if not picked:
            print(f"⚠️ 未找到可用的本地图片文件，已降级为纯文本增强: {local_path}")
            img_url = None
            isNeedMultiMod = False
            image_skip_reason = "local_image_not_found"
        else:
            # 不依赖图床：本地图片直接转 data URI 并在工具内部发给 API（不回传给 Agent，避免 token 爆炸）
            # 仍需控制大小，避免请求体过大导致超时/被拒。
            MAX_LOCAL_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB
            try:
                size = os.path.getsize(picked)
            except Exception:
                size = None
            if size is not None and size > MAX_LOCAL_IMAGE_BYTES:
                print(
                    f"⚠️ 本地图片过大({size} bytes)，为避免请求过大/超时，已降级为纯文本增强（图片未使用）。"
                )
                img_url = None
                isNeedMultiMod = False
                image_used = False
                image_skip_reason = "local_image_too_large"
            else:
                data_uri = image_file_to_data_uri(picked)
                if data_uri:
                    img_url = data_uri
                    image_used = True
                else:
                    print("⚠️ 本地图片转换 data URI 失败，已降级为纯文本增强（图片未使用）。")
                    img_url = None
                    isNeedMultiMod = False
                    image_used = False
                    image_skip_reason = "local_image_to_data_uri_failed"

    # 如果用户传入的是 data: URL（通常很长），允许使用，但要控制尺寸，且不要打印/回传该字符串
    if isNeedMultiMod and isinstance(img_url, str) and img_url.strip().startswith("data:"):
        # 粗略限制 data URI 字符串长度，避免请求过大
        # 5MB 二进制图片转 base64 后大约 6.7MB 字符，再加前缀，给一个宽松上限
        MAX_DATA_URI_CHARS = 8_000_000
        if len(img_url) > MAX_DATA_URI_CHARS:
            print("⚠️ 参考图 data URI 过大，已降级为纯文本增强（图片未使用）。")
            img_url = None
            isNeedMultiMod = False
            image_used = False
            image_skip_reason = "data_uri_too_large"
        else:
            image_used = True

    # 仅允许 http(s) URL 或 data URI 进入请求体
    if isNeedMultiMod and isinstance(img_url, str):
        s = img_url.strip()
        if s.startswith("data:"):
            img_url = s
        elif not (s.startswith("http://") or s.startswith("https://")):
            print("⚠️ 参考图不是有效的 http(s) URL，已降级为纯文本增强（图片未使用）。")
            img_url = None
            isNeedMultiMod = False
            image_used = False
            image_skip_reason = "invalid_image_url_format"
        else:
            img_url = s
            image_used = True

    # 走到这里：要么没有图片，要么是公网图片 URL，要么是 Base64 Data URI，用 Apimart 的 gpt-5 做增强
    def _build_payload(include_image: bool) -> dict:
        payload = {
            "model": "gpt-5",
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": instruction2 if include_image else instruction1,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
            ],
        }
        if include_image and img_url:
            payload["input"][1]["content"].append(
                {
                    "type": "input_image",
                    "image_url": img_url,
                }
            )
        return payload

    def _safe_payload_for_log(payload: dict) -> dict:
        """Redact large/secret fields so logs are safe and readable."""
        try:
            p = json.loads(json.dumps(payload))
        except Exception:
            return {"_payload_unserializable": True}

        # Redact user/system texts to short previews, keep lengths
        try:
            for item in p.get("input", []) or []:
                for c in item.get("content", []) or []:
                    if c.get("type") == "input_text" and isinstance(c.get("text"), str):
                        txt = c["text"]
                        c["text_len"] = len(txt)
                        c["text_preview"] = txt[:200]
                        c["text"] = "<redacted>"
                    if c.get("type") == "input_image" and isinstance(c.get("image_url"), str):
                        u = c["image_url"]
                        if u.startswith("data:"):
                            c["image_url"] = f"<data_uri len={len(u)}>"
                        else:
                            c["image_url"] = u[:200]
        except Exception:
            pass
        return p

    def _log_apimart_request(payload: dict, reason: str):
        try:
            safe = _safe_payload_for_log(payload)
            print(f"[Apimart] request debug ({reason}):")
            print(json.dumps(safe, ensure_ascii=False, indent=2)[:4000])
        except Exception:
            pass

    # 统一使用与其它模块相同的 APIMart Key：
    # 出于安全考虑：不在仓库中硬编码 Key；请通过环境变量提供。
    url = "https://api.apimart.ai/v1/responses"
    token = (get_apimart_api_key() or "").strip()
    if not token:
        raise RuntimeError('Missing APIMART_API_KEY. Set env APIMART_API_KEY or put it in repo-root "api key.txt".')

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    def _post_with_retry(payload: dict) -> Optional[requests.Response]:
        # 增加超时与重试，避免网络抖动/长时间挂起
        # - 支持通过环境变量覆盖：APIMART_CONNECT_TIMEOUT / APIMART_READ_TIMEOUT / APIMART_MAX_RETRIES
        try:
            connect_timeout = float(os.getenv("APIMART_CONNECT_TIMEOUT", "10") or 10)
        except Exception:
            connect_timeout = 10.0
        try:
            read_timeout = float(os.getenv("APIMART_READ_TIMEOUT", "60") or 60)
        except Exception:
            read_timeout = 60.0
        try:
            max_attempts = int(os.getenv("APIMART_MAX_RETRIES", "5") or 5)
        except Exception:
            max_attempts = 5
        if max_attempts < 1:
            max_attempts = 1

        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(connect_timeout, read_timeout),
                )
                return resp
            except Exception as e:
                last_err = str(e)
            if attempt < max_attempts:
                import time
                # 指数退避（上限 20s），减少对服务端的压力
                delay = min(2 ** attempt, 20)
                time.sleep(delay)
            else:
                print("⚠️ Apimart 接口错误! ", last_err)
                _log_apimart_request(payload, reason=f"exception_after_retries:{last_err}")
                return None

    # 先尝试带图；若 4xx（多为入参问题）则降级为纯文本增强并提示“图片未使用”
    response = None
    if isNeedMultiMod:
        payload_img = _build_payload(include_image=True)
        response = _post_with_retry(payload_img)
        if response is None:
            return None
        if 400 <= response.status_code < 500:
            print(
                f"⚠️ Apimart 4xx: {response.status_code}. 已降级为纯文本增强（图片未使用）。"
                f" resp={response.text[:200]}"
            )
            _log_apimart_request(payload_img, reason=f"4xx_with_image:{response.status_code}")
            image_used = False
            if not image_skip_reason:
                image_skip_reason = f"apimart_rejected_image_{response.status_code}"
            payload_text = _build_payload(include_image=False)
            response = _post_with_retry(payload_text)
            if response is None:
                return None
    else:
        payload_text = _build_payload(include_image=False)
        response = _post_with_retry(payload_text)
        if response is None:
            return None

    if response.status_code != 200:
        # 非 200（包括 5xx/524 等）：降级，避免打断主流程
        print(f"⚠️ Apimart 接口错误! status={response.status_code}, text={response.text[:200]}")
        try:
            _log_apimart_request(payload_text if not isNeedMultiMod else payload_img, reason=f"non_200:{response.status_code}")
        except Exception:
            pass
        return None

    data = response.json()
    raw_text = data.get("output", [{} , {}])[1]["content"][0]["text"]

    # 优先尝试按 JSON 解析（期望模型严格按 positive_prompt/negative_prompt 返回）
    positive_prompt = ""
    negative_prompt = ""
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            positive_prompt = str(parsed.get("positive_prompt", "") or "")
            negative_prompt = str(parsed.get("negative_prompt", "") or "")
    except Exception:
        # 如果不是合法 JSON，就把整段文本当成正向提示词，负面为空，避免工具报错
        positive_prompt = raw_text or ""
        negative_prompt = ""

    # 打印方便在终端观察
    print("[PromptEnhancer] positive_prompt:\n", positive_prompt)
    if negative_prompt:
        print("[PromptEnhancer] negative_prompt:\n", negative_prompt)

    return {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "image_used": bool(image_used and isNeedMultiMod),
        "image_skip_reason": image_skip_reason,
    }


@register_tool("my_PromptEnhancer")
class PromptEnhancer(BaseTool) :
    # 提示器强化，根据用户分镜剧本进行提示词丰富续写，形成文生图 + 图生视频提示词
    
    description = (
        "用于视频生成的提示词增强工具。输入文字描述和参考图片，然后返回增强后的提示词。"
        "你必须在输出的正向提示词中明确：角色数量约束（避免多出角色/重复角色）以及每个镜头所需的环境/场景描述"
        "（地点、背景、时间/光照、关键道具/氛围），并保持跨镜头环境一致性（除非用户要求换场）。"
    )
    
    parameters = [
        {
            "name" : "prompt",
            "type" : "string",
            "description" : "对于想要生成视频的文字描述",
            "required" : True
        },
        {
            "name" : "referedImage",
            "type" : "string",
            "description" : "用于生成视频的参考图片，可以是公网图片 URL，或 WebUI 上传后得到的 file:// 本地路径。",
            "required" : False
        }
    ]
    
    
    def call(self, params: str, **kwargs) -> str :
        # 解析json参数
        args: dict = json5.loads(params)
        prompt = args.get("prompt", "") or ""

        # 强制补充生成要求：数量约束 + 环境描述（不改变剧情，只约束表达完整性）
        prompt_for_api = (
            f"{prompt}\n\n"
            "硬性要求（必须遵守）：\n"
            "1) 角色数量约束：明确全片的总角色数量与每类角色数量（例如“仅 1 只猴子（同一只贯穿全片）/仅 2 只猫：Bobo 和 Toto”），并写明不要出现额外人物/动物/重复角色。\n"
            "2) 环境描述：为每个镜头补齐地点、背景要素、时间/光照、关键道具/氛围；跨镜头保持环境一致，除非明确换场。\n"
        )
        pos_prompt = ""
        neg_prompt = ""

        # 调用底层增强接口，注意做失败兜底
        image_used = False
        image_skip_reason = ""
        if "referedImage" in args:
            result = CallApi(prompt_for_api, args["referedImage"])
        else:
            result = CallApi(prompt_for_api, None)

        if isinstance(result, dict):
            pos_prompt = (result.get("positive_prompt") or "").strip()
            neg_prompt = (result.get("negative_prompt") or "").strip()
            image_used = bool(result.get("image_used", False))
            image_skip_reason = str(result.get("image_skip_reason", "") or "")

        # 如果底层接口出错或返回空结果，则优雅降级：至少返回原始 prompt，避免 Agent 认为工具完全失败
        if not pos_prompt:
            pos_prompt = prompt
        if not isinstance(neg_prompt, str):
            neg_prompt = ""

        # 为了兼容旧逻辑，同时返回：
        # - enhanced_prompt：正向 + 可选负向提示词整体文本（供直接使用）
        # - positive_prompt / negative_prompt：结构化字段，便于 Agent 精确控制
        # 同时输出两种用途的版本：
        # - keyframe_*：更偏“单帧画面”（用于关键帧/分镜图）
        # - video_*：更偏“动态分镜”（用于视频生成时的动作/镜头运动表达）
        keyframe_hint = (
            "（关键帧版本要求：描述一个清晰的单帧瞬间画面；强调人物外观一致性、数量约束、环境/光照、构图与关键道具；"
            "避免使用大量连续动作或复杂镜头运动措辞。）"
        )
        video_hint = (
            "（视频分镜版本要求：描述动作随时间推进、主体运动与镜头运动（推拉摇移/跟拍/慢动作等）；"
            "同时严格遵守数量约束与环境一致性；不要添加额外人物/动物/字幕水印。）"
        )

        keyframe_pos = f"{pos_prompt}\n{keyframe_hint}".strip()
        video_pos = f"{pos_prompt}\n{video_hint}".strip()

        if neg_prompt:
            enhanced_prompt = f"{video_pos}\n\n负面提示词 (negative_prompt)：\n{neg_prompt}"
            enhanced_prompt_keyframe = f"{keyframe_pos}\n\n负面提示词 (negative_prompt)：\n{neg_prompt}"
            enhanced_prompt_video = enhanced_prompt
        else:
            enhanced_prompt = video_pos
            enhanced_prompt_keyframe = keyframe_pos
            enhanced_prompt_video = video_pos

        return json5.dumps(
            {
                "enhanced_prompt": enhanced_prompt,
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "enhanced_prompt_keyframe": enhanced_prompt_keyframe,
                "enhanced_prompt_video": enhanced_prompt_video,
                "keyframe_positive_prompt": keyframe_pos,
                "video_positive_prompt": video_pos,
                "image_used": image_used,
                "image_skip_reason": image_skip_reason,
            },
            ensure_ascii=False,
        )
    


def TestApi() : 
    # 多模态
    CallApi("玻璃上映着女人的倒影", "https://picsum.photos/seed/16246999/1024/1024")
    # 只有提示词
    CallApi("人在路上走", None)
    pass

if __name__ == "__main__" :
    TestApi()
    pass
