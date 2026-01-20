
import os
import time
import tempfile
from urllib.parse import urlparse
from http import HTTPStatus
import dashscope
from dashscope import VideoSynthesis
from dashscope.utils.oss_utils import upload_file

import requests
import re
from urllib.parse import unquote_plus

# ========== 核心配置（通过环境变量设置 API Key） ==========
# 不再在代码中写死 API Key，请在运行前设置环境变量 DASHSCOPE_API_KEY
# dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1' # 根据需要开启

# def sample_async_call(prompt, input_img_url=None):
#     """
#     自适应文生视频和图生视频的函数
#     :param prompt: 视频提示词
#     :param input_img_url: 可选图片URL。如果有值则走I2V，无值则走T2V。
#     """
#     try:
#         # ========== 1. 选择模型逻辑（图片非必要） ==========
#         # 如果有图片，使用 Wan 2.6 图生视频模型；否则使用文生视频模型
#         if input_img_url:
#             model_name = 'wan2.6-i2v'
#             call_kwargs = {"img_url": input_img_url}
#             mode_desc = "图生视频 (I2V)"
#         else:
#             model_name = 'wan2.6-t2v'
#             call_kwargs = {}
#             mode_desc = "文生视频 (T2V)"

#         print(f"🚀 启动模式: {mode_desc} | 模型: {model_name}")

#         # ========== 2. 提交异步任务 ==========
#         rsp = VideoSynthesis.async_call(
#             model=model_name,
#             prompt=prompt,
#             duration=5,        # 视频时长
#             resolution='720P', # 分辨率
#             **call_kwargs      # 动态传入图片参数
#         )

#         print("=== 任务提交响应 ===")
#         if rsp.status_code != HTTPStatus.OK:
#             print(f'❌ 任务提交失败: status_code={rsp.status_code}, code={rsp.code}, message={rsp.message}')
#             return

#         task_id = rsp.output.task_id
#         print(f"✅ 任务提交成功，Task ID: {task_id}")

#         # ========== 3. 轮询任务状态 ==========
#         max_retry = 60  # 最多等待5分钟
#         retry_count = 0
#         video_url = None
#         task_success = False

#         print("\n=== 等待视频生成 ===")
#         while retry_count < max_retry:
#             # fetch仅传rsp对象
#             status = VideoSynthesis.fetch(rsp)
            
#             if status.status_code != HTTPStatus.OK:
#                 print(f'❌ 查询任务失败: status_code={status.status_code}, message={status.message}')
#                 break

#             task_status = status.output.task_status
#             print(f"当前状态: {task_status}（{retry_count+1}/{max_retry}）")

#             if task_status == "SUCCEEDED":
#                 video_url = status.output.video_url
#                 task_success = True
#                 break
#             elif task_status == "FAILED":
#                 fail_msg = getattr(status.output, 'message', "未知失败原因")
#                 print(f'❌ 视频生成失败: {fail_msg}')
#                 break
#             elif task_status in ["PENDING", "RUNNING"]:
#                 retry_count += 1
#                 time.sleep(5)
#             else:
#                 print(f'⚠️ 未知任务状态: {task_status}')
#                 time.sleep(5)

#         # ========== 4. 输出最终结果 ==========
#         if task_success and video_url:
#             print(f"\n🎉 视频生成成功！({mode_desc})")
#             print(f"🔗 视频链接: {video_url}")
#         elif retry_count >= max_retry:
#             print(f"\n⏰ 任务超时，Task ID: {task_id}")
#         else:
#             print("\n❌ 视频生成未完成")

#     except Exception as e:
#         print(f"\n❌ 代码执行异常: {str(e)}")
#         import traceback
#         traceback.print_exc()
def sample_async_call(prompt, input_img_url=None, duration: int = 5, audio_url: str | None = None):
    """
    重构后的函数，返回结果字典，供前端调用。
    返回格式: {
        'status': 'success'/'error'/'timeout',
        'message': '描述信息',
        'video_url': '视频链接 (如果成功)'
    }
    """
    try:
        # 0. 确保每次调用前都正确设置了 DashScope API Key
        # 优先级：配置文件 > 环境变量 > 硬编码
        if not dashscope.api_key or not dashscope.api_key.strip():
            try:
                from config_loader import get_dashscope_api_key
                dashscope.api_key = get_dashscope_api_key()
            except (ImportError, FileNotFoundError, ValueError, KeyError):
                # 如果配置文件不存在或读取失败，回退到环境变量
                dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not dashscope.api_key:
            error_msg = "未配置有效 DASHSCOPE_API_KEY，请在 api_keys.json 配置文件或环境变量中设置后再重试。"
            print(f'❌ {error_msg}')
            return {'status': 'error', 'message': error_msg, 'video_url': None}

        def _short_url(u: str, keep: int = 80) -> str:
            if not isinstance(u, str) or not u:
                return ""
            return u if len(u) <= keep else (u[:keep] + "...")

        def _is_http_url(u: str) -> bool:
            try:
                p = urlparse(u)
                return p.scheme in ("http", "https")
            except Exception:
                return False

        def _looks_like_apimart_upload(u: str) -> bool:
            try:
                host = urlparse(u).netloc.lower()
            except Exception:
                host = ""
            return host.endswith("upload.apimart.ai") or host.endswith("apimart.ai")

        def _download_image_to_temp(url: str, max_bytes: int = 5 * 1024 * 1024) -> str:
            r = requests.get(url, stream=True, timeout=(10, 30))
            r.raise_for_status()

            suffix = ""
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "png" in ctype:
                suffix = ".png"
            elif "webp" in ctype:
                suffix = ".webp"
            elif "jpeg" in ctype or "jpg" in ctype:
                suffix = ".jpg"

            fd, path = tempfile.mkstemp(prefix="dashscope_ref_", suffix=suffix)
            os.close(fd)

            written = 0
            try:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValueError(f"ref image too large: {written} bytes > {max_bytes} bytes")
                        f.write(chunk)
                return path
            except Exception:
                try:
                    os.remove(path)
                except Exception:
                    pass
                raise

        def _proxy_ref_image_to_dashscope_oss(model: str, ref_url: str) -> str | None:
            if not ref_url or not _is_http_url(ref_url):
                return None

            try:
                tmp_path = _download_image_to_temp(ref_url)
            except Exception as e:
                print(f"⚠️ 参考图下载失败，无法转存到 DashScope OSS: {_short_url(ref_url)}; err={e}")
                return None

            try:
                oss_url = upload_file(model=model, upload_path="file://" + tmp_path, api_key=dashscope.api_key)
                if oss_url:
                    print(f"✅ 参考图已转存到 DashScope OSS: {_short_url(oss_url)}")
                return oss_url
            except Exception as e:
                print(f"⚠️ 参考图转存到 DashScope OSS 失败: err={e}")
                return None
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        def _normalize_windows_file_url_for_dashscope(u: str) -> str:
            # DashScope SDK treats file:// URLs specially, but on Windows a common form is:
            #   file:///C:/Users/...  -> urlparse().path == "/C:/Users/..."
            # which becomes a non-existent path. Convert it to:
            #   file://C:/Users/...
            if not isinstance(u, str):
                return u
            if u.startswith("file:///") and re.match(r"^file:///([A-Za-z]:/)", u):
                return "file://" + u[len("file:///") :]
            return u

        def _local_path_from_file_url(u: str) -> str | None:
            if not isinstance(u, str) or not u.startswith("file://"):
                return None
            p = urlparse(u)
            if p.netloc:
                return p.netloc + unquote_plus(p.path)
            return unquote_plus(p.path)

        def _normalize_media_url(u: str | None) -> str | None:
            """Normalize http(s)/file:// or Windows path into a DashScope-friendly URL."""
            if not isinstance(u, str):
                return None
            u0 = u.strip()
            if not u0:
                return None
            if re.match(r"^[A-Za-z]:[\\\\/]", u0):
                u0 = "file://" + u0.replace("\\", "/")
            u0 = _normalize_windows_file_url_for_dashscope(u0)
            if u0.startswith("file://"):
                lp = _local_path_from_file_url(u0)
                if lp and re.match(r"^/[A-Za-z]:/", lp):
                    lp2 = lp[1:]
                    u0 = "file://" + lp2.replace("\\", "/")
                    lp = lp2
                if lp and not os.path.exists(lp):
                    print(f"⚠️ 音频文件不存在，将忽略 audio_url: {lp}")
                    return None
            return u0

        # 1. 选择模型逻辑（图片非必要）
        # Aliyun Model Studio (DashScope) Wan 2.6:
        # - 文生视频: wan2.6-t2v
        # - 图生视频: wan2.6-i2v
        if input_img_url:
            # Accept either:
            # - http(s) URL
            # - file:// URL
            # - plain Windows path like C:\a\b.png
            u0 = str(input_img_url)
            if re.match(r"^[A-Za-z]:[\\\\/]", u0):
                input_img_url = "file://" + u0.replace("\\", "/")
            else:
                input_img_url = u0

            input_img_url = _normalize_windows_file_url_for_dashscope(str(input_img_url))
            # If it's a local file URL, verify it exists; otherwise fallback to T2V.
            if str(input_img_url).startswith("file://"):
                local_path = _local_path_from_file_url(str(input_img_url))
                # Some Windows file URLs may produce a leading '/' path; normalize those too.
                if local_path and re.match(r"^/[A-Za-z]:/", local_path):
                    local_path2 = local_path[1:]
                    input_img_url = "file://" + local_path2.replace("\\", "/")
                    local_path = local_path2
                if local_path and not os.path.exists(local_path):
                    print(f"⚠️ 参考图文件不存在，将降级为文生视频: {local_path}")
                    input_img_url = None
                    model_name = 'wan2.6-t2v'
                    call_kwargs = {}
                    mode_desc = "文生视频 (T2V)"
                else:
                    model_name = 'wan2.6-i2v'
                    call_kwargs = {"img_url": input_img_url}
                    mode_desc = "图生视频 (I2V)"
            else:
                model_name = 'wan2.6-i2v'
                call_kwargs = {"img_url": input_img_url}
                mode_desc = "图生视频 (I2V)"
        else:
            model_name = 'wan2.6-t2v'
            call_kwargs = {}
            mode_desc = "文生视频 (T2V)"

        # Optional: attach external audio track (audio-conditioned video).
        audio_url_n = _normalize_media_url(audio_url)
        if audio_url_n:
            call_kwargs["audio_url"] = audio_url_n

        # 1.5 基本的时长合法性校验 + 与后端允许范围对齐
        try:
            duration = int(duration)
        except Exception:
            duration = 5
        if duration <= 0:
            duration = 5

        # DashScope Wan 2.6 视频生成要求 duration 为离散集合（参考官方文档）：[5, 10, 15]。
        # 因此这里在真正发请求前，将时长映射/裁剪到后端允许集合，避免任意时长导致任务失败。
        allowed = [5, 10, 15]
        if duration not in allowed:
            # 选择与用户期望最接近的合法值
            mapped = min(allowed, key=lambda x: abs(x - duration))
            print(
                f"⚠️ 请求的时长 {duration}s 不在后端允许范围 {allowed} 内，"
                f"将实际使用 {mapped}s 以保证任务能够成功提交。"
            )
            duration = mapped

        print(f"🚀 启动模式: {mode_desc} | 模型: {model_name} | 时长: {duration}s")

        # 2. 提交异步任务
        # Wan 2.6 的参数在文档中以 size=1280*720 为主；这里默认生成 720P（1280*720）。
        # 兼容性：SDK 通过 **kwargs 透传参数，不同模型可能使用 size/resolution 字段。
        common_kwargs = {"duration": duration}
        if model_name.endswith("-t2v"):
            common_kwargs["size"] = "1280*720"
        else:
            common_kwargs["resolution"] = "720P"

        rsp = VideoSynthesis.async_call(
            model=model_name,
            prompt=prompt,
            **common_kwargs,
            **call_kwargs,
        )

        if rsp.status_code != HTTPStatus.OK:
            # 常见问题：图生视频的参考图是外部 URL（例如 upload.apimart.ai），
            # DashScope 在 DataInspection（数据质检）阶段拉取该资源可能超时。
            # 处理策略：若命中该错误，自动把参考图下载到本地并转存到 DashScope OSS 后重试一次。
            if (
                input_img_url
                and getattr(rsp, "code", "") == "InvalidParameter.DataInspection"
                and isinstance(getattr(rsp, "message", ""), str)
                and "Download the media resource timed out" in rsp.message
                and _looks_like_apimart_upload(input_img_url)
            ):
                print(
                    "⚠️ 参考图外链在 DashScope 数据质检阶段下载超时，"
                    "将尝试转存到 DashScope OSS 后重试一次..."
                )
                oss_url = _proxy_ref_image_to_dashscope_oss(model_name, input_img_url)
                if oss_url:
                    call_kwargs2 = dict(call_kwargs)
                    call_kwargs2["img_url"] = oss_url
                    rsp2 = VideoSynthesis.async_call(
                        model=model_name,
                        prompt=prompt,
                        **common_kwargs,
                        **call_kwargs2,
                    )
                    if rsp2.status_code == HTTPStatus.OK:
                        rsp = rsp2
                    else:
                        error_msg = f'任务提交失败(重试后): {rsp2.code} - {rsp2.message}'
                        print(f'❌ {error_msg}')
                        return {'status': 'error', 'message': error_msg, 'video_url': None}
                else:
                    error_msg = f'任务提交失败: {rsp.code} - {rsp.message}'
                    print(f'❌ {error_msg}')
                    return {'status': 'error', 'message': error_msg, 'video_url': None}
            else:
                error_msg = f'任务提交失败: {rsp.code} - {rsp.message}'
                print(f'❌ {error_msg}')
                return {'status': 'error', 'message': error_msg, 'video_url': None}

        task_id = rsp.output.task_id
        print(f"✅ 任务提交成功，Task ID: {task_id}")

        # 3. 轮询任务状态（增加到180次，最多等待15分钟）
        max_retry = 180
        retry_count = 0
        video_url = None

        print(f"\n=== 等待视频生成 (最长{max_retry*5//60}分钟) ===")
        while retry_count < max_retry:
            status = VideoSynthesis.fetch(rsp)
            
            if status.status_code != HTTPStatus.OK:
                error_msg = f'查询任务失败: {status.code} - {status.message}'
                print(f'❌ {error_msg}')
                return {'status': 'error', 'message': error_msg, 'video_url': None}

            task_status = status.output.task_status
            print(f"当前状态: {task_status} ({retry_count+1}/{max_retry})")

            if task_status == "SUCCEEDED":
                video_url = status.output.video_url
                success_msg = f'视频生成成功！({mode_desc})'
                print(f'🎉 {success_msg}')
                print(f'🔗 视频链接: {video_url}')
                return {'status': 'success', 'message': success_msg, 'video_url': video_url}
                
            elif task_status == "FAILED":
                fail_msg = getattr(status.output, 'message', "未知失败原因")
                error_msg = f'视频生成失败: {fail_msg}'
                print(f'❌ {error_msg}')

                return {'status': 'error', 'message': error_msg, 'video_url': None}
                
            elif task_status in ["PENDING", "RUNNING"]:
                retry_count += 1
                time.sleep(5)
            else:
                print(f'⚠️ 未知任务状态: {task_status}')
                time.sleep(5)

        # 超时处理
        timeout_msg = f'任务超时（超过{max_retry*5//60}分钟），Task ID: {task_id}'
        print(f'⏰ {timeout_msg}')
        return {'status': 'timeout', 'message': timeout_msg, 'video_url': None}

    except Exception as e:
        error_msg = f'代码执行异常: {str(e)}'
        print(f'\n❌ {error_msg}')
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': error_msg, 'video_url': None}
if __name__ == '__main__':
    # 从配置文件或环境变量检查 API Key 配置
    if not dashscope.api_key or not dashscope.api_key.strip():
        try:
            from config_loader import get_dashscope_api_key
            dashscope.api_key = get_dashscope_api_key()
        except (ImportError, FileNotFoundError, ValueError, KeyError):
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

    if not dashscope.api_key:
        print("❌ 错误：未配置有效 DASHSCOPE_API_KEY 环境变量！")
    else:
        # 测试场景1：文生视频（图片传入 None）
        # print("--- 测试1：文生视频 ---")
        # sample_async_call(prompt='一个充满科幻感的未来城市，霓虹灯闪烁')

        # 如果你想测试图生视频，可以取消下面的注释：
        # """
        print("\n--- 测试2：图生视频 ---")
        test_img = "https://cdn.translate.alibaba.com/r/wanx-demo-1.png"
        # 示例：生成 8 秒的视频
        sample_async_call(prompt='橘猫在草地上奔跑', input_img_url=test_img, duration=8)
        # """
