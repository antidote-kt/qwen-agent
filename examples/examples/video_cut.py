import json
import time
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from config_loader import get_shotstack_api_key

# ==========================================
# 1. 常量定义
# ==========================================

SHOTSTACK_API_KEY = get_shotstack_api_key()
SHOTSTACK_ENV = "v1"  # stage（试验模型） / v1(正式模型)
SHOTSTACK_BASE_URL = f"https://api.shotstack.io/edit/{SHOTSTACK_ENV}"
RENDER_URL = f"{SHOTSTACK_BASE_URL}/render"

@register_tool('video_editor')
class CommercialVideoEditor(BaseTool):
    description = '视频剪辑工具。接收一组视频链接和背景音乐链接，自动拼接并渲染成完整视频。'
    parameters = [{
        'name': 'video_urls',
        'type': 'array',
        'description': '视频素材的URL链接列表',
        'required': True
    }, {
        'name': 'music_url',
        'type': 'string',
        'description': '背景音乐的URL链接',
        'required': False
    }, {
        'name': 'audio_effect',
        'type': 'string',
        'description': '可选：Shotstack 音频 clip 的 effect（必须是 Shotstack 允许的枚举值；默认不设置）。',
        'required': False
    }, {
        'name': 'music_duration',
        'type': 'number',
        'description': '可选：仅使用背景音乐的前 N 秒（例如 10）。默认使用整段视频长度。',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        # 解析参数
        params = json.loads(params)
        video_urls = params.get('video_urls', [])
        music_url = params.get('music_url', '')
        audio_effect = (params.get('audio_effect') or '').strip()
        music_duration = params.get('music_duration', None)

        if not video_urls:
            return json.dumps({'error': '未提供视频素材'})

        # Shotstack is a cloud service and cannot access local file:// URIs.
        bad_urls = [u for u in (video_urls or []) if isinstance(u, str) and u.strip().startswith("file://")]
        if isinstance(music_url, str) and music_url.strip().startswith("file://"):
            bad_urls.append(music_url)
        if bad_urls:
            return json.dumps(
                {
                    "status": "failed",
                    "attempts": 0,
                    "error": "Shotstack 不支持 file:// 本地路径素材，请提供可公网访问的 http(s) URL，或改用本地 ffmpeg 方式合成。",
                    "bad_assets": bad_urls,
                },
                ensure_ascii=False,
            )

        print(f"--- [工具启动] 开始处理，素材数量: {len(video_urls)} ---")

        # -------------------------------------------------------
        # 核心逻辑：构建 Timeline
        # -------------------------------------------------------
        video_clips = []
        current_start_time = 0.0
        clip_length = 5.0 # 演示用：每个片段取5秒（可删除，目前正在试用中）

        for url in video_urls:
            clip = {
                "asset": {
                    "type": "video",
                    "src": url.strip()
                },
                "start": current_start_time,
                "length": clip_length,
                "fit": "cover",
                "transition": {"in": "fade", "out": "fade"}
            }
            video_clips.append(clip)
            current_start_time += clip_length

        tracks = [{"clips": video_clips}]

        # --- 添加音频并限制时长 ---
        if music_url:
            # Shotstack 对 effect 有严格枚举限制（且不支持 fadeOut 这样的值）。
            # 若传入了不合法的 effect，会导致提交 400 Bad Request。
            allowed_effects = {
                "none",
                "zoomIn", "zoomInSlow", "zoomInFast",
                "zoomOut", "zoomOutSlow", "zoomOutFast",
                "slideLeft", "slideLeftSlow", "slideLeftFast",
                "slideRight", "slideRightSlow", "slideRightFast",
                "slideUp", "slideUpSlow", "slideUpFast",
                "slideDown", "slideDownSlow", "slideDownFast",
            }
            audio_clip = {
                "asset": {"type": "audio", "src": music_url},
                "start": 0,
                # 关键修正：让音乐长度等于视频总长度，避免黑屏
                "length": current_start_time,
            }
            if music_duration is not None:
                try:
                    md = float(music_duration)
                    if md > 0:
                        audio_clip["length"] = min(current_start_time, md)
                except Exception:
                    pass
            if audio_effect and audio_effect in allowed_effects:
                audio_clip["effect"] = audio_effect
            elif audio_effect:
                print(f"⚠️ audio_effect={audio_effect} 不在 Shotstack 允许列表中，将忽略该参数。")

            tracks.append({
                "clips": [{
                    **audio_clip
                }]
            })

        edit_payload = {
            "timeline": {
                "background": "#000000",
                "tracks": tracks
            },
            "output": {
                "format": "mp4",
                "resolution": "sd"
            }
        }

        # --- 打印 JSON 方便调试 ---
        print("\n--- [Debug] 发送给 Shotstack 的 JSON ---")
        print(json.dumps(edit_payload, indent=2))
        print("--------------------------------------\n")

        # -------------------------------------------------------
        # 发送请求
        # -------------------------------------------------------
        headers = {
            "Content-Type": "application/json",
            "x-api-key": SHOTSTACK_API_KEY
        }

        # -------------------------------------------------------
        # 增加简单重试机制：
        # - 若提交或渲染失败，会自动最多重试 1 次（共 2 次）
        # - 超过 2 次仍失败，则将失败原因返回给上层，由 Agent 告知用户
        # -------------------------------------------------------
        max_attempts = 2
        last_error_msg = None

        for attempt in range(1, max_attempts + 1):
            print(f"--- [尝试 {attempt}/{max_attempts}] 提交剪辑任务 ---")
            try:
                response = requests.post(RENDER_URL, json=edit_payload, headers=headers)

                if response.status_code != 201:
                    last_error_msg = f"提交失败: {response.text}"
                    print(last_error_msg)
                    if attempt < max_attempts:
                        print("将进行下一次重试...")
                        continue
                    else:
                        return json.dumps({
                            'status': 'failed',
                            'attempts': attempt,
                            'error': last_error_msg
                        })

                render_id = response.json()['response']['id']
                print(f"--- [任务提交成功] ID: {render_id} ---")

                # ---------------------------------------------------
                # 策略选择：为了 Demo 效果，这里我们选择“轮询等待”
                # 这样 Qwen 可以直接把视频链接吐给用户
                # ---------------------------------------------------
                print("正在等待云端渲染...")
                for i in range(20):
                    time.sleep(3)
                    status_res = requests.get(f"{RENDER_URL}/{render_id}", headers=headers)
                    status_data = status_res.json()
                    status = status_data['response']['status']

                    print(f"Checking status... {status}")

                    if status == 'done':
                        video_url = status_data['response']['url']
                        return json.dumps({
                            'status': 'success',
                            'attempts': attempt,
                            'video_url': video_url
                        })
                    elif status == 'failed':
                        last_error_msg = status_data['response'].get('error', 'unknown error')
                        print(f"本次渲染失败: {last_error_msg}")
                        break  # 跳出轮询，按失败流程处理

                # 若到这里，要么是超时，要么是上面 break 的失败
                if last_error_msg is None:
                    last_error_msg = '渲染时间过长，请稍后检查'
                    print(last_error_msg)

                if attempt < max_attempts:
                    print("将进行下一次重试...")
                    continue
                else:
                    return json.dumps({
                        'status': 'failed',
                        'attempts': attempt,
                        'error': last_error_msg
                    })

            except Exception as e:
                last_error_msg = str(e)
                print(f"调用 Shotstack 接口异常: {last_error_msg}")
                if attempt < max_attempts:
                    print("将进行下一次重试...")
                    continue
                else:
                    return json.dumps({
                        'status': 'failed',
                        'attempts': attempt,
                        'error': last_error_msg
                    })

        # 理论上不应到达这里，兜底返回
        return json.dumps({
            'status': 'failed',
            'attempts': max_attempts,
            'error': last_error_msg or 'unknown error'
        })
        

if __name__ == '__main__':
    # 1. 实例化你的工具
    tool = CommercialVideoEditor()

    # 2. 准备测试数据 (模拟 Qwen 传过来的 JSON 参数)
    # 这里用的是真实的公网测试素材，直接用就行
    test_params = json.dumps({
        "video_urls": [
            # Google 提供的公共测试视频 1
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
            # Google 提供的公共测试视频 2
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
        ],
        # W3Schools 的公共测试音频 (马叫声，虽然短但是绝对能用)
        # 或者是 GitHub 上稳定的开源音乐
        "music_url": "https://github.com/rafaelreis-hotmart/Audio-Sample-files/raw/master/sample.mp3"
    })

    print(">>> 开始单元测试...")
    
    # 3. 调用工具
    result = tool.call(test_params)
    
    # 4. 打印结果
    print("\n>>> 测试结果:")
    print(result)
