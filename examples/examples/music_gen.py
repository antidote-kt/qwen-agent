"""
Workflow:
1) 根据用户输入的视频需要提炼音乐风格与主题提示词（如果需要用户确认则需要在系统工作流处添加）
2) 将提示词整理并提交至suno api进行生成（限定为纯音乐）
3) 用户可下载生成的音频文件

"""

import json
import os
import time
import requests
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

# ===========================================
# dashscope（通过环境变量 DASHSCOPE_API_KEY 配置）


# suno api（通过环境变量 SUNO_API_KEY 配置）
SUNO_API_KEY = os.getenv("SUNO_API_KEY", "").strip()
SUNO_BASE_URL = "https://api.sunoapi.org/api/v1"
# ===========================================

@register_tool("music_prompt_refiner")
class MusicPromptRefiner(BaseTool):
    description = "BGM 风格策划师。根据用户对短视频画面的描述，规划适合的纯音乐风格和标题。"
    parameters = {
        "type": "object",
        "properties": {
            "video_description": {
                "description": "用户关于短视频内容或所需氛围的描述",
                "type": "string",
            }
        },
        "required": ["video_description"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        user_desc = args["video_description"]
        
        if not dashscope.api_key or not str(dashscope.api_key).strip():
            try:
                from config_loader import get_dashscope_api_key
                dashscope.api_key = get_dashscope_api_key()
            except (ImportError, FileNotFoundError, ValueError, KeyError):
                dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a music director for short videos."
                    "Analyze the user's video description and output a JSON object with:\n"
                    "1.'title': A short, relevant title (e.g., 'Cyberpunk City').\n"
                    "2.'style': English music style tags, focusing on atmosphere and instruments (e.g., 'Upbeat, Lofi Hip Hop, Piano, 90bpm').\n"
                    "Output JSON ONLY."
                ),
            },
            {
                "role": "user",
                "content": f"Plan background music for: {user_desc}",
            },
        ]

        try:
            response = dashscope.Generation.call(
                "qwen-max",
                messages=messages,
                result_format="message",
                stream=False,
            )
            content = response.output.choices[0].message.content
            

            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1:
                return content[start:end]
            return json.dumps({"title": "BGM", "style": "Pop, Instrumental"})

        except Exception as e:
            return json.dumps({"title": "Background Music", "style": "Relaxing"})


@register_tool("my_music_gen")
class MyMusicGen(BaseTool):
    description = "调用suno API生成适用于短视频的背景音乐（纯音乐）。"
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "description": "音乐标题",
                "type": "string",
            },
            "style": {
                "description": "音乐风格描述 (English tags)",
                "type": "string",
            }
        },
        "required": ["style"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        title = args.get("title", "Unknown Track")
        style = args.get("style", "Cinematic")
        
        # 调试
        print(f"标题: {title}")
        print(f"风格: {style}")

        headers = {
            'Authorization': f'Bearer {SUNO_API_KEY}',
            'Content-Type': 'application/json'
        }

        if not SUNO_API_KEY:
            return json.dumps({"status": "error", "message": "SUNO_API_KEY missing. Please set environment variable SUNO_API_KEY."})

        payload = {
            "customMode": True,
            "instrumental": True,  # 纯音乐
            "prompt": "Generate a BGM for short videos.",  # 暂固定为对短视频生成        
            "style": style,
            "title": title,
            "model": "V4_5ALL",       
            "callBackUrl": "https://example.com"
        }

        try:
            generate_url = f"{SUNO_BASE_URL}/generate"
            resp = requests.post(generate_url, headers=headers, json=payload, timeout=30)
            
            result = resp.json()
            if result.get('code') != 200:
                return json.dumps({"status": "error", "message": f"API提交失败: {result.get('msg')}"})
            
            task_id = result['data']['taskId']
            print(f"任务提交成功，ID: {task_id}")
            


            max_wait_time = 300        # 最长等待时间
            start_time = time.time()
            
            audio_url = ""
            image_url = ""
            
            while time.time() - start_time < max_wait_time:
                time.sleep(10)
                
                query_url = f"{SUNO_BASE_URL}/generate/record-info?taskId={task_id}"
                q_resp = requests.get(query_url, headers=headers)
                
                if q_resp.status_code != 200:
                    continue
                
                q_data = q_resp.json()
                if 'data' not in q_data:
                    continue
                
                status_info = q_data['data']
                status_code = status_info.get('status')
                
                print(f"当前状态: {status_code}")
                
                if status_code == 'SUCCESS':
                    response_data = status_info.get('response', {})
                    tracks = response_data.get('sunoData', [])
                    
                    if tracks and len(tracks) > 0:
                        track = tracks[0]
                        audio_url = track.get('audioUrl')
                        image_url = track.get('imageUrl')
                        title = track.get('title')
                        break
                elif status_code == 'FAILED':
                    error_msg = status_info.get('errorMessage', '未知错误')
                    return json.dumps({"status": "error", "message": f"生成失败: {error_msg}"})

            if not audio_url:
                return json.dumps({"status": "timeout", "message": "生成超时，请稍后检查。"})


            display_text = (
                f"**{title} (纯音乐)**\n"
                f"风格: {style}\n"
                f"![封面]({image_url})\n"
                f"<audio controls src='{audio_url}'></audio>\n"
                f"[下载音频]({audio_url})"
            )

            return json.dumps({
                "status": "success", 
                "audio_url": audio_url,
                "display_info": display_text
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"status": "error", "message": f"执行异常: {str(e)}"})


def init_agent():
    llm_cfg = {
        "model": "qwen-max",
        "model_server": "dashscope",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    }

    system_instruction = (
        "你是一个专业的短视频配乐助手。"
        "你的目标是生成用于短视频背景配乐的纯音乐（Instrumental/BGM）。"
        "1. 理解用户描述的画面感和情绪。"
        "2. 调用 `music_prompt_refiner` 将描述转化为音乐风格参数。"
        "3. 向用户展示规划的风格。" # 如果需要用户确认，在这里修改"并向用户进行确认，如果用户确认，则继续"
        "4. 调用 `my_music_gen` 生成纯音乐。"
        "5. 提示用户该音乐适合作为背景音乐使用。"
    )

    bot = Assistant(
        llm=llm_cfg,
        name="short_video_BGM_agent",
        system_message=system_instruction,
        function_list=["music_prompt_refiner", "my_music_gen"],
    )
    return bot

if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("dashscope apikey missed")
    if not SUNO_API_KEY:
        print("suno apikey missed (set env SUNO_API_KEY)")

    chatbot_config = {
        "prompt.suggestions": [
            "生成一首适合雨天听的悲伤钢琴曲",
            "来一段赛博朋克风格的快节奏电子乐",
            "我想剪辑一个赛博朋克风格的雨夜视频，需要背景音乐",
        ]
    }
        
    print(f"BGM 生成智能体启动中... 目标 API: {SUNO_BASE_URL}")
    WebUI(init_agent(), chatbot_config=chatbot_config).run()
