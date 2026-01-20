#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台视频下载工具
支持 Mac、Linux、Windows 系统
用于教学研究目的
"""

import os
import sys
import subprocess
import platform
import time
import requests
from urllib.parse import urlparse, unquote
import re
import shutil
import glob

def check_yt_dlp():
    """检查 yt-dlp 是否已安装"""
    try:
        subprocess.run(['yt-dlp', '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_yt_dlp():
    """安装 yt-dlp"""
    print("正在安装 yt-dlp...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 
                      'install', '-U', 'yt-dlp'], check=True)
        print("✓ yt-dlp 安装成功！")
        return True
    except subprocess.CalledProcessError:
        print("✗ 安装失败，请手动安装：")
        print("  pip install -U yt-dlp")
        print("  或访问：https://github.com/yt-dlp/yt-dlp")
        return False



def check_ffmpeg():
    """检查系统中是否存在 ffmpeg 可执行文件"""
    return find_ffmpeg_executable() is not None


def find_ffmpeg_executable():
    """返回可用的 ffmpeg 可执行文件路径，找不到则返回 None。

    搜索顺序：系统 PATH -> 项目虚拟环境 `.venv/Scripts/ffmpeg.exe` -> tools 目录下的 ffmpeg 可执行文件。
    """
    # 1. 系统 PATH
    path = shutil.which('ffmpeg')
    if path:
        return path

    # 2. project's .venv/Scripts
    project_root = os.path.abspath(os.path.dirname(__file__))
    venv_ff = os.path.join(project_root, '.venv', 'Scripts', 'ffmpeg.exe')
    if os.path.exists(venv_ff):
        return venv_ff

    # 3. search tools/**/ffmpeg.exe
    tools_dir = os.path.join(project_root, 'tools')
    pattern = os.path.join(tools_dir, '**', 'ffmpeg.exe')
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    return None


def merge_audio_video_ffmpeg(video_path, audio_path, output_path, ffmpeg_path=None):
    """使用 ffmpeg 将视频（通常无声）和音频合并为一个输出文件

    如果未提供 `ffmpeg_path`，函数会尝试自动查找可执行文件。
    """
    if not ffmpeg_path:
        ffmpeg_path = find_ffmpeg_executable()
    if not ffmpeg_path:
        return False

    cmd = [ffmpeg_path, '-y', '-i', video_path, '-i', audio_path, '-c', 'copy', output_path]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def find_av_pairs(output_dir):
    """寻找形如 name.f12345.mp4 与 name.f67890.m4a 的音视频对，并按 name 分组返回字典"""
    pattern = re.compile(r'(?P<name>.+)\.f\d+\.(?P<ext>mp4|m4a)$')
    pairs = {}
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            name = m.group('name')
            ext = m.group('ext')
            pairs.setdefault(name, {})[ext] = os.path.join(output_dir, fname)
    return pairs


def normalize_output_to_video(output_dir, final_path):
    """把指定文件移动/重命名为 output_dir/video.mp4（覆盖同名文件）。

    返回最终路径或原路径（失败时）。"""
    try:
        dest = os.path.join(output_dir, 'video.mp4')
        # 如果已是目标路径则直接返回
        if os.path.abspath(final_path) == os.path.abspath(dest):
            return dest
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(final_path, dest)
        print(f"已重命名为统一文件名: {dest}")
        return dest
    except Exception as e:
        print(f"⚠️ 无法将 {final_path} 重命名为 video.mp4: {e}")
        return final_path


def find_downloaded_mp4(output_dir):
    """在输出目录寻找最终的 mp4 文件（忽略拆分文件 name.f123.mp4）。优先返回 *.merged.mp4，其次最近修改的 .mp4。"""
    mp4s = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith('.mp4')]
    if not mp4s:
        return None
    # 排除像 name.f123.mp4 的临时拆分文件
    temp_pattern = re.compile(r'.+\.f\d+\.mp4$', re.IGNORECASE)
    merged = [p for p in mp4s if p.lower().endswith('.merged.mp4')]
    if merged:
        # 取第一个
        return merged[0]
    candidates = [p for p in mp4s if not temp_pattern.match(os.path.basename(p))]
    if not candidates:
        return None
    # 按修改时间选最新的
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def post_process_after_yt_dlp(output_dir, url, delete_intermediate=False):
    """在 yt-dlp 成功后进行后处理：针对 B 站（bilibili.com）合并拆分的音视频文件

    参数:
        delete_intermediate (bool): 若合并成功则删除原始的拆分中间文件（*.f*.mp4 和 *.f*.m4a）。
    """
    if 'bilibili.com' not in url:
        return

    pairs = find_av_pairs(output_dir)
    if not pairs:
        return

    ffmpeg_path = find_ffmpeg_executable()
    if not ffmpeg_path:
        print('\n⚠️ 检测到拆分的音视频文件，但系统中未发现 ffmpeg。请安装 ffmpeg 或手动合并。')
        return

    for name, parts in pairs.items():
        if 'mp4' in parts and 'm4a' in parts:
            outpath = os.path.join(output_dir, f"{name}.merged.mp4")
            if os.path.exists(outpath):
                print(f"合并文件已存在，跳过: {outpath}")
                continue
            print(f"合并 {parts['mp4']} + {parts['m4a']} -> {outpath}")
            ok = merge_audio_video_ffmpeg(parts['mp4'], parts['m4a'], outpath, ffmpeg_path=ffmpeg_path)
            if ok:
                print(f"✓ 合并完成: {outpath}")
                # 统一命名为 video.mp4
                try:
                    normalize_output_to_video(output_dir, outpath)
                except Exception:
                    pass
                if delete_intermediate:
                    # 删除中间文件
                    try:
                        if os.path.exists(parts['mp4']):
                            os.remove(parts['mp4'])
                            print(f"已删除中间文件: {parts['mp4']}")
                    except Exception as e:
                        print(f"删除中间文件失败: {parts['mp4']}: {e}")
                    try:
                        if os.path.exists(parts['m4a']):
                            os.remove(parts['m4a'])
                            print(f"已删除中间文件: {parts['m4a']}")
                    except Exception as e:
                        print(f"删除中间文件失败: {parts['m4a']}: {e}")
            else:
                print(f"✗ 合并失败: {name}")


def extract_and_download_douyin(url, output_dir='downloads'):
    """使用 Playwright 渲染 Douyin 页面，提取视频直链并下载到本地。

    若系统中没有安装 Playwright，会打印提示并返回 False。
    返回 True 表示下载成功，False 表示失败或未能提取到直链。
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("✗ Playwright 未安装或不可用。请安装 Playwright 并运行 `python -m playwright install chromium` 后重试。")
        return False

    os.makedirs(output_dir, exist_ok=True)
    candidates = []
    mobile_ua = 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'
    headers = {'User-Agent': mobile_ua, 'Referer': url}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=mobile_ua, viewport={'width':375,'height':667}, locale='zh-CN')
        page = context.new_page()

        def handle_response(response):
            try:
                rurl = response.url
                ct = response.headers.get('content-type', '')
                if 'video' in ct or rurl.endswith('.mp4') or '.m3u8' in rurl or 'playwm' in rurl or 'video' in rurl:
                    candidates.append(rurl)
            except Exception:
                pass

        page.on('response', handle_response)

        try:
            page.goto(url, wait_until='networkidle', timeout=60000)
        except Exception:
            try:
                page.goto(url, wait_until='load', timeout=90000)
            except Exception as e:
                print('页面加载失败（但尝试继续提取）:', e)

        # 等待额外的网络请求
        time.sleep(3)

        try:
            video_src = page.eval_on_selector('video', 'el => el.src')
            if video_src:
                candidates.insert(0, video_src)
        except Exception:
            pass

        # 尝试 meta
        try:
            og = page.locator('meta[property="og:video"]').get_attribute('content')
            if og:
                candidates.append(og)
        except Exception:
            pass

        browser.close()

    # 去重并选择一个候选
    candidates = [c for i,c in enumerate(candidates) if c and c not in candidates[:i]]
    if not candidates:
        print('✗ 未能从页面提取到视频直链')
        return False

    chosen = None
    for c in candidates:
        if '.mp4' in c:
            chosen = c
            break
    if not chosen:
        chosen = candidates[0]

    print('找到视频直链，正在下载：', chosen)

    try:
        with requests.get(chosen, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            fn = None
            cd = r.headers.get('content-disposition')
            if cd and 'filename=' in cd:
                fn = cd.split('filename=')[-1].strip('"')
            if not fn:
                path = urlparse(unquote(chosen)).path
                fn = os.path.basename(path) or 'video.mp4'
            outpath = os.path.join(output_dir, fn)
            with open(outpath, 'wb') as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk:
                        f.write(chunk)
        print('\n✓ 下载完成（Playwright 回退）！')
        # 统一命名为 video.mp4
        try:
            normalize_output_to_video(output_dir, outpath)
        except Exception:
            pass
        return True
    except Exception as e:
        print('\n✗ 通过直链下载失败：', e)
        return False


def download_video(url, output_dir="downloads", delete_intermediate=True):
    """将指定视频下载到本地（最高画质），不返回任何值

    优先使用 `python -m yt_dlp`（确保在正确的 Python 环境中），若失败且 URL 为 Douyin 域名则回退到 Playwright 的提取与下载。

    参数:
        delete_intermediate (bool): 在检测到并成功合并拆分的音视频文件后，是否删除中间文件（默认启用）。
    """
    if not url or not isinstance(url, str):
        raise ValueError("url 必须是非空字符串")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 首先尝试使用当前 Python 解释器的 yt_dlp 模块（更可靠）
    cmd = [sys.executable, '-m', 'yt_dlp', '-f', 'bestvideo+bestaudio/best', '--merge-output-format', 'mp4', '-o', f'{output_dir}/%video.%(ext)s', url]

    try:
        subprocess.run(cmd, check=True)
        print("\n✓ 下载完成（yt-dlp）！")
        # 后处理（例如：B 站可能会拆分音视频文件，尝试合并）
        try:
            post_process_after_yt_dlp(output_dir, url, delete_intermediate=delete_intermediate)
        except Exception as e:
            print('后处理失败：', e)
        # 若 yt-dlp 产出单个 mp4（非拆分），把它重命名为 video.mp4
        try:
            cand = find_downloaded_mp4(output_dir)
            if cand:
                normalize_output_to_video(output_dir, cand)
        except Exception:
            pass
        return
    except subprocess.CalledProcessError as e:
        print("\n✗ yt-dlp 下载失败")
        # 显式抛出异常，方便上层捕获并向用户反馈
        raise RuntimeError(
            f"视频下载失败：yt-dlp 运行出错（命令退出码 {e.returncode}）。"
            " 请检查是否已在当前 Python 环境中安装 yt-dlp，"
            "以及视频 URL 是否可访问。"
        ) from e

    # 对于 Douyin 域名，尝试 Playwright 回退
    if any(d in url for d in ('douyin.com', 'v.douyin.com', 'iesdouyin.com')):
        print('尝试使用 Playwright 渲染并提取 Douyin 视频...')
        ok = extract_and_download_douyin(url, output_dir=output_dir)
        if ok:
            return
        else:
            print('✗ Playwright 回退未成功')

    # 如果走到这里，说明所有下载方式均失败
    msg = '\n✗ 视频下载失败（已尝试 yt-dlp 和 Douyin 回退）。请检查：' \
          '1) 视频链接是否有效且可访问；2) 已在当前环境安装 yt-dlp；' \
          '3) 若为抖音链接，已正确安装并初始化 Playwright 和 Chromium。'
    print(msg)
    raise RuntimeError(msg)





def call_multimodal_local_video(local_path, text, fps=2, api_key=None, model='qwen3-vl-plus'):
    """使用 MultiModalConversation 对本地视频进行问答。

    - 确保视频文件名为 `video.mp4`（会在同目录下复制/重命名为 `video.mp4`，若已存在则覆盖）。
    - 构建 messages 并调用 `MultiModalConversation.call`。

    参数:
        local_path (str): 本地视频路径（会被复制/移动为同目录下的 `video.mp4`）。
        text (str): 要询问的问题文本。
        fps (int): 抽帧频率 (默认 2)。
        api_key (str|None): API Key；若为 None 则尝试从环境变量 `DASHSCOPE_API_KEY` 读取。
        model (str): 使用的模型名。

    返回:
        原始 `MultiModalConversation.call` 的返回值。
    """
    # 确认文件存在
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地文件不存在: {local_path}")

    # 将文件命名为 video.mp4（在同目录下）
    dest_dir = os.path.dirname(os.path.abspath(local_path)) or os.getcwd()
    dest_path = os.path.join(dest_dir, 'video.mp4')
    try:
        # 如果源就是目标，不做复制；否则覆盖复制
        if os.path.abspath(local_path) != os.path.abspath(dest_path):
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.copy2(local_path, dest_path)
    except Exception as e:
        raise RuntimeError(f"无法准备 video.mp4: {e}")

    video_path = f"file://{dest_path}"

    messages = [
        {'role': 'user',
         'content': [
             {'video': video_path, 'fps': fps},
             {'text': text}
         ]}
    ]

    if api_key is None:
        api_key = os.getenv('DASHSCOPE_API_KEY')

    if not api_key:
        raise RuntimeError('未提供 API Key，请设置环境变量 DASHSCOPE_API_KEY 或传入 api_key 参数')

    # 动态导入 SDK 中的 MultiModalConversation
    try:
        # 尝试常见的导入路径
        from dashscope import MultiModalConversation
    except Exception:
        raise ImportError('找不到 MultiModalConversation，确保已安装并正确导入相关 SDK')

    # 调用模型
    response = MultiModalConversation.call(api_key=api_key, model=model, messages=messages)
    # 对于方便使用，打印首条回复文本（与示例一致）
    # try:
    #     print(response.output.choices[0].message.content[0]["text"])
    # except Exception:
    #     # 忽略打印错误，返回 response 供用户进一步处理
    #     pass
    text = response.output.choices[0].message.content[0]["text"]
    return text


# ===================== 分析提示模板与封装函数 =====================
PROMPT_ANALYSIS_TEMPLATE = """注意：所有输入视频均为短视频（建议时长 15-60 秒）。

请首先用一两句话概括该短视频的核心内容。随后请以严格的 JSON 格式返回一个对象，包含以下字段：
- script_summary: 对剧情/剧本的精炼概述（1-2句）。
- detailed_script: 按分镜顺序给出具体剧本（列表），每项为字符串，格式示例：
  "1. 女孩（参考图1）在水族馆，看到男美人鱼（参考图3），男美人鱼向她打招呼（建议时长：3s，镜头：中景）"
  要求：每条必须包含：
  （1）人物与参考图索引（如“参考图1/2/3”）；
  （2）环境/场景信息（地点、背景要素、时间/光照、关键道具/氛围）与动作描述；
  （3）建议镜头/构图（如特写/中景/远景）；
  （4）建议时长（秒）；
  （5）明确的角色数量约束（例如“仅 1 个女孩/仅 1 只猴子/仅 2 只猫：Bobo 和 Toto；不要出现额外人物/动物/重复角色”），避免后续生成出现多余角色。
- detailed_script_keyframe: （新增，建议输出）按分镜顺序给出“关键帧版本”的描述（列表，每项字符串）。
  要求：聚焦单帧画面（人物外观一致性、数量约束、环境/光照、构图、关键道具），不要写大量连续动作或复杂镜头运动措辞。
- detailed_script_video: （新增，建议输出）按分镜顺序给出“视频分镜版本”的描述（列表，每项字符串）。
  要求：聚焦动态推进（起势→过程→结果）、主体运动与镜头运动（推拉摇移/跟拍/慢动作），同时必须严格遵守数量约束与环境一致性，不要添加字幕/水印/额外角色。
- key_scenes: 列表，每项为 {"start":"0:00","end":"0:05","description":"..."}（若无法精确时间可用相对位置）。
- viral_elements: 列表，列出可能成为“爆款”元素（如梗、情绪点、反转、BGM、镜头剪辑手法等）。
- viral_ip: 列表，指出现有可联动的 IP/人物/题材（并说明理由）。
- adaptation_suggestions: 列表，给出 3-5 条改编建议（包括适配短视频（15s/60s）、系列化、选角、节奏调整、封面/标题策略等）。
- recommended_title: 为不同平台给出 2-3 个备选标题（简洁、有吸引力）。
- recommended_cover_text: 为封面文案给出 2-3 个简洁备选（能抓眼球）。
- adaptation_formats: 列出适合的改编形式（例如：短视频（15s/60s）、迷你剧、话题挑战等）。
- virality_score: 0-10 的整数，评估该短视频的爆款潜力。
- notes: 任何额外说明或执行要点（例如拍摄道具、版权注意事项、BGM 建议等）。

请仅返回符合该 JSON 格式的内容（不要包含额外的自然语言解释）。如果无法严格返回 JSON，请在第一行写“ERROR_JSON_PARSE”并随后返回可供人工参考的逐条分镜/建议文本。

说明：示例分镜格式（供参考）：
1. 女孩（参考图1）在水族馆，看到男美人鱼（参考图3），男美人鱼向她打招呼（建议时长：3s，镜头：中景）
2. 女孩面对镜头,背景是城市街道（建议时长：2s，镜头：近景）
3. 女孩和美人鱼（参考图2）在服装店选人鱼服装（建议时长：4s，镜头：分镜切换）

返回内容语言请保持中文，尽量简洁、可直接用于脚本拍摄和运营落地。"""



def process_url_and_call_model(url, text=PROMPT_ANALYSIS_TEMPLATE, fps=2, api_key=None, model='qwen3-vl-plus', output_dir='downloads', delete_intermediate=True):
    """端到端流程：下载 URL -> 确保为 output_dir/video.mp4 -> 调用多模态模型并返回响应。

    如果未提供 api_key，会尝试从环境变量 `DASHSCOPE_API_KEY` 读取；若仍未提供，会在下载完成后返回并提示需提供 API Key 才能继续调用模型。
    """
    try:
        # 1) 下载视频
        download_video(url, output_dir=output_dir, delete_intermediate=delete_intermediate)

        # 2) 确保有 video.mp4
        vpath = os.path.join(output_dir, 'video.mp4')
        if not os.path.exists(vpath):
            cand = find_downloaded_mp4(output_dir)
            if cand:
                normalize_output_to_video(output_dir, cand)
            else:
                raise FileNotFoundError(
                    f"未能在 {output_dir} 找到下载后的视频文件，"
                    "可能是视频下载失败、网络异常或保存路径出错。"
                )

        # 3) 调用模型（若无 api_key 则提示并返回 None）
        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            print('⚠️ 未检测到 DASHSCOPE_API_KEY，视频已下载并命名为 video.mp4，提供 API Key 后可调用模型。')
            return None

        # 4) 调用本地封装函数
        response = call_multimodal_local_video(os.path.join(output_dir, 'video.mp4'), text, fps=fps, api_key=api_key, model=model)
        return response
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass



def main():
    url = "https://www.youtube.com/shorts/guER0B_fACU"
    resp = process_url_and_call_model(url)
    print(resp)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
        sys.exit(0)
