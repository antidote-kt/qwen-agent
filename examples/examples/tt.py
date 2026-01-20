import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from http import HTTPStatus

import requests
import dashscope
import json5
import pprint
from dashscope.aigc.image_synthesis import ImageSynthesis
from dashscope.aigc.video_synthesis import VideoSynthesis

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from apimart_key import get_apimart_api_key

# 辅助函数-调用实际的api
def CallApi(prompt : str, img_url : str) -> str: 
    
    # 指令提示词
    instruction1 = "假设你是一个提示词增强助手，请你丰富并扩写以下这段用于文字生成图片模型的提示词，以达到更好的生成效果。（请直接返回增强后的提示词）"
    # 需要输入多模态信息
    instruction2 = "假设你是一个提示词增强助手，请你根据参考图片，丰富并扩写以下这段用于文字生成图片模型的提示词，以达到更好的生成效果。（请直接返回增强后的提示词）"

    isNeedMultiMod = False if img_url == None else True
    # 暂时使用 apimart 里的 gpt-5进行增强
    payload = {
        "model" : "gpt-5",
        "input" : [
            {
                "role": "system",
                "content": 
                [
                    {
                        "type" : "input_text",
                        "text" : instruction2 if isNeedMultiMod else instruction1
                    }
                ],
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type" : "input_text",
                        "text" : prompt
                    }
                ]
            }
        ]
    }
    
    if isNeedMultiMod :
        payload["input"][1]["content"].append({
            "type" : "input_image",
            "image_url" : img_url
        })
    
    url = "https://api.apimart.ai/v1/responses"
    token = (get_apimart_api_key() or "").strip()
    if not token:
        raise RuntimeError('Missing APIMART_API_KEY. Set env APIMART_API_KEY or put it in repo-root "api key.txt".')
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    #TODO 
    if response.status_code != 200 :
        RuntimeError("TODO: Apimart接口错误!")
        return None
    
    response = response.json()
    
    print(response["output"][1]["content"][0]["text"])
    return response["output"][1]["content"][0]["text"]


@register_tool("my_PromptEnhancer")
class PromptEnhancer(BaseTool) :
    # 提示器强化，根据用户分镜剧本进行提示词丰富续写，形成文生图 + 图生视频提示词
    
    description = (
        "用于视频生成的提示词增强工具"
        "输入文字描述和参考图片，然后返回增强后的提示词"
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
            "description" : "用于生成视频的参考图片",
            "required" : False
        }
    ]
    
    
    def call(self, params: str, **kwargs) -> str :
        # 解析json参数
        args : dict = json5.loads(params)
        
        prompt = args["prompt"]
        
        if "referedImage" in args :
            response = CallApi(prompt, args["referedImage"])
        else :
            response = CallApi(prompt, None)
        
        return json5.dumps(
            {
                "enhanced_prompt" : response
            },
            ensure_ascii = False
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