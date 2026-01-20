# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Iterator, List, Optional, Union

from qwen_agent import Agent
from qwen_agent.agents.assistant import Assistant, get_current_date_str
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import Message, SYSTEM, CONTENT
from qwen_agent.tools import BaseTool


class MyCustomAgent(Agent):
    """
    一个最小可用的自定义 Agent 模板：
    - 通过内置 Assistant 复用 RAG + 工具调用能力
    - 在进入 Assistant 之前，插入自定义的系统指令或上下文（如当前日期）

    你可以基于此类快速定制自己的工作流，例如：
    - 在 Assistant 之前做意图识别、规划或消息改写
    - 在 Assistant 之后对回复做后处理（格式化、过滤、结构化等）
    """

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        files: Optional[List[str]] = None,
        rag_cfg: Optional[Dict] = None,
    ):
        # 初始化基础 Agent（主要负责工具注册、系统指令注入等通用能力）
        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
        )
        # 复用 Assistant 的 RAG 与函数调用能力
        self.assistant = Assistant(
            function_list=function_list,
            llm=self.llm,
            system_message=system_message,
            name=name,
            description=description,
            files=files,
            rag_cfg=rag_cfg,
        )

    def _run(
        self,
        messages: List[Message],
        lang: str = "zh",
        **kwargs,
    ) -> Iterator[List[Message]]:
        # 在进入 Assistant 前，向系统消息追加当前日期，示范如何插入自定义上下文
        date_tip = get_current_date_str(lang=lang)
        if messages and messages[0].role == SYSTEM:
            if isinstance(messages[0][CONTENT], str):
                messages[0][CONTENT] += "\n\n" + date_tip
            else:
                # 列表形式的系统内容（多模态），这里仅追加文本
                messages[0][CONTENT].append({'text': "\n\n" + date_tip})
        else:
            messages = [Message(role=SYSTEM, content=date_tip)] + messages

        # 交给内置 Assistant 执行（包含工具调用与检索）
        for chunk in self.assistant.run(messages=messages, lang=lang, **kwargs):
            yield chunk
