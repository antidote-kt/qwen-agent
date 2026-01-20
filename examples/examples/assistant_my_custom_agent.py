import os
from qwen_agent.agents import MyCustomAgent
from qwen_agent.llm.schema import Message

"""
一个可运行的示例：
- 通过 MyCustomAgent 复用 Assistant 的 RAG+工具能力
- 自定义 system_message（角色指令）和工具列表
- 从终端交互，逐步打印流式输出

运行前准备：
1) 环境变量或配置中提供 LLM 访问（如阿里云通义）。
2) 如需使用 RAG，将文件路径加入 files（本地/URL 均可）。
3) 如需工具调用，按需选择内置工具名（例如 'code_interpreter'）。
"""


def main():
    llm_cfg = {
        'model': 'qwen-max',  # 按需替换为你的模型配置
        # 'api_key': 'YOUR_API_KEY',
        # 'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    }

    tools = [
        # 'code_interpreter',
        # 'amap_weather',
    ]

    system_message = (
        '你是一位可靠的全能助理。遇到复杂问题时先思考、再规划，'
        '必要时调用工具或检索参考资料，最终给出清晰、可执行的回答。'
    )

    files = [
        # os.path.abspath('docs/README_CN.md'),  # 也可以传入 URL 作为知识库
    ]

    bot = MyCustomAgent(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_message,
        files=files,
    )

    messages = [Message(role='user', content='帮我制定一份三天苏州自由行行程，包含花费预估。')]
    for rsp in bot.run(messages=messages, lang='zh'):
        print('bot chunk:', [m.model_dump() for m in rsp])


if __name__ == '__main__':
    main()
