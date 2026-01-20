from qwen_agent import QwenAgent

# 初始化 Agent
agent = QwenAgent(model="qwen-max")

# 简单对话
response = agent.chat("用三句话介绍一下量子计算")
print(response)
