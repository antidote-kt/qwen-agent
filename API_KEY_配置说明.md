# API Key 配置说明

本项目使用统一的配置文件 `api_keys.json` 来管理所有 API Key。

## 配置文件位置

在项目根目录创建 `api_keys.json` 文件，格式如下：

```json
{
    "DASHSCOPE_API_KEY": "你的_DashScope_API_Key",
    "SHOTSTACK_API_KEY": "你的_Shotstack_API_Key"
}
```

## API Key 说明

### 1. DASHSCOPE_API_KEY（DashScope / 通义千问）

**用途：**
- 文本大模型（Prompt 精修、分镜规划）
- 图像生成（关键帧图片）
- 文生/图生视频（万相 2.1）
- 多模态视频理解（参考视频分析）

**获取方式：** 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/) 获取

### 2. SHOTSTACK_API_KEY（Shotstack 云剪辑）

**用途：**
- 视频剪辑和合成（云端渲染）

**获取方式：** 访问 [Shotstack 官网](https://shotstack.io/) 注册并获取

## 配置优先级

代码会按以下优先级读取 API Key：

1. **配置文件** (`api_keys.json`) - 优先使用
2. **环境变量** - 如果配置文件不存在或读取失败，会回退到环境变量

## 环境变量方式（备选）

如果不想使用配置文件，也可以设置环境变量：

**Windows PowerShell:**
```powershell
$env:DASHSCOPE_API_KEY = "你的_DashScope_API_Key"
$env:SHOTSTACK_API_KEY = "你的_Shotstack_API_Key"
```

**Linux/Mac:**
```bash
export DASHSCOPE_API_KEY="你的_DashScope_API_Key"
export SHOTSTACK_API_KEY="你的_Shotstack_API_Key"
```

## 注意事项

1. **安全性**：`api_keys.json` 已添加到 `.gitignore`，不会被提交到 Git 仓库
2. **配置文件格式**：必须是有效的 JSON 格式
3. **必需字段**：两个 API Key 都必须填写，不能为空
4. **错误提示**：如果配置文件不存在或格式错误，程序会显示明确的错误信息

## 验证配置

运行程序时，如果 API Key 未正确配置，会看到相应的错误提示。请根据提示检查配置文件或环境变量设置。

