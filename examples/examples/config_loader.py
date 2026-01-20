"""
API Key 配置加载模块
从 api_keys.json 文件读取 API Key 配置
"""
import json
import os
from pathlib import Path


def load_api_keys():
    """
    从 api_keys.json 文件加载 API Keys
    
    Returns:
        dict: 包含 API Keys 的字典，格式为 {"DASHSCOPE_API_KEY": "...", "SHOTSTACK_API_KEY": "..."}
    """
    # 获取项目根目录（向上查找，直到找到 api_keys.json）
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent  # examples/examples -> examples -> root
    
    config_path = root_dir / "api_keys.json"
    
    if not config_path.exists():
        # 如果根目录没有，尝试在当前目录查找
        config_path = current_dir.parent.parent / "api_keys.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"未找到 api_keys.json 配置文件。\n"
                f"请创建 {root_dir / 'api_keys.json'} 文件，格式如下：\n"
                f'{{\n'
                f'    "DASHSCOPE_API_KEY": "你的_DashScope_API_Key",\n'
                f'    "SHOTSTACK_API_KEY": "你的_Shotstack_API_Key"\n'
                f'}}'
            )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证必需的 key（只验证核心必需的，其他为可选）
        required_keys = ["DASHSCOPE_API_KEY", "SHOTSTACK_API_KEY"]
        missing_keys = [key for key in required_keys if not config.get(key) or config.get(key).strip() == ""]
        
        if missing_keys:
            raise ValueError(
                f"api_keys.json 中缺少或为空的值: {', '.join(missing_keys)}\n"
                f"请检查配置文件: {config_path}"
            )
        
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"api_keys.json 格式错误: {e}\n请检查文件: {config_path}")


# 全局缓存
_api_keys_cache = None


def get_api_key(key_name: str, required: bool = True) -> str:
    """
    获取指定的 API Key
    
    Args:
        key_name: API Key 名称，如 "DASHSCOPE_API_KEY" 或 "SHOTSTACK_API_KEY"
        required: 是否为必需的 key，如果为 False 且未找到则返回空字符串
    
    Returns:
        str: API Key 值，如果 required=False 且未找到则返回空字符串
    """
    global _api_keys_cache
    
    if _api_keys_cache is None:
        _api_keys_cache = load_api_keys()
    
    if key_name not in _api_keys_cache:
        if required:
            raise KeyError(f"配置文件中未找到 {key_name}")
        else:
            return ""
    
    value = _api_keys_cache.get(key_name, "").strip()
    if not value and required:
        raise ValueError(f"配置文件中 {key_name} 的值为空")
    
    return value


# 便捷函数
def get_dashscope_api_key() -> str:
    """获取 DashScope API Key"""
    return get_api_key("DASHSCOPE_API_KEY")


def get_shotstack_api_key() -> str:
    """获取 Shotstack API Key"""
    return get_api_key("SHOTSTACK_API_KEY")


def get_apimart_api_key() -> str:
    """获取 APIMart API Key（可选）"""
    try:
        return get_api_key("APIMART_API_KEY", required=False)
    except (FileNotFoundError, ValueError):
        return ""


def get_imgbb_api_key() -> str:
    """获取 ImgBB API Key（可选）"""
    try:
        return get_api_key("IMGBB_API_KEY", required=False)
    except (FileNotFoundError, ValueError):
        return ""

