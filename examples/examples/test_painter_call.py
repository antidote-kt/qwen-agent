import os
import json
from examples.t2i import BatchStoryboardPainter

# 1) 测试用 prompts 文件路径
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storyboard_output"))
os.makedirs(out_dir, exist_ok=True)
json_path = os.path.join(out_dir, "test_prompts_multi.json")

# 2) 用户提供的提示词（中文）
prompt_text = (
    "清晨的丛林小径，落叶与碎石铺地，阳光斜切过画面。"
    "一只棕黄色的猕猴从左侧树藤上跃下，四肢着地蹲伏片刻，黑亮的眼睛警惕地扫视四周。"
)

# 3) 构造 JSON（标注 ref=true）
prompts = [
    {
        "shot_id": "01",
        "t2i_prompt": prompt_text,
        "ref": True
    }
]

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)

print("[*] 已写入 prompts 文件:", json_path)

# 4) 本地参考图（用户提供）
ref_image = r"C:\Users\liuww\Downloads\生成人物三视图.png"
if not os.path.exists(ref_image):
    print(f"⚠️ 参考图不存在: {ref_image}")
    print("请确认文件路径或将图片放到该位置后重试。")
else:
    # 5) 调用 Painter
    params = {
        "json_path": os.path.abspath(json_path),
        "ref_image_dir": ref_image,
        "resolution": "1280*720",
        "style_modifier": ""
    }

    p = BatchStoryboardPainter()
    print("[*] 调用 painter，参数:", params)
    res = p.call(json.dumps(params, ensure_ascii=False))
    print("[*] Painter 返回:")
    print(res)
