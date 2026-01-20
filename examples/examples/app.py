import streamlit as st
import os
from video_generator import sample_async_call

st.set_page_config(page_title="万相视频生成", layout="centered")
st.title("🎬 阿里万相视频生成")

# 输入区域
prompt = st.text_area(
    "视频描述词",
    placeholder="例如：一只橘猫在阳光下追蝴蝶，电影感，温暖色调...",
    height=100
)

uploaded_file = st.file_uploader("上传参考图片 (可选，用于图生视频)", type=["png", "jpg", "jpeg"])

# 处理图片路径
img_path_for_backend = None
if uploaded_file is not None:
    # 保存上传的文件到临时位置
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_path_for_backend = f"file://{os.path.abspath(temp_path)}"
    st.image(uploaded_file, caption="已上传图片", use_column_width=True)

# 生成按钮
# 生成按钮
if st.button("🚀 开始生成视频", type="primary", use_container_width=True):
    if not prompt:
        st.error("请输入视频描述词！")
    else:
        with st.spinner("正在提交并生成视频，这可能需要几分钟，请耐心等待..."):
            # 调用重构后的后端函数，它会返回一个结果字典
            try:
                result = sample_async_call(prompt, img_path_for_backend)
                
                # 根据返回的状态，在前端显示不同的信息
                if result['status'] == 'success':
                    st.success(result['message'])  # 显示成功消息
                    if result['video_url']:
                        # 重点：使用 st.video 将视频嵌入到网页中
                        st.video(result['video_url'])
                        # 同时提供一个可点击的链接备用
                        st.markdown(f"**视频直链:** [{result['video_url']}]({result['video_url']})")
                        
                elif result['status'] == 'error':
                    st.error(f"生成失败: {result['message']}")
                elif result['status'] == 'timeout':
                    st.warning(result['message'])
                    
            except Exception as e:
                st.error(f"调用过程发生异常: {e}")