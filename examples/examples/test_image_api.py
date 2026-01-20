"""测试图片生成 API 接口

使用 api_server.py 中的 /paint_storyboard 接口测试图片生成功能
"""

import json
import requests

def test_paint_storyboard_api():
    """测试分镜绘制接口"""
    
    print("=" * 60)
    print("测试 /paint_storyboard 接口")
    print("=" * 60)
    
    # API 端点
    api_url = "http://localhost:8000/paint_storyboard"
    
    # 准备测试数据
    test_data = {
        "json_content": json.dumps([
            {
                "shot_id": "01",
                "t2i_prompt": "一只可爱的橘色小猫坐在窗台上，阳光洒在它的毛发上，温馨的室内场景，高质量摄影"
            }
        ], ensure_ascii=False),
        "resolution": "1024*1024",
        "save_images": False,
        "max_workers": 1
    }
    
    print(f"\n📝 测试数据:")
    print(f"   提示词: {json.loads(test_data['json_content'])[0]['t2i_prompt']}")
    print(f"   分辨率: {test_data['resolution']}")
    
    try:
        print(f"\n🚀 发送请求到: {api_url}")
        response = requests.post(
            api_url,
            json=test_data,
            timeout=180  # 3分钟超时
        )
        
        print(f"\n📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 60)
            print("响应结果:")
            print("=" * 60)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            if result.get("status") == "success":
                data = result.get("data", {})
                
                if data.get("status") == "success":
                    print("\n✅ 图片生成成功!")
                    
                    results = data.get("results", [])
                    if results:
                        for r in results:
                            shot_id = r.get("shot_id", "未知")
                            image_url = r.get("image_url", "")
                            
                            print(f"\n镜头 {shot_id}:")
                            if image_url:
                                print(f"  ✅ 图片URL: {image_url}")
                            else:
                                print(f"  ⚠️ 未生成图片URL")
                    else:
                        print("⚠️ 未返回图片结果")
                    
                    return True
                else:
                    print(f"\n❌ 生成失败: {data.get('message', '未知错误')}")
                    return False
            else:
                print(f"\n❌ API 返回错误: {result.get('message', '未知错误')}")
                return False
        else:
            print(f"\n❌ HTTP 错误: {response.status_code}")
            print(f"响应内容: {response.text[:500]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ 连接错误: 无法连接到 API 服务器")
        print("请确保 api_server.py 正在运行:")
        print("  python api_server.py")
        return False
    except requests.exceptions.Timeout:
        print("\n❌ 请求超时")
        return False
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_server_health():
    """检查服务器是否运行"""
    
    print("\n🔍 检查 API 服务器状态...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API 服务器正在运行")
            return True
        else:
            print(f"⚠️ API 服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API 服务器未运行")
        print("\n请先启动服务器:")
        print("  cd examples")
        print("  python api_server.py")
        return False
    except Exception as e:
        print(f"❌ 检查服务器状态时出错: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "🎨" * 30)
    print("图片生成 API 测试")
    print("🎨" * 30 + "\n")
    
    # 1. 检查服务器
    if not check_server_health():
        print("\n请先启动 API 服务器再运行此测试")
        exit(1)
    
    # 2. 测试图片生成
    success = test_paint_storyboard_api()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if success:
        print("✅ 图片生成功能测试通过")
    else:
        print("❌ 图片生成功能测试失败")
    
    print("\n测试完成！\n")
