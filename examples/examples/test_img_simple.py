#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接测试图片生成工具"""

import json
import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from t2i import BatchStoryboardPainter

def test_image_generation():
    print('=' * 60)
    print('测试图片生成工具')
    print('=' * 60)
    
    # 创建工具实例
    try:
        painter = BatchStoryboardPainter()
        print('✅ BatchStoryboardPainter 工具实例化成功')
    except Exception as e:
        print(f'❌ 工具实例化失败: {e}')
        return False
    
    # 准备测试数据
    test_shots = [{
        'shot_id': '01',
        't2i_prompt': '一只可爱的橘色小猫坐在窗台上，阳光洒在它的毛发上，温馨的室内场景'
    }]
    
    test_data = {
        'json_content': json.dumps(test_shots, ensure_ascii=False),
        'resolution': '1024*1024',
        'save_images': False,
        'max_workers': 1
    }
    
    print(f'\n📝 测试提示词: {test_shots[0]["t2i_prompt"]}')
    print(f'🚀 开始生成图片...\n')
    
    # 调用工具
    try:
        result = painter.call(json.dumps(test_data, ensure_ascii=False))
        result_data = json.loads(result)
        
        print('=' * 60)
        print('生成结果:')
        print('=' * 60)
        
        if result_data.get('status') == 'success':
            print('✅ 生成成功!')
            results = result_data.get('results', [])
            if results:
                for r in results:
                    shot_id = r.get('shot_id', '未知')
                    image_url = r.get('image_url', '')
                    print(f'\n镜头 {shot_id}:')
                    if image_url:
                        print(f'  图片URL: {image_url}')
                        return True
                    else:
                        print(f'  ⚠️ 未生成图片URL')
                        return False
            else:
                print('⚠️ 未返回图片结果')
                return False
        else:
            print(f'❌ 生成失败: {result_data.get("message", "未知错误")}')
            print(f'\n完整响应:')
            print(json.dumps(result_data, ensure_ascii=False, indent=2))
            return False
            
    except Exception as e:
        print(f'❌ 调用过程中出错: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_image_generation()
    print('\n' + '=' * 60)
    if success:
        print('✅ 图片生成功能正常')
    else:
        print('❌ 图片生成功能异常')
    print('=' * 60)
