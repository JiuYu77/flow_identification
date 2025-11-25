#!/usr/bin/env python3
"""
将 SVG favicon 转换为 ICO 格式
需要安装: pip install Pillow cairosvg
或使用在线工具: https://convertio.co/svg-ico/
"""

import os
import sys

def convert_svg_to_ico():
    try:
        import cairosvg
        from PIL import Image
        import io
        
        # 读取 SVG 文件
        svg_path = 'public/favicon.svg'
        ico_path = 'public/favicon.ico'
        
        if not os.path.exists(svg_path):
            print(f"错误: {svg_path} 不存在")
            return False
        
        # 将 SVG 转换为 PNG (多个尺寸)
        sizes = [16, 32, 48]
        images = []
        
        for size in sizes:
            png_data = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size)
            img = Image.open(io.BytesIO(png_data))
            images.append(img)
        
        # 保存为 ICO
        images[0].save(ico_path, format='ICO', sizes=[(img.width, img.height) for img in images])
        print(f"成功将 {svg_path} 转换为 {ico_path}")
        return True
        
    except ImportError as e:
        print("错误: 缺少必要的库")
        print("安装方法: pip install Pillow cairosvg")
        print("\n或者使用在线工具转换:")
        print("1. 打开 https://convertio.co/svg-ico/")
        print(f"2. 上传 {svg_path}")
        print(f"3. 下载并保存为 {ico_path}")
        return False
    except Exception as e:
        print(f"转换失败: {e}")
        return False

if __name__ == '__main__':
    convert_svg_to_ico()

