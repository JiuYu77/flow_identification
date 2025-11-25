# Favicon 转换说明

## 当前状态
- ✅ 已创建 SVG favicon (`public/favicon.svg`)
- ✅ 已更新 `index.html` 支持 SVG favicon（现代浏览器优先使用）
- ⚠️ 需要创建 `public/favicon.ico`（用于旧浏览器兼容）

## 转换方法

### 方法 1: 使用在线工具（推荐）
1. 访问 https://convertio.co/svg-ico/ 或 https://cloudconvert.com/svg-to-ico
2. 上传 `public/favicon.svg`
3. 下载转换后的文件
4. 保存为 `public/favicon.ico`

### 方法 2: 使用 Python 脚本
```bash
# 安装依赖
pip install Pillow cairosvg

# 运行转换脚本
python3 convert_favicon.py
```

### 方法 3: 使用 ImageMagick
```bash
# 安装 ImageMagick
sudo apt install imagemagick

# 转换 SVG 到 ICO
convert -density 256x256 -background transparent public/favicon.svg \
  -define icon:auto-resize -colors 256 public/favicon.ico
```

## 图标设计说明
- **主题**: 流型识别系统
- **元素**: 
  - 蓝色渐变背景（主色调 #409eff）
  - 流动的波浪线（代表流体）
  - 数据点（代表识别）
  - 箭头（代表流动方向）
- **尺寸**: 64x64（可缩放）

