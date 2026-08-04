---
name: image-grayscale
description: 将图片转换为黑白（灰度）版本。当用户要求"黑白化"、"去色"、"转灰度"、"变成黑白照片"等处理某张图片时使用。
---

# Image Grayscale（图片黑白化）

将彩色图片转换为黑白（灰度）图像，保留原始分辨率，仅去除色彩信息。

## 依赖

```bash
pip install Pillow --break-system-packages
```

## 基本用法

```python
from PIL import Image

def to_grayscale(input_path: str, output_path: str, quality: int = 95) -> None:
    img = Image.open(input_path)
    gray = img.convert("L")
    gray.save(output_path, quality=quality)

to_grayscale("input.jpg", "output_bw.jpg")
```

## 何时需要查看 references

- 想了解灰度转换背后的原理（加权公式、和 OpenCV 的区别）→ 查看 `references/grayscale_methods.md`
- 需要处理带透明通道的 PNG，或大批量处理一个目录下的图片 → 查看 `references/batch_and_alpha.md`

日常单张 JPG/PNG 转黑白，用上面的基本用法就够了，不需要额外加载 references。
