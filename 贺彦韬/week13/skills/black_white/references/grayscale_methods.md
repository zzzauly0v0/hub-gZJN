# 灰度转换方法对比

## PIL 默认方式：`convert("L")`

PIL 的 `"L"` 模式（8-bit 灰度）使用的是 ITU-R BT.601 标准的亮度加权公式：

```
L = R * 299/1000 + G * 587/1000 + B * 114/1000
```

绿色权重最高，因为人眼对绿光最敏感；蓝色权重最低。这是最常见、最快速的灰度转换方式，适合绝大多数场景。

## OpenCV 方式（对比用）

```python
import cv2
img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output_bw.jpg", gray)
```

OpenCV 默认使用的公式和 PIL 基本一致（同样是 BT.601 加权），但注意 OpenCV 读入的通道顺序是 BGR 不是 RGB，如果你自己手写加权公式需要注意通道顺序，避免颜色对调。

## 什么时候选哪种

- 只是单纯要黑白效果、不追求特殊风格 → PIL `convert("L")` 足够，代码最简单
- 项目里已经在用 OpenCV 做其他图像处理（比如边缘检测、人脸识别）→ 用 OpenCV 的方式保持技术栈统一
- 想要更接近人眼感知亮度、或者做专业级黑白摄影效果 → 可以考虑用 `convert("L")` 后再手动调整对比度/伽马值，或者用 Lab 色彩空间的 L 通道单独提取（更接近专业黑白胶片的观感）：

```python
from PIL import Image
img = Image.open("input.jpg").convert("LAB")
l_channel, a, b = img.split()
l_channel.save("output_bw_lab.jpg")
```

这种方式和 `convert("L")` 效果略有差异，通常对比度更柔和一些，适合人像/风景类照片。
