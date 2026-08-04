# 批量处理 & 透明通道处理

## 带透明通道的 PNG

直接 `convert("L")` 会丢弃 alpha 通道（透明度信息）。如果原图是带透明背景的 PNG，需要保留透明度：

```python
from PIL import Image

def to_grayscale_keep_alpha(input_path: str, output_path: str) -> None:
    img = Image.open(input_path)
    if img.mode in ("RGBA", "LA"):
        gray = img.convert("LA")  # 灰度 + alpha 通道
    else:
        gray = img.convert("L")
    gray.save(output_path)
```

`"LA"` 模式 = 8-bit 灰度 + 8-bit alpha，保存为 PNG 才能保留透明效果（JPEG 不支持透明通道，会自动填充白色/黑色背景）。

## 批量处理整个目录

```python
from pathlib import Path
from PIL import Image

def batch_grayscale(input_dir: str, output_dir: str, quality: int = 95) -> int:
    """
    批量将 input_dir 下所有 jpg/png 图片转为黑白，输出到 output_dir。
    返回处理成功的文件数量。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in Path(input_dir).glob("*"):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = Image.open(f)
        if img.mode in ("RGBA", "LA") and f.suffix.lower() == ".png":
            gray = img.convert("LA")
        else:
            gray = img.convert("L")
        gray.save(out / f.name, quality=quality)
        count += 1
    return count
```

## 大分辨率图片的注意事项

- 转换速度本身很快（灰度转换是逐像素的简单加权计算，不是耗时操作）
- 但保存时如果不控制质量参数，文件体积可能不降反升（比如超高分辨率的原图）
- JPEG 建议 `quality=90~95` 之间，兼顾体积和画质
- 如果追求无损，改存 PNG 并加 `optimize=True`：
  ```python
  gray.save(output_path, format="PNG", optimize=True)
  ```
