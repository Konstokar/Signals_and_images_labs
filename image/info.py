from pathlib import Path

from PIL import Image


def get_image_info(file_path: str) -> dict:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        with Image.open(path) as img:
            width, height = img.size
            image_format = img.format or "Не определён"
            mode = img.mode

            color_depths = {
                "1": 1,
                "L": 8,
                "P": 8,
                "RGB": 24,
                "RGBA": 32,
                "CMYK": 32,
                "LA": 16,
                "I": 32,
                "F": 32,
                "I;16": 16,
            }

            color_depth = color_depths.get(mode)

            if color_depth is None:
                color_depth = (
                    len(img.getbands()) * 8
                )

            return {
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size,
                "file_size_kb": path.stat().st_size / 1024,
                "width": width,
                "height": height,
                "resolution": f"{width} × {height}",
                "format": image_format,
                "mode": mode,
                "color_depth": color_depth,
                "color_depth_text": f"{color_depth} бит/пиксель",
                "channels": len(img.getbands()),
                "has_transparency": (
                    "A" in img.getbands()
                    or "transparency" in img.info
                ),
            }

    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Не удалось прочитать изображение: {exc}"
        ) from exc
