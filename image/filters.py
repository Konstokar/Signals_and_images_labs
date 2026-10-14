import cv2
import numpy as np
from PIL import Image


SUPPORTED_FILTERS = (
    "Повышение резкости",
    "Размытие в движении",
    "Тиснение",
    "Медианная фильтрация",
)


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Ожидается объект PIL.Image.Image.")

    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    if not 0 <= strength <= 5:
        raise ValueError("Сила резкости должна быть от 0 до 5.")

    rgb = _to_rgb_array(image)

    if strength == 0:
        return image.copy()

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 4 * strength + 1, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    result = cv2.filter2D(rgb, -1, kernel)
    return Image.fromarray(result)


def motion_blur(
    image: Image.Image,
    kernel_size: int = 15,
    direction: str = "Горизонтально",
) -> Image.Image:

    if not isinstance(kernel_size, int) or not 3 <= kernel_size <= 51:
        raise ValueError("Размер ядра размытия должен быть от 3 до 51.")

    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным.")

    directions = {
        "Горизонтально": (0, 1),
        "Вертикально": (1, 0),
        "Диагонально": (2, 2),
    }

    if direction not in directions:
        raise ValueError("Неизвестное направление размытия.")

    rgb = _to_rgb_array(image)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    if direction == "Горизонтально":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "Вертикально":
        kernel[:, kernel_size // 2] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)

    kernel /= kernel.sum()

    result = cv2.filter2D(
        rgb,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    return Image.fromarray(result)


def emboss(image: Image.Image, strength: float = 1.0) -> Image.Image:

    if not 0.1 <= strength <= 5:
        raise ValueError("Сила тиснения должна быть от 0.1 до 5.")

    rgb = _to_rgb_array(image)

    kernel = np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.float32,
    ) * strength

    filtered = cv2.filter2D(
        rgb.astype(np.float32),
        ddepth=cv2.CV_32F,
        kernel=kernel,
    )

    result = np.clip(filtered + 128, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def median_filter(
    image: Image.Image,
    kernel_size: int = 3,
) -> Image.Image:

    if not isinstance(kernel_size, int) or not 3 <= kernel_size <= 31:
        raise ValueError("Размер ядра медианного фильтра должен быть от 3 до 31.")

    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным.")

    rgb = _to_rgb_array(image)
    result = cv2.medianBlur(rgb, kernel_size)

    return Image.fromarray(result)


def apply_filter(
    image: Image.Image,
    filter_name: str,
    kernel_size: int = 3,
    strength: float = 1.0,
    direction: str = "Горизонтально",
) -> Image.Image:

    if filter_name == "Повышение резкости":
        return sharpen(image, strength)

    if filter_name == "Размытие в движении":
        return motion_blur(image, kernel_size, direction)

    if filter_name == "Тиснение":
        return emboss(image, strength)

    if filter_name == "Медианная фильтрация":
        return median_filter(image, kernel_size)

    raise ValueError(f"Неизвестный фильтр: {filter_name}")
