import cv2
import numpy as np
from PIL import Image


SUPPORTED_EDGE_DETECTORS = (
    "Canny",
    "Roberts",
)


def _to_grayscale(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Ожидается объект PIL.Image.Image.")

    return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def canny_edges(
    image: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_size: int = 3,
) -> Image.Image:
    if not 0 <= low_threshold <= 255:
        raise ValueError("Нижний порог должен находиться в диапазоне 0–255.")

    if not 0 <= high_threshold <= 255:
        raise ValueError("Верхний порог должен находиться в диапазоне 0–255.")

    if low_threshold > high_threshold:
        raise ValueError("Нижний порог не может превышать верхний.")

    if not isinstance(blur_size, int) or blur_size not in (1, 3, 5, 7, 9):
        raise ValueError("Размер размытия должен быть 1, 3, 5, 7 или 9.")

    gray = _to_grayscale(image)

    if blur_size > 1:
        gray = cv2.GaussianBlur(
            gray,
            (blur_size, blur_size),
            sigmaX=0,
        )

    result = cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        L2gradient=True,
    )

    return Image.fromarray(result)


def roberts_edges(
    image: Image.Image,
    threshold: int = 40,
) -> Image.Image:
    if not 0 <= threshold <= 255:
        raise ValueError("Порог Roberts должен находиться в диапазоне 0–255.")

    gray = _to_grayscale(image).astype(np.float32)

    # Две диагональные маски оператора Roberts.
    kernel_x = np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=np.float32,
    )

    kernel_y = np.array(
        [
            [0, 1],
            [-1, 0],
        ],
        dtype=np.float32,
    )

    gradient_x = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_x,
        borderType=cv2.BORDER_REPLICATE,
    )

    gradient_y = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_y,
        borderType=cv2.BORDER_REPLICATE,
    )

    magnitude = np.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    result = np.where(
        magnitude >= threshold,
        255,
        0,
    ).astype(np.uint8)

    return Image.fromarray(result)


def detect_edges(
    image: Image.Image,
    detector: str,
    low_threshold: int = 50,
    high_threshold: int = 150,
    roberts_threshold: int = 40,
    blur_size: int = 3,
) -> Image.Image:

    if detector == "Canny":
        return canny_edges(
            image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blur_size=blur_size,
        )

    if detector == "Roberts":
        return roberts_edges(
            image,
            threshold=roberts_threshold,
        )

    raise ValueError(f"Неизвестный детектор границ: {detector}")
