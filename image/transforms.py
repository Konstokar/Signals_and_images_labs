import cv2
import numpy as np
from PIL import Image


def _to_rgb_array(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Ожидается объект PIL.Image.Image.")

    return np.asarray(image.convert("RGB")).copy()


def _to_image(array):
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def to_grayscale(image):
    rgb = _to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray, "L")


def adjust_brightness(image, value):
    if not -255 <= value <= 255:
        raise ValueError("Яркость должна быть в диапазоне от -255 до 255.")

    rgb = _to_rgb_array(image).astype(np.int16)
    return _to_image(rgb + int(value))


def adjust_contrast(image, factor):
    if not 0 <= factor <= 4:
        raise ValueError("Коэффициент контраста должен быть от 0 до 4.")

    rgb = _to_rgb_array(image).astype(np.float32)
    result = (rgb - 127.5) * factor + 127.5
    return _to_image(result)


def adjust_saturation(image, factor):
    if not 0 <= factor <= 4:
        raise ValueError("Коэффициент насыщенности должен быть от 0 до 4.")

    rgb = _to_rgb_array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

    result = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB,
    )
    return _to_image(result)


def linear_correction(image, black_point=0, white_point=255):
    if not 0 <= black_point < white_point <= 255:
        raise ValueError(
            "Требуется 0 <= black_point < white_point <= 255."
        )

    gray = np.asarray(image.convert("L"), dtype=np.float32)

    result = (
        (gray - black_point)
        * 255.0
        / (white_point - black_point)
    )

    return Image.fromarray(
        np.clip(result, 0, 255).astype(np.uint8),
        "L",
    )


def nonlinear_correction(image, gamma=1.0):
    if not 0.1 <= gamma <= 5:
        raise ValueError("Gamma должна быть в диапазоне 0.1–5.")

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    result = np.power(gray, gamma) * 255.0

    return Image.fromarray(
        np.clip(result, 0, 255).astype(np.uint8),
        "L",
    )
