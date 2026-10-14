import colorsys

import cv2
import numpy as np


SUPPORTED_MODELS = ("RGB", "CMYK", "HSL", "LAB", "HSV", "YCbCr")


def _validate_components(values, model):
    if len(values) != 3 and model != "CMYK":
        raise ValueError(f"Модель {model} должна содержать 3 компонента.")

    if model == "CMYK" and len(values) != 4:
        raise ValueError("Модель CMYK должна содержать 4 компонента.")

    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("Компоненты цвета должны быть конечными числами.")


def _to_rgb(values, model):
    _validate_components(values, model)
    values = [float(v) for v in values]

    if model == "RGB":
        if any(v < 0 or v > 255 for v in values):
            raise ValueError("Компоненты RGB должны быть в диапазоне 0–255.")
        return np.array(values, dtype=np.uint8)

    if model == "CMYK":
        c, m, y, k = values
        if any(v < 0 or v > 100 for v in values):
            raise ValueError("Компоненты CMYK должны быть в диапазоне 0–100.")

        c, m, y, k = [v / 100 for v in (c, m, y, k)]
        rgb = [
            255 * (1 - c) * (1 - k),
            255 * (1 - m) * (1 - k),
            255 * (1 - y) * (1 - k),
        ]
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    if model == "HSL":
        h, s, lightness = values
        if not 0 <= h <= 360 or not 0 <= s <= 100 or not 0 <= lightness <= 100:
            raise ValueError("HSL: H должен быть 0–360, S и L — 0–100.")

        rgb = colorsys.hls_to_rgb(
            (h % 360) / 360,
            lightness / 100,
            s / 100,
        )
        return np.clip(np.rint(np.array(rgb) * 255), 0, 255).astype(np.uint8)

    if model == "HSV":
        h, s, v = values
        if not 0 <= h <= 360 or not 0 <= s <= 100 or not 0 <= v <= 100:
            raise ValueError("HSV: H должен быть 0–360, S и V — 0–100.")

        hsv = np.array(
            [[[h % 360 / 2, s * 2.55, v * 2.55]]],
            dtype=np.uint8,
        )
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]

    if model == "LAB":
        l, a, b = values
        if not 0 <= l <= 100 or not -128 <= a <= 127 or not -128 <= b <= 127:
            raise ValueError("LAB: L — 0–100, a и b — примерно от -128 до 127.")

        lab = np.array(
            [[[l * 255 / 100, a + 128, b + 128]]],
            dtype=np.uint8,
        )
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]

    if model == "YCbCr":
        y, cb, cr = values
        if any(v < 0 or v > 255 for v in values):
            raise ValueError("Компоненты YCbCr должны быть в диапазоне 0–255.")

        ycbcr = np.array([[[y, cb, cr]]], dtype=np.uint8)
        return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)[0, 0]

    raise ValueError(f"Неизвестная цветовая модель: {model}")


def _from_rgb(rgb, model):
    r, g, b = [int(v) for v in rgb]

    if model == "RGB":
        return (r, g, b)

    if model == "CMYK":
        rf, gf, bf = r / 255, g / 255, b / 255
        k = 1 - max(rf, gf, bf)

        if k >= 1:
            return (0.0, 0.0, 0.0, 100.0)

        c = (1 - rf - k) / (1 - k)
        m = (1 - gf - k) / (1 - k)
        y = (1 - bf - k) / (1 - k)
        return tuple(round(v * 100, 2) for v in (c, m, y, k))

    if model == "HSL":
        h, lightness, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        return (round(h * 360, 2), round(s * 100, 2), round(lightness * 100, 2))

    if model == "HSV":
        hsv = cv2.cvtColor(
            np.array([[[r, g, b]]], dtype=np.uint8),
            cv2.COLOR_RGB2HSV,
        )[0, 0]
        return (int(hsv[0]) * 2, round(float(hsv[1]) * 100 / 255, 2),
                round(float(hsv[2]) * 100 / 255, 2))

    if model == "LAB":
        lab = cv2.cvtColor(
            np.array([[[r, g, b]]], dtype=np.uint8),
            cv2.COLOR_RGB2LAB,
        )[0, 0]
        return (
            round(float(lab[0]) * 100 / 255, 2),
            int(lab[1]) - 128,
            int(lab[2]) - 128,
        )

    if model == "YCbCr":
        ycbcr = cv2.cvtColor(
            np.array([[[r, g, b]]], dtype=np.uint8),
            cv2.COLOR_RGB2YCrCb,
        )[0, 0]
        return (int(ycbcr[0]), int(ycbcr[2]), int(ycbcr[1]))

    raise ValueError(f"Неизвестная цветовая модель: {model}")


def convert_color(values, source_model, target_model):
    source_model = source_model.upper()
    target_model = target_model.upper()

    if source_model not in SUPPORTED_MODELS:
        raise ValueError(f"Неизвестная исходная модель: {source_model}")

    if target_model not in SUPPORTED_MODELS:
        raise ValueError(f"Неизвестная целевая модель: {target_model}")

    rgb = _to_rgb(values, source_model)
    return _from_rgb(rgb, target_model)
