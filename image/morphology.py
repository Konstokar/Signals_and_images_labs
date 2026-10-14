import cv2
import numpy as np
from PIL import Image


SUPPORTED_OPERATIONS = (
    "Эрозия",
    "Дилатация",
    "Открытие",
    "Закрытие",
    "Градиент",
)


def apply_morphology(
    image: Image.Image,
    operation: str,
    kernel_size: int = 3,
    iterations: int = 1,
) -> Image.Image:

    if not isinstance(image, Image.Image):
        raise TypeError("Ожидается объект PIL.Image.Image.")

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Неизвестная операция: {operation}. "
            f"Доступные операции: {', '.join(SUPPORTED_OPERATIONS)}."
        )

    if not isinstance(kernel_size, int) or not 1 <= kernel_size <= 31:
        raise ValueError("Размер ядра должен быть целым числом от 1 до 31.")

    if not isinstance(iterations, int) or not 1 <= iterations <= 10:
        raise ValueError("Количество повторений должно быть от 1 до 10.")

    gray = np.asarray(image.convert("L"), dtype=np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )

    if operation == "Эрозия":
        result = cv2.erode(
            gray,
            kernel,
            iterations=iterations,
        )

    elif operation == "Дилатация":
        result = cv2.dilate(
            gray,
            kernel,
            iterations=iterations,
        )

    elif operation == "Открытие":
        result = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iterations,
        )

    elif operation == "Закрытие":
        result = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations,
        )

    else:  # Градиент
        result = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            kernel,
            iterations=iterations,
        )

    return Image.fromarray(result, mode="L")
