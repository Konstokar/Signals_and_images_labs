from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


def process_video_background(
    input_path: str,
    output_path: str,
    history: int = 500,
    var_threshold: float = 16.0,
    detect_shadows: bool = True,
    warmup_frames: int = 30,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    if not 1 <= history <= 10000:
        raise ValueError(
            "История MOG2 должна быть от 1 до 10000 кадров."
        )

    if not 1.0 <= var_threshold <= 100.0:
        raise ValueError(
            "Порог MOG2 должен быть в диапазоне 1–100."
        )

    if not 0 <= warmup_frames <= 1000:
        raise ValueError(
            "Число кадров обучения должно быть от 0 до 1000."
        )

    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError(
            "Выходной файл должен отличаться от исходного видео."
        )

    source = cv2.VideoCapture(str(input_path))

    if not source.isOpened():
        source.release()
        raise ValueError(
            f"Не удалось открыть видео: {input_path}"
        )

    fps = source.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 25.0

    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        source.release()
        raise ValueError(
            "Не удалось определить размеры видео."
        )

    output = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )

    if not output.isOpened():
        source.release()
        output.release()
        raise ValueError(
            "Не удалось создать выходной файл. "
            "Выберите путь с расширением .mp4."
        )

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    processed = 0

    try:
        while True:
            success, frame = source.read()

            if not success:
                break

            mask = subtractor.apply(frame)

            if processed < warmup_frames:
                mask = np.zeros_like(mask)

            mask_bgr = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2BGR,
            )

            output.write(mask_bgr)
            processed += 1

            if progress_callback is not None:
                progress_callback(processed, total_frames)

    finally:
        source.release()
        output.release()

    if processed == 0:
        try:
            Path(output_path).unlink(missing_ok=True)
        except OSError:
            pass

        raise ValueError(
            "В выбранном видео не удалось прочитать ни одного кадра."
        )

    return {
        "frames_processed": processed,
        "fps": fps,
        "width": width,
        "height": height,
        "output_path": str(output_path),
    }
