from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


def _make_subtractor(
    history: int,
    var_threshold: float,
    detect_shadows: bool = False,
):
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(
        frame, (64, 64), interpolation=cv2.INTER_AREA
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]
    )
    cv2.normalize(histogram, histogram)
    return histogram


def _clean_mask(
    mask: np.ndarray,
    morphology_kernel: int = 3,
    min_area: int = 0,
) -> np.ndarray:
    if morphology_kernel <= 1:
        cleaned = mask.copy()
    else:
        if morphology_kernel % 2 == 0:
            morphology_kernel += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology_kernel, morphology_kernel),
        )

        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )

        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel
        )

    if min_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        filtered = np.zeros_like(cleaned)

        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]

            if area >= min_area:
                filtered[labels == label] = 255

        cleaned = filtered

    return cleaned


def process_video_background(
    input_path: str,
    output_path: str,
    history: int = 120,
    var_threshold: float = 25.0,
    detect_shadows: bool = False,
    warmup_frames: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    learning_rate: float = 0.01,
    scene_change_threshold: float = 0.45,
    morphology_kernel: int = 3,
    min_component_area: int = 20,
    overlay_output_path: Optional[str] = None,
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
            "Кадры обучения должны быть в диапазоне 0–1000."
        )

    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError(
            "Скорость обучения должна быть в диапазоне 0–1."
        )

    if not 0.0 <= scene_change_threshold <= 1.0:
        raise ValueError(
            "Порог смены сцены должен быть в диапазоне 0–1."
        )

    if not 1 <= morphology_kernel <= 31:
        raise ValueError(
            "Размер морфологического ядра должен быть от 1 до 31."
        )

    if min_component_area < 0:
        raise ValueError(
            "Минимальная площадь компонента не может быть отрицательной."
        )

    input_resolved = Path(input_path).resolve()
    output_resolved = Path(output_path).resolve()

    overlay_resolved = (
        Path(overlay_output_path).resolve()
        if overlay_output_path
        else None
    )

    if input_resolved == output_resolved:
        raise ValueError(
            "Выходная маска не должна перезаписывать исходное видео."
        )

    if overlay_resolved and overlay_resolved in (
        input_resolved,
        output_resolved,
    ):
        raise ValueError(
            "Пути исходного видео, маски и наложения должны различаться."
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
        True,
    )

    if not output.isOpened():
        source.release()
        output.release()
        raise ValueError(
            "Не удалось создать видео маски. "
            "Используй путь с расширением .mp4."
        )

    overlay = None

    if overlay_output_path:
        overlay = cv2.VideoWriter(
            str(overlay_output_path),
            fourcc,
            fps,
            (width, height),
            True,
        )

        if not overlay.isOpened():
            source.release()
            output.release()
            overlay.release()

            raise ValueError(
                "Не удалось создать видео наложения. "
                "Проверь путь и расширение .mp4."
            )

    subtractor = _make_subtractor(
        history, var_threshold, detect_shadows
    )

    previous_histogram = None
    cooldown = 0
    warmup_remaining = warmup_frames
    scene_changes = 0
    processed = 0
    foreground_pixels_total = 0

    try:
        while True:
            success, frame = source.read()

            if not success:
                break

            histogram = _frame_histogram(frame)
            scene_changed = False

            if previous_histogram is not None and cooldown <= 0:
                distance = cv2.compareHist(
                    previous_histogram,
                    histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )

                if distance >= scene_change_threshold:
                    subtractor = _make_subtractor(
                        history, var_threshold, detect_shadows
                    )

                    warmup_remaining = warmup_frames
                    cooldown = max(5, warmup_frames)
                    scene_changed = True
                    scene_changes += 1

            previous_histogram = histogram

            if scene_changed:
                subtractor.apply(frame, learningRate=1.0)

                mask = np.zeros(
                    (height, width), dtype=np.uint8
                )

            elif warmup_remaining > 0:
                init_rate = max(
                    learning_rate,
                    1.0 / max(1, warmup_remaining),
                )

                subtractor.apply(
                    frame, learningRate=init_rate
                )

                mask = np.zeros(
                    (height, width), dtype=np.uint8
                )

                warmup_remaining -= 1

            else:
                raw_mask = subtractor.apply(
                    frame,
                    learningRate=learning_rate,
                )

                _, mask = cv2.threshold(
                    raw_mask,
                    200,
                    255,
                    cv2.THRESH_BINARY,
                )

                mask = _clean_mask(
                    mask,
                    morphology_kernel=morphology_kernel,
                    min_area=min_component_area,
                )

            if cooldown > 0:
                cooldown -= 1

            foreground_pixels_total += cv2.countNonZero(mask)

            mask_bgr = cv2.cvtColor(
                mask, cv2.COLOR_GRAY2BGR
            )
            output.write(mask_bgr)

            if overlay is not None:
                highlighted = frame.copy()
                selected = mask > 0

                if np.any(selected):
                    original_pixels = frame[selected].astype(
                        np.float32
                    )

                    red_pixels = np.zeros_like(
                        original_pixels
                    )
                    red_pixels[:, 2] = 255

                    highlighted[selected] = np.clip(
                        original_pixels * 0.45
                        + red_pixels * 0.55,
                        0,
                        255,
                    ).astype(np.uint8)

                overlay.write(highlighted)

            processed += 1

            if progress_callback is not None:
                progress_callback(processed, total_frames)

    finally:
        source.release()
        output.release()

        if overlay is not None:
            overlay.release()

    if processed == 0:
        for path in (output_path, overlay_output_path):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

        raise ValueError(
            "В выбранном видео не удалось прочитать ни одного кадра."
        )

    total_pixels = processed * width * height

    foreground_ratio = (
        foreground_pixels_total / total_pixels
        if total_pixels
        else 0.0
    )

    return {
        "frames_processed": processed,
        "fps": fps,
        "width": width,
        "height": height,
        "output_path": str(output_path),
        "overlay_output_path": (
            str(overlay_output_path)
            if overlay_output_path
            else None
        ),
        "scene_changes_detected": scene_changes,
        "average_foreground_percent": round(
            foreground_ratio * 100, 2
        ),
    }
