from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


def _clean_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    min_area: int = 20,
) -> np.ndarray:

    if kernel_size < 1 or kernel_size > 31:
        raise ValueError(
            "Размер ядра должен быть в диапазоне 1–31."
        )

    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

    if min_area > 0:
        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        filtered = np.zeros_like(mask)

        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]

            if area >= min_area:
                filtered[labels == label] = 255

        mask = filtered

    return mask


def _histogram(frame: np.ndarray) -> np.ndarray:

    small = cv2.resize(
        frame,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2HSV,
    )

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        histogram,
        histogram,
    )

    return histogram


def process_video_frame_difference(
    input_path: str,
    output_path: str,
    difference_threshold: int = 25,
    warmup_frames: int = 1,
    scene_change_threshold: float = 0.45,
    morphology_kernel: int = 3,
    min_component_area: int = 20,
    overlay_output_path: Optional[str] = None,
    progress_callback: Optional[
        Callable[[int, int], None]
    ] = None,
) -> dict:

    if not 0 <= difference_threshold <= 255:
        raise ValueError(
            "Порог разности должен быть в диапазоне 0–255."
        )

    if not 0 <= warmup_frames <= 1000:
        raise ValueError(
            "Число начальных кадров должно быть от 0 до 1000."
        )

    if not 0.0 <= scene_change_threshold <= 1.0:
        raise ValueError(
            "Порог смены сцены должен быть в диапазоне 0–1."
        )

    if not 1 <= morphology_kernel <= 31:
        raise ValueError(
            "Размер ядра должен быть в диапазоне 1–31."
        )

    if min_component_area < 0:
        raise ValueError(
            "Минимальная площадь не может быть отрицательной."
        )

    source_path = Path(input_path).resolve()
    mask_path = Path(output_path).resolve()

    overlay_path = (
        Path(overlay_output_path).resolve()
        if overlay_output_path
        else None
    )

    if source_path == mask_path:
        raise ValueError(
            "Маска не должна перезаписывать исходное видео."
        )

    if overlay_path is not None and overlay_path in (
        source_path,
        mask_path,
    ):
        raise ValueError(
            "Пути исходного видео, маски и наложения "
            "должны различаться."
        )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        capture.release()
        raise ValueError(
            f"Не удалось открыть видео: {input_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 25.0

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(
            "Не удалось определить размеры видео."
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    mask_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
        True,
    )

    if not mask_writer.isOpened():
        capture.release()
        mask_writer.release()

        raise ValueError(
            "Не удалось создать видео маски. "
            "Выбери путь с расширением .mp4."
        )

    overlay_writer = None

    if overlay_output_path:
        overlay_writer = cv2.VideoWriter(
            str(overlay_output_path),
            fourcc,
            fps,
            (width, height),
            True,
        )

        if not overlay_writer.isOpened():
            capture.release()
            mask_writer.release()
            overlay_writer.release()

            raise ValueError(
                "Не удалось создать видео наложения."
            )

    previous_gray = None
    previous_histogram = None

    processed = 0
    scene_changes = 0
    foreground_pixels_total = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0,
            )

            histogram = _histogram(frame)

            mask = np.zeros(
                (height, width),
                dtype=np.uint8,
            )

            scene_changed = False

            if previous_histogram is not None:
                distance = cv2.compareHist(
                    previous_histogram,
                    histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )

                scene_changed = (
                    distance >= scene_change_threshold
                )

                if scene_changed:
                    scene_changes += 1

            if (
                previous_gray is not None
                and not scene_changed
                and processed >= warmup_frames
            ):
                difference = cv2.absdiff(
                    gray,
                    previous_gray,
                )

                _, mask = cv2.threshold(
                    difference,
                    difference_threshold,
                    255,
                    cv2.THRESH_BINARY,
                )

                mask = _clean_mask(
                    mask,
                    kernel_size=morphology_kernel,
                    min_area=min_component_area,
                )

            previous_gray = gray
            previous_histogram = histogram

            foreground_pixels_total += cv2.countNonZero(
                mask
            )

            mask_bgr = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2BGR,
            )

            mask_writer.write(mask_bgr)

            if overlay_writer is not None:
                highlighted = frame.copy()
                selected = mask > 0

                if np.any(selected):
                    original_pixels = frame[
                        selected
                    ].astype(np.float32)

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

                overlay_writer.write(highlighted)

            processed += 1

            if progress_callback is not None:
                progress_callback(
                    processed,
                    total_frames,
                )

    finally:
        capture.release()
        mask_writer.release()

        if overlay_writer is not None:
            overlay_writer.release()

    if processed == 0:
        for path in (output_path, overlay_output_path):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

        raise ValueError(
            "В выбранном видео не удалось прочитать кадры."
        )

    total_pixels = processed * width * height

    foreground_percent = (
        foreground_pixels_total
        / total_pixels
        * 100
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
            foreground_percent,
            2,
        ),
    }
