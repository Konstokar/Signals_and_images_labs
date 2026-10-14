from itertools import combinations

import cv2
import numpy as np
from PIL import Image


SUPPORTED_DETECTORS = ("Harris", "SIFT", "FAST")


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Ожидается объект PIL.Image.Image.")

    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def detect_keypoints(
    image: Image.Image,
    detector: str,
    max_features: int = 500,
    threshold: float = 0.01,
) -> tuple[Image.Image, int]:
    if detector not in SUPPORTED_DETECTORS:
        raise ValueError(f"Неизвестный детектор: {detector}")

    if not 1 <= max_features <= 10000:
        raise ValueError("Количество точек должно быть от 1 до 10000.")

    if not 0.001 <= threshold <= 0.2:
        raise ValueError("Порог должен быть в диапазоне 0.001–0.2.")

    rgb = _to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if detector == "Harris":
        gray_float = np.float32(gray)
        response = cv2.cornerHarris(
            gray_float,
            blockSize=2,
            ksize=3,
            k=0.04,
        )

        max_response = float(response.max())

        if max_response <= 0:
            return Image.fromarray(rgb), 0

        mask = response > threshold * max_response

        local_max = cv2.dilate(
            response,
            np.ones((3, 3), dtype=np.uint8),
        )
        mask &= response == local_max

        y_coords, x_coords = np.where(mask)
        strengths = response[y_coords, x_coords]

        if len(strengths) > max_features:
            selected = np.argpartition(
                strengths, -max_features
            )[-max_features:]
            x_coords = x_coords[selected]
            y_coords = y_coords[selected]

        result = rgb.copy()

        for x, y in zip(x_coords, y_coords):
            cv2.circle(
                result,
                (int(x), int(y)),
                radius=3,
                color=(255, 0, 0),
                thickness=1,
            )

        return Image.fromarray(result), len(x_coords)

    if detector == "SIFT":
        sift = cv2.SIFT_create(nfeatures=max_features)
        keypoints = sift.detect(gray, None)

    else:  # FAST
        fast = cv2.FastFeatureDetector_create(
            threshold=max(1, int(threshold * 255)),
            nonmaxSuppression=True,
        )
        keypoints = fast.detect(gray, None)

        keypoints = sorted(
            keypoints,
            key=lambda point: point.response,
            reverse=True,
        )[:max_features]

    result = cv2.drawKeypoints(
        rgb,
        keypoints,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return Image.fromarray(result), len(keypoints)


def _sift_descriptors(image: Image.Image):
    rgb = _to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=1500)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def compare_images(
    first_image: Image.Image,
    second_image: Image.Image,
) -> dict:
    first_keypoints, first_descriptors = _sift_descriptors(first_image)
    second_keypoints, second_descriptors = _sift_descriptors(second_image)

    if first_descriptors is None or second_descriptors is None:
        return {
            "similarity_percent": 0.0,
            "good_matches": 0,
            "first_keypoints": len(first_keypoints),
            "second_keypoints": len(second_keypoints),
        }

    if len(first_descriptors) < 2 or len(second_descriptors) < 2:
        return {
            "similarity_percent": 0.0,
            "good_matches": 0,
            "first_keypoints": len(first_keypoints),
            "second_keypoints": len(second_keypoints),
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(
        first_descriptors,
        second_descriptors,
        k=2,
    )

    good_matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]

    denominator = min(
        len(first_descriptors),
        len(second_descriptors),
    )

    similarity = 100.0 * len(good_matches) / denominator
    similarity = min(similarity, 100.0)

    return {
        "similarity_percent": round(similarity, 2),
        "good_matches": len(good_matches),
        "first_keypoints": len(first_keypoints),
        "second_keypoints": len(second_keypoints),
    }


def find_most_similar_pair(image_paths: list[str]) -> dict:
    if not 3 <= len(image_paths) <= 10:
        raise ValueError(
            "Для поиска необходимо выбрать от 3 до 10 изображений."
        )

    loaded_images = []

    for path in image_paths:
        try:
            with Image.open(path) as image:
                loaded_images.append((path, image.convert("RGB").copy()))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Не удалось открыть изображение {path}: {exc}"
            ) from exc

    best_pair = None
    best_score = -1.0

    for (path_a, image_a), (path_b, image_b) in combinations(
        loaded_images, 2
    ):
        comparison = compare_images(image_a, image_b)
        score = comparison["similarity_percent"]

        if score > best_score:
            best_score = score
            best_pair = {
                "first_path": path_a,
                "second_path": path_b,
                **comparison,
            }

    if best_pair is None:
        raise ValueError("Не удалось сравнить выбранные изображения.")

    return best_pair
