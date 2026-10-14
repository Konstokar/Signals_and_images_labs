import numpy as np
from PIL import Image
from matplotlib.figure import Figure


CHANNELS = {
    "Красный (R)": 0,
    "Зелёный (G)": 1,
    "Синий (B)": 2,
}

CHANNEL_COLORS = {
    "Красный (R)": "red",
    "Зелёный (G)": "green",
    "Синий (B)": "blue",
}


def _get_channel_data(image, channel):
    if not isinstance(image, Image.Image):
        raise TypeError("Ожидается объект PIL.Image.Image.")

    if channel == "Яркость":
        return np.asarray(image.convert("L"), dtype=np.uint8).ravel()

    if channel not in CHANNELS:
        raise ValueError(f"Неизвестный канал: {channel}")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb[:, :, CHANNELS[channel]].ravel()


def calculate_histogram(image, channel):
    pixels = _get_channel_data(image, channel)

    counts = np.bincount(pixels, minlength=256)
    values = np.arange(256)

    return values, counts


def create_histogram_figure(
    original_image,
    processed_image,
    channel="Яркость",
):
    if original_image is None or processed_image is None:
        raise ValueError("Необходимо загрузить изображение.")

    figure = Figure(figsize=(9, 6), tight_layout=True)

    axes = figure.subplots(2, 1)

    datasets = [
        (axes[0], original_image, "Исходное изображение"),
        (axes[1], processed_image, "Обработанное изображение"),
    ]

    for axis, image, title in datasets:
        axis.set_title(title)
        axis.set_xlim(0, 255)
        axis.set_xlabel("Интенсивность пикселя")
        axis.set_ylabel("Количество пикселей")
        axis.grid(True, alpha=0.25)

        if channel == "Все RGB":
            for channel_name, index in CHANNELS.items():
                rgb = np.asarray(
                    image.convert("RGB"),
                    dtype=np.uint8,
                )
                counts = np.bincount(
                    rgb[:, :, index].ravel(),
                    minlength=256,
                )

                axis.plot(
                    np.arange(256),
                    counts,
                    color=CHANNEL_COLORS[channel_name],
                    label=channel_name,
                    linewidth=1,
                    alpha=0.75,
                )

            axis.legend()
        else:
            values, counts = calculate_histogram(image, channel)

            color = (
                CHANNEL_COLORS.get(channel, "gray")
                if channel != "Яркость"
                else "dimgray"
            )

            axis.plot(
                values,
                counts,
                color=color,
                linewidth=1.2,
            )

            axis.fill_between(
                values,
                counts,
                color=color,
                alpha=0.2,
            )

    return figure
