from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QScrollArea,
)

from image.info import get_image_info


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Лабораторная работа №1 — Обработка изображений"
        )
        self.resize(1200, 800)

        self.original_image = None
        self.current_image = None
        self.current_file_path = None

        self._create_interface()

    def _create_interface(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        image_layout = QVBoxLayout()

        self.image_label = QLabel(
            "Загрузите изображение для начала работы"
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(
            "QLabel {"
            "background-color: #252525;"
            "color: #dddddd;"
            "border: 1px solid #555555;"
            "}"
        )

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        self.open_button = QPushButton("Открыть изображение")
        self.open_button.clicked.connect(self.open_image)

        image_layout.addWidget(self.scroll_area, stretch=1)
        image_layout.addWidget(self.open_button)

        side_layout = QVBoxLayout()

        info_group = QGroupBox("Информация об изображении")
        info_layout = QFormLayout(info_group)

        self.info_labels = {}

        fields = [
            ("file_name", "Имя файла:"),
            ("file_size", "Размер файла:"),
            ("resolution", "Разрешение:"),
            ("format", "Формат:"),
            ("mode", "Режим цвета:"),
            ("color_depth", "Глубина цвета:"),
            ("channels", "Количество каналов:"),
            ("transparency", "Прозрачность:"),
        ]

        for key, title in fields:
            value_label = QLabel("—")
            value_label.setWordWrap(True)
            self.info_labels[key] = value_label
            info_layout.addRow(title, value_label)

        tools_group = QGroupBox("Обработка изображения")
        tools_layout = QVBoxLayout(tools_group)

        tools_layout.addWidget(
            QLabel(
                "На следующих этапах здесь появятся инструменты "
                "цветовых преобразований, фильтрации, морфологии "
                "и анализа контуров."
            )
        )

        side_layout.addWidget(info_group)
        side_layout.addWidget(tools_group)
        side_layout.addStretch()

        main_layout.addLayout(image_layout, stretch=3)
        main_layout.addLayout(side_layout, stretch=2)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            (
                "Изображения (*.png *.jpg *.jpeg *.bmp *.tif "
                "*.tiff *.webp);;Все файлы (*)"
            ),
        )

        if not file_path:
            return

        try:
            with Image.open(file_path) as img:
                img.load()
                self.original_image = img.convert("RGB").copy()

            self.current_image = self.original_image.copy()
            self.current_file_path = file_path

            self._display_image(self.current_image)
            self._display_image_info(file_path)

            self.statusBar().showMessage(
                f"Загружено: {Path(file_path).name}"
            )

        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Ошибка загрузки",
                f"Не удалось открыть изображение:\n{exc}",
            )

    def _display_image(self, image):

        image = image.convert("RGB")
        width, height = image.size

        raw_data = image.tobytes()
        q_image = QImage(
            raw_data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(q_image)

        available_size = self.scroll_area.viewport().size()

        scaled_pixmap = pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    def _display_image_info(self, file_path: str):
        info = get_image_info(file_path)

        values = {
            "file_name": info["file_name"],
            "file_size": (
                f'{info["file_size_kb"]:.2f} КБ '
                f'({info["file_size_bytes"]} байт)'
            ),
            "resolution": info["resolution"],
            "format": info["format"],
            "mode": info["mode"],
            "color_depth": info["color_depth_text"],
            "channels": str(info["channels"]),
            "transparency": (
                "Есть" if info["has_transparency"] else "Нет"
            ),
        }

        for key, value in values.items():
            self.info_labels[key].setText(value)

    def resizeEvent(self, event):

        super().resizeEvent(event)

        if self.current_image is not None:
            self._display_image(self.current_image)
