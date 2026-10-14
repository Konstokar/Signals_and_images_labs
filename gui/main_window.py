from pathlib import Path

import cv2
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from image.histogram import create_histogram_figure
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QMessageBox, QScrollArea, QTabWidget, QComboBox,
    QDoubleSpinBox, QSpinBox, QLineEdit, QGridLayout,
    QCheckBox, QApplication,
)
from image.morphology import (
    SUPPORTED_OPERATIONS,
    apply_morphology,
)
from image.filters import SUPPORTED_FILTERS, apply_filter
from image.edges import SUPPORTED_EDGE_DETECTORS, detect_edges

from image.info import get_image_info
from image.color_models import SUPPORTED_MODELS, convert_color
from image.transforms import (
    to_grayscale, adjust_brightness, adjust_contrast,
    adjust_saturation, linear_correction, nonlinear_correction,
)
from image.features import (
    SUPPORTED_DETECTORS,
    detect_keypoints,
    find_most_similar_pair,
)
from video.background import process_video_background
from video.frame_difference import process_video_frame_difference
from video.motion_blur import (
    process_video_motion_blur,
    DIRECTIONS,
)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Лабораторная работа №1 — Обработка изображений")
        self.resize(1250, 800)

        self.original_image = None
        self.current_image = None
        self.current_file_path = None

        self._create_interface()

    def _create_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        image_layout = QVBoxLayout()

        self.image_label = QLabel("Загрузите изображение")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(
            "QLabel { background-color: #252525; color: #dddddd; }"
        )

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        self.open_button = QPushButton("Открыть изображение")
        self.open_button.clicked.connect(self.open_image)

        self.reset_button = QPushButton("Сбросить изменения")
        self.reset_button.clicked.connect(self.reset_image)

        image_layout.addWidget(self.scroll_area, stretch=1)
        image_layout.addWidget(self.open_button)
        image_layout.addWidget(self.reset_button)

        right_layout = QVBoxLayout()

        info_group = QGroupBox("Информация об изображении")
        info_layout = QFormLayout(info_group)
        self.info_labels = {}

        for key, title in [
            ("file_name", "Имя файла:"),
            ("file_size", "Размер файла:"),
            ("resolution", "Разрешение:"),
            ("format", "Формат:"),
            ("mode", "Режим цвета:"),
            ("color_depth", "Глубина цвета:"),
            ("channels", "Количество каналов:"),
            ("transparency", "Прозрачность:"),
        ]:
            label = QLabel("—")
            label.setWordWrap(True)
            self.info_labels[key] = label
            info_layout.addRow(title, label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_color_tab(), "Цветовые модели")
        self.tabs.addTab(self._create_transform_tab(), "Преобразования")
        self.tabs.addTab(self._create_histogram_tab(), "Гистограммы")
        self.tabs.addTab(self._create_morphology_tab(), "Морфология")
        self.tabs.addTab(self._create_filter_tab(), "Фильтры")
        self.tabs.addTab(self._create_edges_tab(), "Контуры")
        self.tabs.addTab(self._create_features_tab(), "Ключевые точки")
        self.tabs.addTab(self._create_video_tab(), "Видео")

        right_layout.addWidget(info_group)
        right_layout.addWidget(self.tabs, stretch=1)

        main_layout.addLayout(image_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=2)

    def _create_color_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Введите компоненты одного цвета. "
            "Для RGB и YCbCr используются значения 0–255; "
            "для HSL/HSV — градусы и проценты; "
            "для LAB — L: 0–100, a/b: -128–127; "
            "для CMYK — проценты."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        grid = QGridLayout()
        layout.addLayout(grid)

        grid.addWidget(QLabel("Исходная модель:"), 0, 0)
        self.source_model = QComboBox()
        self.source_model.addItems(SUPPORTED_MODELS)
        grid.addWidget(self.source_model, 0, 1)

        grid.addWidget(QLabel("Целевая модель:"), 1, 0)
        self.target_model = QComboBox()
        self.target_model.addItems(SUPPORTED_MODELS)
        self.target_model.setCurrentText("HSV")
        grid.addWidget(self.target_model, 1, 1)

        grid.addWidget(QLabel("Компоненты:"), 2, 0)
        self.color_input = QLineEdit("255, 0, 0")
        self.color_input.setPlaceholderText("Например: 255, 0, 0")
        grid.addWidget(self.color_input, 2, 1)

        self.convert_button = QPushButton("Преобразовать цвет")
        self.convert_button.clicked.connect(self.convert_pixel_color)
        layout.addWidget(self.convert_button)

        self.color_result = QLabel("Результат: —")
        self.color_result.setWordWrap(True)
        layout.addWidget(self.color_result)
        layout.addStretch()

        return tab

    def _create_transform_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        brightness_row = QHBoxLayout()
        brightness_row.addWidget(QLabel("Яркость:"))
        self.brightness_value = QSpinBox()
        self.brightness_value.setRange(-255, 255)
        self.brightness_value.setValue(0)
        brightness_row.addWidget(self.brightness_value)

        brightness_button = QPushButton("Применить яркость")
        brightness_button.clicked.connect(self.apply_brightness)
        brightness_row.addWidget(brightness_button)
        layout.addLayout(brightness_row)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Контрастность:"))
        self.contrast_value = QDoubleSpinBox()
        self.contrast_value.setRange(0, 4)
        self.contrast_value.setSingleStep(0.1)
        self.contrast_value.setValue(1.0)
        contrast_row.addWidget(self.contrast_value)

        contrast_button = QPushButton("Применить контраст")
        contrast_button.clicked.connect(self.apply_contrast)
        contrast_row.addWidget(contrast_button)
        layout.addLayout(contrast_row)

        saturation_row = QHBoxLayout()
        saturation_row.addWidget(QLabel("Насыщенность:"))
        self.saturation_value = QDoubleSpinBox()
        self.saturation_value.setRange(0, 4)
        self.saturation_value.setSingleStep(0.1)
        self.saturation_value.setValue(1.0)
        saturation_row.addWidget(self.saturation_value)

        saturation_button = QPushButton("Применить насыщенность")
        saturation_button.clicked.connect(self.apply_saturation)
        saturation_row.addWidget(saturation_button)
        layout.addLayout(saturation_row)

        gray_button = QPushButton("Преобразовать в градации серого")
        gray_button.clicked.connect(self.apply_grayscale)
        layout.addWidget(gray_button)

        linear_row = QHBoxLayout()
        self.black_point = QSpinBox()
        self.black_point.setRange(0, 254)
        self.black_point.setValue(0)
        self.white_point = QSpinBox()
        self.white_point.setRange(1, 255)
        self.white_point.setValue(255)
        linear_row.addWidget(QLabel("Чёрный уровень:"))
        linear_row.addWidget(self.black_point)
        linear_row.addWidget(QLabel("Белый уровень:"))
        linear_row.addWidget(self.white_point)
        layout.addLayout(linear_row)

        linear_button = QPushButton("Линейная коррекция")
        linear_button.clicked.connect(self.apply_linear_correction)
        layout.addWidget(linear_button)

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma:"))
        self.gamma_value = QDoubleSpinBox()
        self.gamma_value.setRange(0.1, 5)
        self.gamma_value.setSingleStep(0.1)
        self.gamma_value.setValue(1.0)
        gamma_row.addWidget(self.gamma_value)

        gamma_button = QPushButton("Гамма-коррекция")
        gamma_button.clicked.connect(self.apply_gamma_correction)
        gamma_row.addWidget(gamma_button)
        layout.addLayout(gamma_row)

        layout.addStretch()
        return tab

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
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
            self.statusBar().showMessage(f"Загружено: {Path(file_path).name}")

        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))

    def _display_image(self, image):
        image = image.convert("RGB")
        width, height = image.size
        q_image = QImage(
            image.tobytes(), width, height, width * 3,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(q_image)
        size = self.scroll_area.viewport().size()
        scaled = pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def _display_image_info(self, file_path):
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
            "transparency": "Есть" if info["has_transparency"] else "Нет",
        }
        for key, value in values.items():
            self.info_labels[key].setText(value)

    def _apply_transform(self, function, *args):
        if self.original_image is None:
            QMessageBox.information(
                self, "Нет изображения", "Сначала загрузите изображение."
            )
            return

        try:
            self.current_image = function(self.current_image, *args)
            self._display_image(self.current_image)
            self.statusBar().showMessage("Преобразование применено.")
            if self.tabs.currentWidget() is not None:
                if self.tabs.currentIndex() == 2:
                    self.update_histograms()
        except (ValueError, TypeError, cv2.error) as exc:
            QMessageBox.warning(self, "Ошибка преобразования", str(exc))

    def apply_brightness(self):
        self._apply_transform(adjust_brightness, self.brightness_value.value())

    def apply_contrast(self):
        self._apply_transform(adjust_contrast, self.contrast_value.value())

    def apply_saturation(self):
        self._apply_transform(adjust_saturation, self.saturation_value.value())

    def apply_grayscale(self):
        if self.current_image is None:
            QMessageBox.information(self, "Нет изображения", "Сначала загрузите изображение.")
            return
        self.current_image = to_grayscale(self.current_image)
        self._display_image(self.current_image)

    def apply_linear_correction(self):
        if self.current_image is None:
            QMessageBox.information(self, "Нет изображения", "Сначала загрузите изображение.")
            return
        try:
            self.current_image = linear_correction(
                self.current_image,
                self.black_point.value(),
                self.white_point.value(),
            )
            self._display_image(self.current_image)
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка коррекции", str(exc))

    def apply_gamma_correction(self):
        if self.current_image is None:
            QMessageBox.information(self, "Нет изображения", "Сначала загрузите изображение.")
            return
        self.current_image = nonlinear_correction(
            self.current_image, self.gamma_value.value()
        )
        self._display_image(self.current_image)

    def convert_pixel_color(self):
        try:
            components = [
                float(component.strip())
                for component in self.color_input.text().split(",")
            ]
            result = convert_color(
                components,
                self.source_model.currentText(),
                self.target_model.currentText(),
            )
            self.color_result.setText(
                "Результат: " + ", ".join(f"{value:.2f}" for value in result)
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка конвертации", str(exc))

    def reset_image(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self._display_image(self.current_image)
        self.statusBar().showMessage("Исходное изображение восстановлено.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None:
            self._display_image(self.current_image)


    def _create_histogram_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Сравнение распределения интенсивностей пикселей "
            "исходного и обработанного изображений."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Отображаемый канал:"))

        self.histogram_channel = QComboBox()
        self.histogram_channel.addItems([
            "Яркость",
            "Красный (R)",
            "Зелёный (G)",
            "Синий (B)",
            "Все RGB",
        ])

        controls.addWidget(self.histogram_channel)

        self.histogram_button = QPushButton("Построить гистограммы")
        self.histogram_button.clicked.connect(
            self.update_histograms
        )

        controls.addWidget(self.histogram_button)
        layout.addLayout(controls)

        self.histogram_canvas = FigureCanvasQTAgg(
            create_histogram_figure(
                Image.new("RGB", (1, 1), "black"),
                Image.new("RGB", (1, 1), "black"),
            )
        )

        layout.addWidget(self.histogram_canvas, stretch=1)

        return tab

    def update_histograms(self):
        if self.original_image is None or self.current_image is None:
            QMessageBox.information(
                self,
                "Нет изображения",
                "Сначала загрузите изображение.",
            )
            return

        try:
            figure = create_histogram_figure(
                self.original_image,
                self.current_image,
                self.histogram_channel.currentText(),
            )

            self.histogram_canvas.figure.clear()
            self.histogram_canvas.figure = figure
            self.histogram_canvas.draw()

            self.statusBar().showMessage(
                "Гистограммы построены."
            )

        except (ValueError, TypeError) as exc:
            QMessageBox.warning(
                self,
                "Ошибка построения гистограммы",
                str(exc),
            )

    def _create_morphology_tab(self):

        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Морфологические операции изменяют структуру светлых "
            "областей изображения. Обработка выполняется в градациях "
            "серого с использованием квадратного структурного элемента."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        operation_layout = QHBoxLayout()
        operation_layout.addWidget(QLabel("Операция:"))

        self.morphology_operation = QComboBox()
        self.morphology_operation.addItems(SUPPORTED_OPERATIONS)
        operation_layout.addWidget(self.morphology_operation)

        layout.addLayout(operation_layout)

        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Размер ядра:"))

        self.morphology_kernel_size = QSpinBox()
        self.morphology_kernel_size.setRange(1, 31)
        self.morphology_kernel_size.setSingleStep(2)
        self.morphology_kernel_size.setValue(3)

        kernel_layout.addWidget(self.morphology_kernel_size)
        layout.addLayout(kernel_layout)

        iterations_layout = QHBoxLayout()
        iterations_layout.addWidget(QLabel("Повторения:"))

        self.morphology_iterations = QSpinBox()
        self.morphology_iterations.setRange(1, 10)
        self.morphology_iterations.setValue(1)

        iterations_layout.addWidget(self.morphology_iterations)
        layout.addLayout(iterations_layout)

        self.morphology_button = QPushButton(
            "Применить морфологическую операцию"
        )
        self.morphology_button.clicked.connect(
            self.apply_morphology_operation
        )
        layout.addWidget(self.morphology_button)

        layout.addWidget(
            QLabel(
                "Результат появится в основной области просмотра. "
                "Для возврата к исходнику используйте кнопку сброса."
            )
        )

        layout.addStretch()
        return tab

    def apply_morphology_operation(self):

        if self.current_image is None:
            QMessageBox.information(
                self,
                "Нет изображения",
                "Сначала загрузите изображение.",
            )
            return

        try:
            result = apply_morphology(
                self.current_image,
                self.morphology_operation.currentText(),
                self.morphology_kernel_size.value(),
                self.morphology_iterations.value(),
            )

            self.current_image = result
            self._display_image(result)

            self.statusBar().showMessage(
                "Морфологическая операция применена."
            )

            if self.tabs.currentIndex() == 2:
                self.update_histograms()

        except (ValueError, TypeError, cv2.error) as exc:
            QMessageBox.warning(
                self,
                "Ошибка морфологической обработки",
                str(exc),
            )

    def _create_filter_tab(self):

        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Выберите фильтр и настройте его параметры. "
            "Результат будет показан в основной области просмотра."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        layout.addWidget(QLabel("Фильтр:"))

        self.filter_selector = QComboBox()
        self.filter_selector.addItems(SUPPORTED_FILTERS)
        self.filter_selector.currentTextChanged.connect(
            self._update_filter_controls
        )
        layout.addWidget(self.filter_selector)

        layout.addWidget(QLabel("Размер ядра:"))

        self.filter_kernel_size = QSpinBox()
        self.filter_kernel_size.setRange(3, 51)
        self.filter_kernel_size.setSingleStep(2)
        self.filter_kernel_size.setValue(15)
        layout.addWidget(self.filter_kernel_size)

        layout.addWidget(QLabel("Сила эффекта:"))

        self.filter_strength = QDoubleSpinBox()
        self.filter_strength.setRange(0.1, 5.0)
        self.filter_strength.setSingleStep(0.1)
        self.filter_strength.setValue(1.0)
        layout.addWidget(self.filter_strength)

        layout.addWidget(QLabel("Направление размытия:"))

        self.filter_direction = QComboBox()
        self.filter_direction.addItems([
            "Горизонтально",
            "Вертикально",
            "Диагонально",
        ])
        layout.addWidget(self.filter_direction)

        self.filter_apply_button = QPushButton("Применить фильтр")
        self.filter_apply_button.clicked.connect(self.apply_selected_filter)
        layout.addWidget(self.filter_apply_button)

        layout.addStretch()

        self._update_filter_controls(
            self.filter_selector.currentText()
        )

        return tab

    def _update_filter_controls(self, filter_name):

        uses_kernel = filter_name in (
            "Размытие в движении",
            "Медианная фильтрация",
        )
        uses_strength = filter_name in (
            "Повышение резкости",
            "Тиснение",
        )
        uses_direction = filter_name == "Размытие в движении"

        self.filter_kernel_size.setEnabled(uses_kernel)
        self.filter_strength.setEnabled(uses_strength)
        self.filter_direction.setEnabled(uses_direction)

        if filter_name == "Размытие в движении":
            self.filter_kernel_size.setValue(15)
        elif filter_name == "Медианная фильтрация":
            self.filter_kernel_size.setValue(3)

    def apply_selected_filter(self):

        if self.current_image is None:
            QMessageBox.information(
                self,
                "Нет изображения",
                "Сначала загрузите изображение.",
            )
            return

        try:
            result = apply_filter(
                image=self.current_image,
                filter_name=self.filter_selector.currentText(),
                kernel_size=self.filter_kernel_size.value(),
                strength=self.filter_strength.value(),
                direction=self.filter_direction.currentText(),
            )

            self.current_image = result
            self._display_image(result)

            self.statusBar().showMessage(
                f"Применён фильтр: {self.filter_selector.currentText()}."
            )

        except (ValueError, TypeError, cv2.error) as exc:
            QMessageBox.warning(
                self,
                "Ошибка фильтрации",
                str(exc),
            )

    def _create_edges_tab(self):

        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Обнаружение границ позволяет выделять переходы яркости "
            "и контуры объектов. Доступны алгоритмы Canny и Roberts."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        layout.addWidget(QLabel("Детектор границ:"))

        self.edge_detector = QComboBox()
        self.edge_detector.addItems(SUPPORTED_EDGE_DETECTORS)
        self.edge_detector.currentTextChanged.connect(
            self._update_edge_controls
        )
        layout.addWidget(self.edge_detector)

        layout.addWidget(QLabel("Нижний порог Canny:"))

        self.canny_low_threshold = QSpinBox()
        self.canny_low_threshold.setRange(0, 255)
        self.canny_low_threshold.setValue(50)
        layout.addWidget(self.canny_low_threshold)

        layout.addWidget(QLabel("Верхний порог Canny:"))

        self.canny_high_threshold = QSpinBox()
        self.canny_high_threshold.setRange(0, 255)
        self.canny_high_threshold.setValue(150)
        layout.addWidget(self.canny_high_threshold)

        layout.addWidget(QLabel("Предварительное размытие Canny:"))

        self.canny_blur_size = QComboBox()
        self.canny_blur_size.addItems(["1", "3", "5", "7", "9"])
        self.canny_blur_size.setCurrentText("3")
        layout.addWidget(self.canny_blur_size)

        layout.addWidget(QLabel("Порог Roberts:"))

        self.roberts_threshold = QSpinBox()
        self.roberts_threshold.setRange(0, 255)
        self.roberts_threshold.setValue(40)
        layout.addWidget(self.roberts_threshold)

        self.edge_apply_button = QPushButton(
            "Обнаружить границы"
        )
        self.edge_apply_button.clicked.connect(
            self.apply_edge_detector
        )
        layout.addWidget(self.edge_apply_button)

        layout.addWidget(
            QLabel(
                "Результат отображается как чёрно-белая карта границ. "
                "Для нового эксперимента можно восстановить исходник."
            )
        )

        layout.addStretch()

        self._update_edge_controls(
            self.edge_detector.currentText()
        )

        return tab

    def _update_edge_controls(self, detector):

        is_canny = detector == "Canny"
        is_roberts = detector == "Roberts"

        self.canny_low_threshold.setEnabled(is_canny)
        self.canny_high_threshold.setEnabled(is_canny)
        self.canny_blur_size.setEnabled(is_canny)
        self.roberts_threshold.setEnabled(is_roberts)

    def apply_edge_detector(self):

        if self.current_image is None:
            QMessageBox.information(
                self,
                "Нет изображения",
                "Сначала загрузите изображение.",
            )
            return

        try:
            result = detect_edges(
                image=self.current_image,
                detector=self.edge_detector.currentText(),
                low_threshold=self.canny_low_threshold.value(),
                high_threshold=self.canny_high_threshold.value(),
                roberts_threshold=self.roberts_threshold.value(),
                blur_size=int(self.canny_blur_size.currentText()),
            )

            self.current_image = result
            self._display_image(result)

            self.statusBar().showMessage(
                f"Обнаружение границ: {self.edge_detector.currentText()}."
            )

        except (ValueError, TypeError, cv2.error) as exc:
            QMessageBox.warning(
                self,
                "Ошибка обнаружения границ",
                str(exc),
            )

    def _create_features_tab(self):

        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Обнаружение ключевых точек"))

        self.feature_detector = QComboBox()
        self.feature_detector.addItems(SUPPORTED_DETECTORS)
        layout.addWidget(self.feature_detector)

        layout.addWidget(QLabel("Максимум ключевых точек:"))

        self.feature_max_points = QSpinBox()
        self.feature_max_points.setRange(1, 10000)
        self.feature_max_points.setValue(500)
        layout.addWidget(self.feature_max_points)

        layout.addWidget(QLabel("Порог чувствительности:"))

        self.feature_threshold = QDoubleSpinBox()
        self.feature_threshold.setRange(0.001, 0.2)
        self.feature_threshold.setDecimals(3)
        self.feature_threshold.setSingleStep(0.005)
        self.feature_threshold.setValue(0.01)
        layout.addWidget(self.feature_threshold)

        self.feature_apply_button = QPushButton(
            "Обнаружить ключевые точки"
        )
        self.feature_apply_button.clicked.connect(
            self.apply_feature_detector
        )
        layout.addWidget(self.feature_apply_button)

        self.feature_result_label = QLabel("Ключевых точек: —")
        layout.addWidget(self.feature_result_label)

        layout.addWidget(QLabel("Поиск наиболее похожей пары"))

        self.similarity_button = QPushButton(
            "Выбрать 3–10 изображений"
        )
        self.similarity_button.clicked.connect(
            self.find_similar_images
        )
        layout.addWidget(self.similarity_button)

        self.similarity_result_label = QLabel(
            "Результат сравнения появится здесь."
        )
        self.similarity_result_label.setWordWrap(True)
        layout.addWidget(self.similarity_result_label)

        layout.addStretch()
        return tab

    def apply_feature_detector(self):

        if self.current_image is None:
            QMessageBox.information(
                self,
                "Нет изображения",
                "Сначала загрузите изображение.",
            )
            return

        try:
            result, count = detect_keypoints(
                self.current_image,
                self.feature_detector.currentText(),
                self.feature_max_points.value(),
                self.feature_threshold.value(),
            )

            self.current_image = result
            self._display_image(result)

            self.feature_result_label.setText(
                f"Обнаружено ключевых точек: {count}"
            )

            self.statusBar().showMessage(
                f"Детектор {self.feature_detector.currentText()}: "
                f"{count} точек."
            )

        except (ValueError, TypeError, cv2.error) as exc:
            QMessageBox.warning(
                self,
                "Ошибка обнаружения точек",
                str(exc),
            )

    def find_similar_images(self):

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите от 3 до 10 изображений",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )

        if not paths:
            return

        if not 3 <= len(paths) <= 10:
            QMessageBox.warning(
                self,
                "Некорректное количество файлов",
                "Выберите от 3 до 10 изображений.",
            )
            return

        try:
            result = find_most_similar_pair(paths)

            self.similarity_result_label.setText(
                "Наиболее похожая пара:\n"
                f"1. {result['first_path']}\n"
                f"2. {result['second_path']}\n\n"
                f"Оценка сходства: {result['similarity_percent']:.2f}%\n"
                f"Хороших совпадений: {result['good_matches']}"
            )

            self.statusBar().showMessage(
                "Поиск наиболее похожей пары завершён."
            )

        except (ValueError, TypeError, cv2.error) as exc:
            QMessageBox.warning(
                self,
                "Ошибка сравнения изображений",
                str(exc),
            )



    def _create_video_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Сравнение двух методов обнаружения движения: MOG2 "
            "и разности соседних кадров. Можно сохранить маску "
            "и видео с красным наложением на исходный ролик."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.video_path_label = QLabel("Видео не выбрано.")
        self.video_path_label.setWordWrap(True)
        layout.addWidget(self.video_path_label)

        self.video_open_button = QPushButton("Выбрать видео")
        self.video_open_button.clicked.connect(self.open_video)
        layout.addWidget(self.video_open_button)

        layout.addWidget(QLabel("Алгоритм обнаружения движения:"))

        self.video_algorithm = QComboBox()
        self.video_algorithm.addItems([
            "MOG2",
            "Разность соседних кадров",
        ])
        self.video_algorithm.currentTextChanged.connect(
            self._update_video_algorithm_controls
        )
        layout.addWidget(self.video_algorithm)

        self.video_mog2_group = QGroupBox("Параметры MOG2")
        mog2_layout = QVBoxLayout(self.video_mog2_group)

        mog2_layout.addWidget(
            QLabel("История модели (кадры):")
        )
        self.video_history = QSpinBox()
        self.video_history.setRange(1, 10000)
        self.video_history.setValue(120)
        mog2_layout.addWidget(self.video_history)

        mog2_layout.addWidget(
            QLabel("Порог чувствительности MOG2:")
        )
        self.video_threshold = QDoubleSpinBox()
        self.video_threshold.setRange(1.0, 100.0)
        self.video_threshold.setDecimals(1)
        self.video_threshold.setValue(25.0)
        mog2_layout.addWidget(self.video_threshold)

        self.video_shadows = QCheckBox(
            "Учитывать тени при вычитании фона"
        )
        self.video_shadows.setChecked(False)
        mog2_layout.addWidget(self.video_shadows)

        mog2_layout.addWidget(
            QLabel("Скорость адаптации фона:")
        )
        self.video_learning_rate = QDoubleSpinBox()
        self.video_learning_rate.setRange(0.001, 0.1)
        self.video_learning_rate.setDecimals(3)
        self.video_learning_rate.setSingleStep(0.005)
        self.video_learning_rate.setValue(0.01)
        mog2_layout.addWidget(self.video_learning_rate)

        layout.addWidget(self.video_mog2_group)

        self.video_difference_group = QGroupBox(
            "Параметры разности кадров"
        )
        difference_layout = QVBoxLayout(
            self.video_difference_group
        )

        difference_layout.addWidget(
            QLabel("Порог изменения яркости (0–255):")
        )
        self.video_diff_threshold = QSpinBox()
        self.video_diff_threshold.setRange(0, 255)
        self.video_diff_threshold.setValue(25)
        difference_layout.addWidget(
            self.video_diff_threshold
        )

        layout.addWidget(self.video_difference_group)

        common_group = QGroupBox("Общие параметры")
        common_layout = QVBoxLayout(common_group)

        common_layout.addWidget(
            QLabel("Количество начальных кадров обучения:")
        )
        self.video_warmup = QSpinBox()
        self.video_warmup.setRange(0, 1000)
        self.video_warmup.setValue(10)
        common_layout.addWidget(self.video_warmup)

        common_layout.addWidget(
            QLabel("Порог обнаружения смены сцены:")
        )
        self.video_scene_threshold = QDoubleSpinBox()
        self.video_scene_threshold.setRange(0.1, 0.9)
        self.video_scene_threshold.setDecimals(2)
        self.video_scene_threshold.setSingleStep(0.05)
        self.video_scene_threshold.setValue(0.45)
        common_layout.addWidget(
            self.video_scene_threshold
        )

        common_layout.addWidget(
            QLabel("Размер ядра очистки маски:")
        )
        self.video_morphology_kernel = QSpinBox()
        self.video_morphology_kernel.setRange(1, 31)
        self.video_morphology_kernel.setSingleStep(2)
        self.video_morphology_kernel.setValue(3)
        common_layout.addWidget(
            self.video_morphology_kernel
        )

        common_layout.addWidget(
            QLabel("Минимальная площадь области (пиксели):")
        )
        self.video_min_area = QSpinBox()
        self.video_min_area.setRange(0, 10000)
        self.video_min_area.setValue(20)
        common_layout.addWidget(self.video_min_area)

        layout.addWidget(common_group)

        self.video_background_button = QPushButton(
            "Обнаружить движение и сохранить результаты"
        )
        self.video_background_button.clicked.connect(
            self.run_video_background
        )
        layout.addWidget(self.video_background_button)

        layout.addWidget(QLabel("Размытие движения"))

        layout.addWidget(
            QLabel("Размер ядра (нечётное число):")
        )
        self.video_blur_kernel = QSpinBox()
        self.video_blur_kernel.setRange(3, 51)
        self.video_blur_kernel.setSingleStep(2)
        self.video_blur_kernel.setValue(15)
        layout.addWidget(self.video_blur_kernel)

        layout.addWidget(
            QLabel("Направление размытия:")
        )
        self.video_blur_direction = QComboBox()
        self.video_blur_direction.addItems(DIRECTIONS)
        layout.addWidget(self.video_blur_direction)

        self.video_blur_button = QPushButton(
            "Размыть движущиеся объекты и сохранить видео"
        )
        self.video_blur_button.clicked.connect(
            self.run_video_motion_blur
        )
        layout.addWidget(self.video_blur_button)

        self.video_result_label = QLabel(
            "Выберите ролик и алгоритм обработки."
        )
        self.video_result_label.setWordWrap(True)
        layout.addWidget(self.video_result_label)

        layout.addStretch()

        self._update_video_algorithm_controls(
            self.video_algorithm.currentText()
        )

        return tab

    def _update_video_algorithm_controls(self, algorithm):
        is_mog2 = algorithm == "MOG2"

        self.video_mog2_group.setVisible(is_mog2)
        self.video_difference_group.setVisible(not is_mog2)



    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Видео (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv)",
        )

        if not path:
            return

        capture = cv2.VideoCapture(path)

        if not capture.isOpened():
            capture.release()
            QMessageBox.warning(
                self,
                "Ошибка открытия",
                "OpenCV не смог открыть выбранный видеофайл.",
            )
            return

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        self.video_path = path

        self.video_path_label.setText(
            f"Файл: {Path(path).name}\n"
            f"Разрешение: {width} × {height}\n"
            f"FPS: {fps:.2f} | Кадров: {frames}"
        )

        self.video_result_label.setText(
            "Видео готово к обработке."
        )
        self.statusBar().showMessage(
            f"Выбрано видео: {Path(path).name}"
        )

    def _choose_video_output(self, title):
        path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            "",
            "Видео MP4 (*.mp4)",
        )

        if path and Path(path).suffix.lower() != ".mp4":
            path = str(Path(path).with_suffix(".mp4"))

        return path

    def _video_progress(self, processed, total):
        if total > 0:
            percent = min(
                100,
                int(processed * 100 / total),
            )

            self.video_result_label.setText(
                f"Обработка видео: {processed} из {total} "
                f"кадров ({percent}%)."
            )
        else:
            self.video_result_label.setText(
                f"Обработано кадров: {processed}."
            )

        QApplication.processEvents()



    def run_video_background(self):
        if not getattr(self, "video_path", None):
            QMessageBox.information(
                self,
                "Нет видео",
                "Сначала выберите видеоролик.",
            )
            return

        output_path = self._choose_video_output(
            "Сохранить маску движения"
        )

        if not output_path:
            return

        output_file = Path(output_path)

        overlay_path = str(
            output_file.with_name(
                f"{output_file.stem}_overlay.mp4"
            )
        )

        # Не допускаем перезаписи исходника.
        if Path(self.video_path).resolve() in (
            Path(output_path).resolve(),
            Path(overlay_path).resolve(),
        ):
            QMessageBox.warning(
                self,
                "Некорректный путь",
                "Выходные файлы не должны совпадать с исходным видео.",
            )
            return

        algorithm = self.video_algorithm.currentText()

        self.video_background_button.setEnabled(False)
        self.video_blur_button.setEnabled(False)
        self.video_open_button.setEnabled(False)

        try:
            if algorithm == "MOG2":
                result = process_video_background(
                    input_path=self.video_path,
                    output_path=output_path,
                    history=self.video_history.value(),
                    var_threshold=self.video_threshold.value(),
                    detect_shadows=self.video_shadows.isChecked(),
                    warmup_frames=self.video_warmup.value(),
                    learning_rate=self.video_learning_rate.value(),
                    scene_change_threshold=(
                        self.video_scene_threshold.value()
                    ),
                    morphology_kernel=(
                        self.video_morphology_kernel.value()
                    ),
                    min_component_area=self.video_min_area.value(),
                    overlay_output_path=overlay_path,
                    progress_callback=self._video_progress,
                )

            else:
                result = process_video_frame_difference(
                    input_path=self.video_path,
                    output_path=output_path,
                    difference_threshold=(
                        self.video_diff_threshold.value()
                    ),
                    warmup_frames=self.video_warmup.value(),
                    scene_change_threshold=(
                        self.video_scene_threshold.value()
                    ),
                    morphology_kernel=(
                        self.video_morphology_kernel.value()
                    ),
                    min_component_area=self.video_min_area.value(),
                    overlay_output_path=overlay_path,
                    progress_callback=self._video_progress,
                )

            self.video_result_label.setText(
                f"Алгоритм: {algorithm}\n"
                "Обработка завершена.\n"
                f"Маска: {result['output_path']}\n"
                f"Наложение: {result['overlay_output_path']}\n"
                f"Кадров обработано: "
                f"{result['frames_processed']}\n"
                f"Разрешение: {result['width']} × "
                f"{result['height']}\n"
                f"Обнаружено смен сцен: "
                f"{result['scene_changes_detected']}\n"
                f"Средняя доля переднего плана: "
                f"{result['average_foreground_percent']}%"
            )

            self.statusBar().showMessage(
                f"Обнаружение движения завершено: {algorithm}."
            )

        except (ValueError, OSError, cv2.error) as exc:
            QMessageBox.warning(
                self,
                "Ошибка обработки видео",
                str(exc),
            )

        finally:
            self.video_background_button.setEnabled(True)
            self.video_blur_button.setEnabled(True)
            self.video_open_button.setEnabled(True)

    def run_video_motion_blur(self):
        if not getattr(self, "video_path", None):
            QMessageBox.information(
                self,
                "Нет видео",
                "Сначала выберите видеоролик.",
            )
            return

        output_path = self._choose_video_output(
            "Сохранить видео с размытием движения"
        )

        if not output_path:
            return

        self.video_background_button.setEnabled(False)
        self.video_blur_button.setEnabled(False)

        try:
            result = process_video_motion_blur(
                self.video_path,
                output_path,
                kernel_size=self.video_blur_kernel.value(),
                direction=self.video_blur_direction.currentText(),
                history=self.video_history.value(),
                var_threshold=self.video_threshold.value(),
                warmup_frames=self.video_warmup.value(),
                progress_callback=self._video_progress,
            )

            self.video_result_label.setText(
                "Размытие движения завершено.\n"
                f"Файл: {result['output_path']}\n"
                f"Кадров обработано: {result['frames_processed']}\n"
                f"Разрешение: {result['width']} × "
                f"{result['height']}\n"
                "Исходная звуковая дорожка не сохраняется."
            )

            self.statusBar().showMessage(
                "Размытие движения завершено."
            )

        except (ValueError, OSError, cv2.error) as exc:
            QMessageBox.warning(
                self,
                "Ошибка обработки видео",
                str(exc),
            )

        finally:
            self.video_background_button.setEnabled(True)
            self.video_blur_button.setEnabled(True)
