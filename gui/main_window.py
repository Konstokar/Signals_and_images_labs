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
