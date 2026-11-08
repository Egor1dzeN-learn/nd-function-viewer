"""Visualization of functions of one to four variables.

f(x)          - a curve whose color depends on the function value.
f(x, y)       - a 3D surface that can be rotated with the mouse; lowlands are
                blue ("sea"), peaks are red ("mountains").
f(x, y, z)    - the same surface as a slice; the remaining variable is a slider
f(x, y, z, w)   that can be animated (the two remaining variables in 4D), or a
                grid of heatmap slices (dimensional stacking).
f(x1, ..., xN)  N > 4: HyperSlice, a matrix of slices through a movable focus point.
"""
import itertools
import math
import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtGui, QtWidgets

import plugin_loader
import test_functions as tf
from colormaps import COLORMAP_NAMES, get_colormap
from function_parser import HELP_TEXT, VARIABLES, Formula, FormulaError, variable_names
from hyperslice import HyperSliceView

SLICED_DIMENSIONS = len(VARIABLES)  # above this dimension the function is shown by HyperSlice


def _indexed_sum(term: str, count: int, start: int = 1) -> str:
    """Formula text: term with # and @ replaced by i and i + 1, summed over i."""
    return " + ".join(
        term.replace("#", str(i)).replace("@", str(i + 1)) for i in range(start, start + count)
    )


# (name, formula, ranges): variables without a range use the range of x
PRESETS = [
    ("Gaussian hill and pit", "exp(-((x-1)^2 + (y-1)^2)) - 0.8*exp(-((x+1)^2 + (y+1)^2)/0.5)", [(-3, 3)]),
    ("Peaks", "3*(1-x)^2*exp(-x^2-(y+1)^2) - 10*(x/5-x^3-y^5)*exp(-x^2-y^2) - exp(-(x+1)^2-y^2)/3", [(-3, 3)]),
    ("Paraboloid", "x^2 + y^2", [(-3, 3)]),
    ("Saddle", "x^2 - y^2", [(-2, 2)]),
    ("Waves", "sin(x)*cos(y)", [(-2 * np.pi, 2 * np.pi)]),
    ("Rastrigin", "20 + x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y)", [(-5.12, 5.12)]),
    ("Ackley", "-20*exp(-0.2*sqrt(0.5*(x^2+y^2))) - exp(0.5*(cos(2*pi*x)+cos(2*pi*y))) + e + 20", [(-5, 5)]),
    ("Himmelblau", "(x^2 + y - 11)^2 + (x + y^2 - 7)^2", [(-5, 5)]),
    ("Rosenbrock", "(1-x)^2 + 100*(y-x^2)^2", [(-2, 2), (-1, 3)]),
    ("1D: x·sin(x)", "x*sin(x)", [(-10, 10)]),
    ("1D: Rastrigin", "10 + x^2 - 10*cos(2*pi*x)", [(-5.12, 5.12)]),
    ("3D: Moving hill", "exp(-((x-z)^2 + (y-sin(2*z))^2))", [(-3, 3)]),
    ("3D: Ripple", "sin(3*sqrt(x^2 + y^2) - 2*z)", [(-4, 4), (-4, 4), (-3.1416, 3.1416)]),
    ("3D: Sphere", "x^2 + y^2 + z^2", [(-3, 3)]),
    ("3D: Rastrigin", "30 + x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y) + z^2 - 10*cos(2*pi*z)", [(-5.12, 5.12)]),
    ("4D: Sphere", "x^2 + y^2 + z^2 + w^2", [(-3, 3)]),
    ("4D: Styblinski-Tang", "0.5*(x^4-16*x^2+5*x + y^4-16*y^2+5*y + z^4-16*z^2+5*z + w^4-16*w^2+5*w)", [(-5, 5)]),
    ("4D: Rosenbrock", "(1-x)^2 + 100*(y-x^2)^2 + (1-y)^2 + 100*(z-y^2)^2 + (1-z)^2 + 100*(w-z^2)^2", [(-2, 2)]),
    ("5D: Rastrigin", "50 + " + _indexed_sum("x#^2 - 10*cos(2*pi*x#)", 5), [(-5.12, 5.12)]),
    ("6D: Rosenbrock", _indexed_sum("(1-x#)^2 + 100*(x@-x#^2)^2", 5), [(-2, 2)]),
    ("8D: Styblinski-Tang", "0.5*(" + _indexed_sum("x#^4 - 16*x#^2 + 5*x#", 8) + ")", [(-5, 5)]),
]

BOX_SIZE = 4.0      # size of the longer side of the domain in the 3D scene
BASE_HEIGHT = 2.0   # surface height at z scale = 100 %
LIGHT_DIR = np.array([0.4, 0.3, 0.85]) / np.linalg.norm([0.4, 0.3, 0.85])
LABEL_COLOR = (220, 220, 220, 255)
OPTIMUM_COLOR = (1.0, 1.0, 1.0, 1.0)
CLASSIC_FAMILY = "Classic"
FORMULA_TAB, TEST_TAB, PLUGIN_TAB = 0, 1, 2
MODE_SURFACE, MODE_GRID = "Animated surface", "Slice grid"
SLICE_DEFAULTS = {3: 41, 4: 17}  # slices per extra variable; odd, so the middle of the range is included
MAX_POINTS = 4_000_000           # 3D/4D: the surface resolution is reduced to stay within this budget
MIN_SURFACE_POINTS = 20


def lighting(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Per-vertex Lambert brightness so the relief looks three-dimensional."""
    dzdx, dzdy = np.gradient(zs, xs, ys)
    normals = np.stack([-dzdx, -dzdy, np.ones_like(zs)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return 0.45 + 0.55 * np.clip(normals @ LIGHT_DIR, 0.0, 1.0)


class ColorBar(QtWidgets.QWidget):
    """Vertical scale showing which color corresponds to which function value."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(95)
        self._cmap = None
        self._range = (0.0, 1.0)

    def set_data(self, cmap: pg.ColorMap, vmin: float, vmax: float):
        self._cmap = cmap
        self._range = (vmin, vmax)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        if self._cmap is None:
            return
        top, bottom = 20.0, self.height() - 20.0
        bar = QtCore.QRectF(10, top, 18, bottom - top)

        gradient = QtGui.QLinearGradient(0, bottom, 0, top)
        pos, colors = self._cmap.getStops(mode="byte")
        for t, c in zip(pos, colors):
            gradient.setColorAt(float(t), QtGui.QColor(*(int(v) for v in c)))
        painter.fillRect(bar, QtGui.QBrush(gradient))

        painter.setPen(QtGui.QColor(*LABEL_COLOR))
        painter.drawRect(bar)
        vmin, vmax = self._range
        for i in range(6):
            t = i / 5
            y = bottom - t * (bottom - top)
            painter.drawLine(QtCore.QPointF(28, y), QtCore.QPointF(33, y))
            painter.drawText(
                QtCore.QRectF(36, y - 9, self.width() - 36, 18),
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                f"{vmin + t * (vmax - vmin):.3g}",
            )


class SliceControl(QtWidgets.QWidget):
    """Slider fixing a variable that is not on the surface axes, with a play button."""

    def __init__(self):
        super().__init__()
        self.name = ""
        self.values = np.zeros(1)
        self.label = QtWidgets.QLabel()
        self.label.setMinimumWidth(85)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.play = QtWidgets.QToolButton()
        self.play.setCheckable(True)
        self.set_playing(False)
        self.play.setToolTip("Animate this variable")
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.label)
        row.addWidget(self.slider, 1)
        row.addWidget(self.play)
        self.slider.valueChanged.connect(self._update_label)

    def set_playing(self, playing: bool):
        pixmap = QtWidgets.QStyle.StandardPixmap
        self.play.setIcon(self.style().standardIcon(pixmap.SP_MediaPause if playing else pixmap.SP_MediaPlay))

    def set_axis(self, name: str, values: np.ndarray, index: int):
        self.name, self.values = name, values
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(values) - 1)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self._update_label()

    def _update_label(self):
        self.label.setText(f"{self.name} = {self.value:.4g}")

    @property
    def index(self) -> int:
        return self.slider.value()

    @property
    def value(self) -> float:
        return float(self.values[self.index])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Function Visualization")
        self.resize(1300, 820)
        self._data = None  # result of the last build
        self._slice_dimension = None  # dimension the "Extra dimensions" controls are set up for
        self._playing = None          # SliceControl being animated
        self._direction = 1
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        self._build_controls()
        self._build_views()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._controls)
        splitter.addWidget(self._view_area)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._formula_ranges = None  # ranges of the "Formula" tab while test problems are open
        if self._classic_catalog:
            self._on_family_changed()
        if self._plugin_catalog:
            self._on_plugin_changed()
        self.preset_box.setCurrentIndex(0)
        self._apply_preset(0)

    # ---------- UI ----------

    def _build_controls(self):
        self._controls = QtWidgets.QWidget()
        self._controls.setMinimumWidth(340)
        self._controls.setMaximumWidth(440)
        layout = QtWidgets.QVBoxLayout(self._controls)

        self.source_tabs = QtWidgets.QTabWidget()
        self.source_tabs.addTab(self._build_formula_page(), "Formula")
        self.source_tabs.addTab(self._build_test_page(), "Test problems")
        self.source_tabs.addTab(self._build_plugin_page(), "Plugins")
        self._source_tab = FORMULA_TAB
        self.source_tabs.currentChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_tabs)

        form = QtWidgets.QFormLayout()
        self._range_form = form
        self.range_spins = [(self._range_spin(), self._range_spin()) for _ in VARIABLES]
        self._range_rows = [self._pair(*spins) for spins in self.range_spins]
        for name, row in zip(VARIABLES, self._range_rows):
            form.addRow(f"{name} from / to:", row)
            form.setRowVisible(row, name in "xy")

        self.resolution = QtWidgets.QSpinBox()
        self.resolution.setRange(20, 600)
        self.resolution.setValue(200)
        self.resolution.setToolTip("Number of grid points along each axis")
        form.addRow("Points per axis:", self.resolution)

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(COLORMAP_NAMES)
        self.cmap_box.currentIndexChanged.connect(self._redraw)
        form.addRow("Colormap:", self.cmap_box)

        self.z_scale = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_scale.setRange(5, 300)
        self.z_scale.setValue(100)
        self.z_scale.valueChanged.connect(self._redraw)
        form.addRow("Z scale:", self.z_scale)

        self.clip_outliers = QtWidgets.QCheckBox("Clip outliers (1–99 %)")
        self.clip_outliers.setToolTip("Useful when the function has sharp spikes, e.g. 1/x")
        self.clip_outliers.toggled.connect(self._redraw)
        form.addRow(self.clip_outliers)

        self.show_optimum = QtWidgets.QCheckBox("Mark global minimum")
        self.show_optimum.setChecked(True)
        self.show_optimum.setToolTip("White dot with a flag: the known minimum of the test problem")
        self.show_optimum.toggled.connect(self._redraw)
        form.addRow(self.show_optimum)
        layout.addLayout(form)
        layout.addWidget(self._build_slice_group())
        layout.addWidget(self._build_hyperslice_group())

        build_button = QtWidgets.QPushButton("Plot")
        build_button.setDefault(True)
        build_button.clicked.connect(self.build)
        reset_button = QtWidgets.QPushButton("Reset view")
        reset_button.clicked.connect(self.reset_view)
        layout.addWidget(build_button)
        layout.addWidget(reset_button)

        help_label = QtWidgets.QLabel(
            "3D: left mouse button rotates, wheel zooms, "
            "middle button or Ctrl + left button pans."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray;")
        layout.addWidget(help_label)
        layout.addStretch(1)

    def _build_slice_group(self) -> QtWidgets.QWidget:
        """Controls of the variables that do not fit on the surface (3D and 4D functions)."""
        self.slice_group = QtWidgets.QGroupBox("Extra dimensions")
        form = QtWidgets.QFormLayout(self.slice_group)
        self._slice_form = form

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems([MODE_SURFACE, MODE_GRID])
        self.mode_box.setToolTip(
            "Animated surface: one slice at a time.\n"
            "Slice grid: all slices at once as heatmaps (dimensional stacking)."
        )
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("View:", self.mode_box)

        self.axes_box = QtWidgets.QComboBox()
        self.axes_box.setToolTip("Variables on the surface axes; the others become sliders")
        self.axes_box.activated.connect(self.build)
        form.addRow("Surface axes:", self.axes_box)

        self.slices_spin = QtWidgets.QSpinBox()
        self.slices_spin.setRange(3, 121)
        self.slices_spin.setToolTip("How many values each extra variable takes; applied by Plot")
        form.addRow("Slices per variable:", self.slices_spin)

        self.slice_controls = [SliceControl(), SliceControl()]
        for control in self.slice_controls:
            control.slider.valueChanged.connect(self._show_slice)
            control.play.toggled.connect(lambda playing, c=control: self._on_play_toggled(c, playing))
            form.addRow(control)

        self.speed_spin = QtWidgets.QSpinBox()
        self.speed_spin.setRange(1, 60)
        self.speed_spin.setValue(12)
        self.speed_spin.setSuffix(" slices/s")
        self.speed_spin.valueChanged.connect(lambda value: self._timer.setInterval(1000 // value))
        self._timer.setInterval(1000 // self.speed_spin.value())
        form.addRow("Speed:", self.speed_spin)

        self.slice_group.setVisible(False)
        return self.slice_group

    def _build_hyperslice_group(self) -> QtWidgets.QWidget:
        """Focus point shortcuts of the HyperSlice view (N > 4)."""
        self.hyperslice_group = QtWidgets.QGroupBox("HyperSlice focus point")
        row = QtWidgets.QHBoxLayout(self.hyperslice_group)
        self.to_minimum_button = QtWidgets.QPushButton("To minimum")
        self.to_minimum_button.setToolTip("Move the focus point to the known global minimum")
        self.to_minimum_button.clicked.connect(lambda: self.hyperslice.focus_start())
        center_button = QtWidgets.QPushButton("To center")
        center_button.setToolTip("Move the focus point to the center of the domain")
        center_button.clicked.connect(lambda: self.hyperslice.focus_center())
        row.addWidget(self.to_minimum_button)
        row.addWidget(center_button)
        self.hyperslice_group.setVisible(False)
        return self.hyperslice_group

    def _build_formula_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.preset_box = QtWidgets.QComboBox()
        self.preset_box.addItems([p[0] for p in PRESETS])
        self.preset_box.activated.connect(self._apply_preset)
        form.addRow("Example:", self.preset_box)

        self.formula_edit = QtWidgets.QLineEdit()
        self.formula_edit.setPlaceholderText("e.g. sin(x)*cos(y) + z^2")
        self.formula_edit.returnPressed.connect(self.build)
        form.addRow("f =", self.formula_edit)

        help_label = QtWidgets.QLabel(HELP_TEXT + "\nThe dimension is set by the variables used: x → 1D, y → 2D, z → 3D, w → 4D.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray;")
        form.addRow(help_label)
        return page

    def _build_test_page(self) -> QtWidgets.QWidget:
        """Problems from global-opt: GCGen families and classic functions."""
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self._test_form = form

        available, error = tf.is_available()
        if not available:
            label = QtWidgets.QLabel(error)
            label.setWordWrap(True)
            form.addRow(label)
            self._classic_catalog = []
            return page
        self._classic_catalog = tf.classic_catalog()

        self.family_box = QtWidgets.QComboBox()
        for family in tf.GCGEN_FAMILIES:
            dimensions = family.dimensions
            label = f"{dimensions[0]}D" if len(dimensions) == 1 else f"{dimensions[0]}–{dimensions[-1]}D"
            self.family_box.addItem(f"{family.name} ({label}, {family.count} problems)", family)
        self.family_box.addItem(f"{CLASSIC_FAMILY} ({len(self._classic_catalog)} functions)", CLASSIC_FAMILY)
        form.addRow("Family:", self.family_box)

        self.number_spin = QtWidgets.QSpinBox()
        form.addRow("Problem number:", self.number_spin)

        self.gkls_class_box = QtWidgets.QComboBox()
        self.gkls_class_box.addItems(tf.GKLS_CLASSES)
        form.addRow("GKLS class:", self.gkls_class_box)
        self.gkls_type_box = QtWidgets.QComboBox()
        self.gkls_type_box.addItems(tf.GKLS_TYPES)
        self.gkls_type_box.setToolTip("ND: non-differentiable, D: differentiable, D2: twice differentiable")
        form.addRow("GKLS type:", self.gkls_type_box)
        self.gkls_dim = QtWidgets.QSpinBox()
        gkls_dimensions = next(f.dimensions for f in tf.GCGEN_FAMILIES if f.name == "GKLS")
        self.gkls_dim.setRange(gkls_dimensions[0], gkls_dimensions[-1])
        form.addRow("GKLS dimension:", self.gkls_dim)

        self.classic_box = QtWidgets.QComboBox()
        for info in self._classic_catalog:
            self.classic_box.addItem(info.name, info)
        form.addRow("Function:", self.classic_box)
        self.classic_dim = QtWidgets.QSpinBox()
        self.classic_dim.setRange(1, 2)
        self.classic_dim.setValue(2)
        form.addRow("Dimension:", self.classic_dim)

        self.problem_info = QtWidgets.QLabel()
        self.problem_info.setWordWrap(True)
        self.problem_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.problem_info)

        self.family_box.currentIndexChanged.connect(self._on_family_changed)
        self.classic_box.currentIndexChanged.connect(self._on_classic_changed)
        for spin in (self.number_spin, self.classic_dim, self.gkls_dim):
            spin.valueChanged.connect(self._load_problem)
        for box in (self.gkls_class_box, self.gkls_type_box):
            box.currentIndexChanged.connect(self._load_problem)
        return page

    def _build_plugin_page(self) -> QtWidgets.QWidget:
        """Functions written in Python or C++ in the plugins folder."""
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self._plugin_catalog, errors = plugin_loader.load_catalog()
        message = "\n".join(errors)
        if message:
            label = QtWidgets.QLabel(message)
            label.setWordWrap(True)
            label.setStyleSheet("color: #d33;")
            form.addRow(label)
        if not self._plugin_catalog:
            self._plugin_error = message or "No plugins found. Add a .py file to the plugins folder and restart."
            if not message:
                form.addRow(QtWidgets.QLabel(self._plugin_error))
            return page
        self._plugin_error = ""

        self.plugin_box = QtWidgets.QComboBox()
        for info in self._plugin_catalog:
            self.plugin_box.addItem(info.name, info)
        form.addRow("Function:", self.plugin_box)
        self.plugin_dim = QtWidgets.QSpinBox()
        form.addRow("Dimension:", self.plugin_dim)
        self.plugin_info = QtWidgets.QLabel()
        self.plugin_info.setWordWrap(True)
        self.plugin_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.plugin_info)

        self.plugin_box.currentIndexChanged.connect(self._on_plugin_changed)
        self.plugin_dim.valueChanged.connect(self._load_problem)
        return page

    @staticmethod
    def _range_spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(4)
        spin.setSingleStep(0.5)
        return spin

    @staticmethod
    def _pair(a: QtWidgets.QWidget, b: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(a)
        row.addWidget(b)
        return box

    def _build_views(self):
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor((30, 30, 30))
        self._scene_items = []
        self._marker_items = []

        self.plot_widget = pg.PlotWidget(background=(30, 30, 30))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("bottom", "x")
        self.plot_widget.setLabel("left", "f(x)")

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.gl_view)
        self.stack.addWidget(self.plot_widget)

        self.grid_widget = pg.PlotWidget(background=(30, 30, 30))
        self.grid_widget.setAspectLocked(True)
        self.grid_widget.scene().sigMouseMoved.connect(self._on_grid_hover)
        self.stack.addWidget(self.grid_widget)

        self.hyperslice = HyperSliceView()
        self.hyperslice.statusChanged.connect(self.statusBar().showMessage)
        self.hyperslice.rangeChanged.connect(self._on_hyperslice_range)
        self.stack.addWidget(self.hyperslice)
        self._tiles = None  # layout of the last drawn slice grid
        self.color_bar = ColorBar()

        self._view_area = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self._view_area)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.stack, 1)
        row.addWidget(self.color_bar)

    def _apply_preset(self, index: int):
        _, formula, ranges = PRESETS[index]
        self.formula_edit.setText(formula)
        self._set_ranges(ranges)
        self.build()

    def _set_ranges(self, ranges):
        """Sets the ranges of the variables; the variables left out get the range of x."""
        ranges = [*ranges, *[ranges[0]] * (len(VARIABLES) - len(ranges))]
        for spins, values in zip(self.range_spins, ranges):
            for spin, value in zip(spins, values):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

    def _ranges(self):
        return [(lo.value(), hi.value()) for lo, hi in self.range_spins]

    def _test_mode(self) -> bool:
        return self.source_tabs.currentIndex() != FORMULA_TAB

    def _source_error(self) -> str:
        """Why the open problem tab cannot be used; empty if it can."""
        if self.source_tabs.currentIndex() == PLUGIN_TAB:
            return self._plugin_error
        return "" if self._classic_catalog else tf.is_available()[1]

    def _on_source_changed(self, index: int):
        # Each tab has its own ranges: restore the formula ranges when switching back.
        if index == FORMULA_TAB:
            if self._formula_ranges is not None:
                self._set_ranges(self._formula_ranges)
            self.build()
        else:
            if self._source_tab == FORMULA_TAB:
                self._formula_ranges = self._ranges()
            if not self._source_error():
                self._load_problem()
        self._source_tab = index

    def _on_family_changed(self):
        family = self.family_box.currentData()
        classic = family == CLASSIC_FAMILY
        gkls = not classic and family.name == "GKLS"
        form = self._test_form
        form.setRowVisible(self.number_spin, not classic)
        form.setRowVisible(self.gkls_class_box, gkls)
        form.setRowVisible(self.gkls_type_box, gkls)
        form.setRowVisible(self.gkls_dim, gkls)
        form.setRowVisible(self.classic_box, classic)
        form.setRowVisible(self.classic_dim, classic)
        if not classic:
            self.number_spin.blockSignals(True)
            self.number_spin.setRange(1, family.count)
            self.number_spin.blockSignals(False)
        self._on_classic_changed()

    @staticmethod
    def _set_dimension_choices(spin: QtWidgets.QSpinBox, dimensions: tuple):
        spin.blockSignals(True)
        spin.setRange(min(dimensions), max(dimensions))
        # e.g. powell exists only for multiples of 4
        spin.setSingleStep(dimensions[1] - dimensions[0] if len(dimensions) > 1 else 1)
        spin.setValue(2 if 2 in dimensions else min(dimensions))
        spin.blockSignals(False)
        spin.setEnabled(len(dimensions) > 1)

    def _on_classic_changed(self):
        self._set_dimension_choices(self.classic_dim, self.classic_box.currentData().dimensions)
        self._load_problem()

    def _on_plugin_changed(self):
        self._set_dimension_choices(self.plugin_dim, self.plugin_box.currentData().dimensions)
        self._load_problem()

    def _current_problem(self) -> tf.TestProblem:
        if self.source_tabs.currentIndex() == PLUGIN_TAB:
            return self.plugin_box.currentData().create(self.plugin_dim.value())
        family = self.family_box.currentData()
        if family == CLASSIC_FAMILY:
            return tf.classic_problem(self.classic_box.currentText(), self.classic_dim.value())
        return tf.gcgen_problem(
            family, self.number_spin.value(), self.gkls_class_box.currentText(), self.gkls_type_box.currentText(),
            self.gkls_dim.value(),
        )

    def _load_problem(self):
        """Another test problem was selected: apply its domain and plot it."""
        if not self._test_mode():
            return
        try:
            problem = self._current_problem()
        except ValueError as exc:
            self.statusBar().showMessage(f"Failed to create the problem: {exc}")
            return
        lo, hi = problem.lower, problem.upper
        self._set_ranges(list(zip(lo, hi)))

        bounds = " × ".join(f"[{a:g}, {b:g}]" for a, b in zip(lo, hi))
        text = f"Domain: {bounds}"
        if problem.optimum_point is not None:
            point = ", ".join(f"{v:.6g}" for v in problem.optimum_point)
            text += f"\nGlobal minimum: f* = {problem.optimum_value:.6g}\nat x* = ({point})"
        else:
            text += "\nGlobal minimum is unknown"
        info_label = self.plugin_info if self.source_tabs.currentIndex() == PLUGIN_TAB else self.problem_info
        info_label.setText(text)
        self.build()

    # ---------- computation ----------

    def build(self):
        try:
            if self._test_mode():
                if self._source_error():
                    raise ValueError(self._source_error())
                problem = self._current_problem()
                dimension, title = problem.dimension, problem.title
                optimum = None
                if problem.optimum_point is not None:
                    optimum = (problem.optimum_point, problem.optimum_value)

                def evaluate(*axes):
                    return problem.evaluate(np.stack(axes, axis=-1)).reshape(axes[0].shape)
            else:
                formula = Formula(self.formula_edit.text())
                dimension, evaluate, optimum = formula.dimension, formula.evaluate, None
                names = variable_names(dimension)
                title = f"f({', '.join(names)})" if dimension <= SLICED_DIMENSIONS else f"f({names[0]}, …, {names[-1]})"

            # Above 4D all variables share one range: all test problems of such dimensions have equal bounds.
            ranges = self._ranges()[:dimension] if dimension <= SLICED_DIMENSIONS else [self._ranges()[0]] * dimension
            if any(lo >= hi for lo, hi in ranges):
                raise FormulaError("The lower bound of a range must be less than the upper bound")

            self._stop_animation()
            if dimension > SLICED_DIMENSIONS:
                self._build_hyperslice(dimension, title, evaluate, ranges, optimum)
                return
            self._setup_slice_controls(dimension)
            plot_axes = self.axes_box.currentData() if dimension > 2 else tuple(range(dimension))
            slice_axes = [i for i in range(dimension) if i not in plot_axes]
            n, slices = self._grid_sizes(dimension)
            axes = [
                np.linspace(lo, hi, n) if i in plot_axes else self._slice_values(lo, hi, slices, i, optimum)
                for i, (lo, hi) in enumerate(ranges)
            ]
            # z[i, j, ...] = f(axes[0][i], axes[1][j], ...)
            z = evaluate(*np.meshgrid(*axes, indexing="ij"))
        except ValueError as exc:  # FormulaError is a ValueError too
            if not self._test_mode():
                self.formula_edit.setStyleSheet("border: 1px solid #d33;")
            self.statusBar().showMessage(str(exc))
            return

        self.formula_edit.setStyleSheet("")
        bad = ~np.isfinite(z)
        if bad.all():
            self.statusBar().showMessage("The function is undefined at every point of the range")
            return
        # Points outside the domain (log(-1), 1/0) are replaced by the minimum to keep the mesh intact.
        z[bad] = z[~bad].min()

        self._update_range_rows(dimension)
        self._data = {
            "dimension": dimension, "title": title, "optimum": optimum, "names": VARIABLES[:dimension],
            "axes": axes, "plot_axes": plot_axes, "slice_axes": slice_axes, "z": z, "bad": int(bad.sum()),
            "note": f"surface reduced to {n}×{n} points" if dimension > 2 and n < self.resolution.value() else "",
        }
        for control, axis in zip(self.slice_controls, slice_axes):
            start = len(axes[axis]) // 2
            if optimum is not None and ranges[axis][0] <= optimum[0][axis] <= ranges[axis][1]:
                start = int(np.abs(axes[axis] - optimum[0][axis]).argmin())  # open on the slice with the minimum
            control.set_axis(VARIABLES[axis], axes[axis], start)
        self._update_slice_rows()
        self._redraw()
        self.reset_view()

    def _build_hyperslice(self, dimension: int, title: str, evaluate, ranges, optimum):
        # Raises ValueError for a bad function: the current view stays as it is.
        self.hyperslice.set_function(
            lambda points: evaluate(*points.T), title, variable_names(dimension), ranges, optimum,
            self.resolution.value(),
        )
        self._setup_slice_controls(dimension)
        self.stack.setCurrentWidget(self.hyperslice)
        self.formula_edit.setStyleSheet("")
        self._update_range_rows(dimension)
        self._data = {"dimension": dimension, "title": title, "optimum": optimum, "slice_axes": []}
        self._update_slice_rows()
        self.to_minimum_button.setEnabled(self.hyperslice.has_optimum())
        self._redraw()

    def _on_hyperslice_range(self, vmin: float, vmax: float):
        self.color_bar.set_data(get_colormap(self.cmap_box.currentText()), vmin, vmax)

    def _update_range_rows(self, dimension: int):
        shared = dimension > SLICED_DIMENSIONS
        label = self._range_form.labelForField(self._range_rows[0])
        label.setText(f"x1 … x{dimension} from / to:" if shared else "x from / to:")
        for i, row in enumerate(self._range_rows):
            row.setEnabled(i < dimension)
            if i >= 1:
                self._range_form.setRowVisible(row, not shared and (i == 1 or i < dimension))

    def _setup_slice_controls(self, dimension: int):
        if dimension == self._slice_dimension:
            return
        self._slice_dimension = dimension
        sliced = 2 < dimension <= SLICED_DIMENSIONS
        self.slice_group.setVisible(sliced)
        self.hyperslice_group.setVisible(dimension > SLICED_DIMENSIONS)
        if sliced:
            self.axes_box.clear()
            for a, b in itertools.combinations(range(dimension), 2):
                self.axes_box.addItem(f"{VARIABLES[a]}, {VARIABLES[b]}", (a, b))
            self.slices_spin.setValue(SLICE_DEFAULTS[dimension])

    def _grid_sizes(self, dimension: int) -> tuple[int, int]:
        """(points along each plotted axis, values of each extra variable)."""
        n = self.resolution.value()
        if dimension == 1:
            return n * 10, 0
        if dimension == 2:
            return n, 0
        slices = self.slices_spin.value()
        affordable = int(math.sqrt(MAX_POINTS / slices ** (dimension - 2)))
        return max(min(n, affordable), MIN_SURFACE_POINTS), slices

    @staticmethod
    def _slice_values(lo: float, hi: float, count: int, axis: int, optimum) -> np.ndarray:
        """Evenly spaced slices; the nearest one is moved onto the known minimum so that a slice contains it."""
        values = np.linspace(lo, hi, count)
        if optimum is not None and lo <= optimum[0][axis] <= hi:
            values[np.abs(values - optimum[0][axis]).argmin()] = optimum[0][axis]
        return values

    def _grid_mode(self) -> bool:
        return (
            self._data is not None
            and 2 < self._data["dimension"] <= SLICED_DIMENSIONS
            and self.mode_box.currentText() == MODE_GRID
        )

    def _update_slice_rows(self):
        grid = self._grid_mode()
        count = len(self._data["slice_axes"]) if self._data else 0
        for i, control in enumerate(self.slice_controls):
            self._slice_form.setRowVisible(control, not grid and i < count)
        self._slice_form.setRowVisible(self.speed_spin, not grid)
        self.z_scale.setEnabled(self._data is not None and 1 < self._data["dimension"] <= SLICED_DIMENSIONS and not grid)

    def _on_mode_changed(self):
        self._stop_animation()
        self._update_slice_rows()
        self._redraw()
        self.reset_view()

    # ---------- animation ----------

    def _on_play_toggled(self, control: SliceControl, playing: bool):
        control.set_playing(playing)
        if playing:
            for other in self.slice_controls:
                if other is not control:
                    other.play.setChecked(False)
            self._playing = control
            self._direction = 1
            self._timer.start()
        elif self._playing is control:
            self._playing = None
            self._timer.stop()

    def _stop_animation(self):
        for control in self.slice_controls:
            control.play.setChecked(False)

    def _next_frame(self):
        """Moves the animated slider back and forth over its range."""
        slider = self._playing.slider
        index = slider.value() + self._direction
        if not 0 <= index <= slider.maximum():
            self._direction = -self._direction
            index = slider.value() + self._direction
        slider.setValue(index)

    def _slice_key(self, indices) -> tuple:
        """Index of the 2D slice of z with the extra variables fixed at the given positions."""
        key = [slice(None)] * self._data["dimension"]
        for axis, index in zip(self._data["slice_axes"], indices):
            key[axis] = index
        return tuple(key)

    def _current_indices(self) -> list[int]:
        return [control.index for control, _ in zip(self.slice_controls, self._data["slice_axes"])]

    def _show_slice(self):
        """A slider moved: update only the surface, keeping the camera and the scene."""
        if self._data is None or self._data["dimension"] < 3 or self._grid_mode():
            return
        self._update_surface()
        self._report()

    def _value_range(self, z: np.ndarray) -> tuple[np.ndarray, float, float]:
        if self.clip_outliers.isChecked():
            lo, hi = np.percentile(z, [1, 99])
            z = np.clip(z, lo, hi)
        zmin, zmax = float(z.min()), float(z.max())
        if zmax - zmin < 1e-12:  # constant function
            zmin, zmax = zmin - 1.0, zmax + 1.0
        return z, zmin, zmax

    # ---------- drawing ----------

    def _redraw(self):
        if self._data is None:
            return
        cmap = get_colormap(self.cmap_box.currentText())
        if self._data["dimension"] > SLICED_DIMENSIONS:
            self.hyperslice.set_appearance(cmap, self.clip_outliers.isChecked(), self.show_optimum.isChecked())
            return
        # The color and height scales are shared by all slices, so slices can be compared.
        z, zmin, zmax = self._value_range(self._data["z"])
        self._cmap, self._shown, self._span = cmap, z, (zmin, zmax)
        if self._data["dimension"] == 1:
            self._draw_curve(cmap, z, zmin, zmax)
            self.stack.setCurrentWidget(self.plot_widget)
        elif self._grid_mode():
            self._draw_grid(cmap, z, zmin, zmax)
            self.stack.setCurrentWidget(self.grid_widget)
        else:
            self._draw_surface()
            self.stack.setCurrentWidget(self.gl_view)
        self.color_bar.set_data(cmap, zmin, zmax)
        self._report()

    def _report(self):
        d = self._data
        zmin, zmax = self._span
        message = f"{d['title']}: values from {zmin:.6g} to {zmax:.6g}"
        if d["dimension"] > 2 and not self._grid_mode():
            indices = self._current_indices()
            current = self._shown[self._slice_key(indices)]
            fixed = ", ".join(c.label.text() for c, _ in zip(self.slice_controls, indices))
            message += f"; at {fixed}: from {current.min():.4g} to {current.max():.4g}"
        if d["bad"]:
            message += f"; undefined at {d['bad']} points (colored as the minimum)"
        if d["note"]:
            message += f"; {d['note']}"
        self.statusBar().showMessage(message)

    def _draw_curve(self, cmap: pg.ColorMap, z, zmin, zmax):
        fill = get_colormap(self.cmap_box.currentText(), alpha=90)
        self.plot_widget.clear()
        self.plot_widget.plot(
            self._data["axes"][0], z,
            pen=cmap.getPen(span=(zmin, zmax), width=3),
            fillLevel=zmin,
            fillBrush=fill.getBrush(span=(zmin, zmax)),
        )
        optimum = self._visible_optimum()
        if optimum is not None:
            (x_opt,), f_opt = optimum
            self.plot_widget.plot(
                [x_opt], [f_opt], pen=None, symbol="star", symbolSize=18,
                symbolBrush="w", symbolPen=pg.mkPen("k", width=1),
            )

    def _surface_mesh(self):
        """Heights and vertex colors of the current slice."""
        xs, ys = self._surface_xy
        zmin, zmax = self._span
        height = BASE_HEIGHT * self.z_scale.value() / 100
        t = (self._shown[self._slice_key(self._current_indices())] - zmin) / (zmax - zmin)
        zs = t * height
        colors = self._cmap.map(t.ravel(), mode="float").reshape(*t.shape, 4)
        colors[..., :3] *= lighting(xs, ys, zs)[..., None]
        return zs, colors

    def _draw_surface(self):
        d = self._data
        a, b = d["plot_axes"]
        x, y = d["axes"][a], d["axes"][b]
        self._surface_mid = (x[0] + x[-1]) / 2, (y[0] + y[-1]) / 2
        self._surface_scale = BOX_SIZE / max(x[-1] - x[0], y[-1] - y[0])  # keep the x/y aspect ratio
        xs, ys = (x - self._surface_mid[0]) * self._surface_scale, (y - self._surface_mid[1]) * self._surface_scale
        self._surface_xy = xs, ys
        height = BASE_HEIGHT * self.z_scale.value() / 100

        zs, colors = self._surface_mesh()
        for item in (*self._scene_items, *self._marker_items):
            self.gl_view.removeItem(item)
        # Lighting is baked into the colors, so normals are not needed; this keeps the animation fast.
        self._surface = gl.GLSurfacePlotItem(
            x=xs, y=ys, z=zs, colors=colors, smooth=True, computeNormals=False, glOptions="opaque"
        )
        names = d["names"][a], d["names"][b]
        self._scene_items = [self._surface, *self._axes(x, y, xs, ys, names, *self._span, height)]
        self._marker_items = []
        self._marker_shown = False
        for item in self._scene_items:
            self.gl_view.addItem(item)
        self._update_marker()

    def _update_surface(self):
        zs, colors = self._surface_mesh()
        self._surface.setData(z=zs, colors=colors)
        self._update_marker()

    def _update_marker(self):
        """The minimum is marked only on the slice that contains it."""
        optimum = self._visible_optimum()
        visible = optimum is not None and all(
            np.isclose(self._data["axes"][axis][index], optimum[0][axis])
            for axis, index in zip(self._data["slice_axes"], self._current_indices())
        )
        if visible == self._marker_shown:
            return
        self._marker_shown = visible
        for item in self._marker_items:
            self.gl_view.removeItem(item)
        self._marker_items = []
        if visible:
            point, f_opt = optimum
            a, b = self._data["plot_axes"]
            zmin, zmax = self._span
            height = BASE_HEIGHT * self.z_scale.value() / 100
            base = np.array([(point[a] - self._surface_mid[0]) * self._surface_scale,
                             (point[b] - self._surface_mid[1]) * self._surface_scale,
                             np.clip((f_opt - zmin) / (zmax - zmin), 0, 1) * height])
            self._marker_items = self._optimum_marker(base, height, f_opt)
            for item in self._marker_items:
                self.gl_view.addItem(item)

    def _draw_grid(self, cmap: pg.ColorMap, z, zmin, zmax):
        """Dimensional stacking: heatmaps of the plotted axes nested in a grid of the extra variables."""
        d = self._data
        axes, names = d["axes"], d["names"]
        a, b = d["plot_axes"]
        slice_axes = d["slice_axes"]
        nx, ny = len(axes[a]), len(axes[b])
        gap = max(1, min(nx, ny) // 25)
        if len(slice_axes) == 1:
            count = len(axes[slice_axes[0]])
            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
        else:
            cols, rows = len(axes[slice_axes[0]]), len(axes[slice_axes[1]])
        self._tiles = nx, ny, gap, cols, rows

        lut = cmap.getLookupTable(nPts=1024, alpha=True, mode="byte")
        colors = lut[((z - zmin) / (zmax - zmin) * (len(lut) - 1)).astype(int)]
        image = np.zeros((cols * (nx + gap) - gap, rows * (ny + gap) - gap, 4), dtype=np.uint8)

        optimum = self._visible_optimum()
        optimum_tile = None
        if optimum is not None:
            optimum_tile = tuple(int(np.abs(axes[s] - optimum[0][s]).argmin()) for s in slice_axes)

        widget = self.grid_widget
        widget.clear()
        star = None
        for col, row in itertools.product(range(cols), range(rows)):
            indices = self._tile_indices(col, row)
            if indices is None:
                continue
            x0, y0 = col * (nx + gap), row * (ny + gap)
            image[x0:x0 + nx, y0:y0 + ny] = colors[self._slice_key(indices)]
            if len(slice_axes) == 1:
                text = pg.TextItem(
                    f"{names[slice_axes[0]]} = {axes[slice_axes[0]][indices[0]]:.3g}",
                    color=LABEL_COLOR, fill=(30, 30, 30, 170),
                )
                text.setPos(x0, y0 + ny)
                widget.addItem(text)
            if indices == optimum_tile:
                point = optimum[0]
                star = (x0 + (point[a] - axes[a][0]) / (axes[a][-1] - axes[a][0]) * (nx - 1) + 0.5,
                        y0 + (point[b] - axes[b][0]) / (axes[b][-1] - axes[b][0]) * (ny - 1) + 0.5)
        item = pg.ImageItem(image, axisOrder="col-major")
        item.setZValue(-1)
        widget.addItem(item)
        if star is not None:
            widget.plot([star[0]], [star[1]], pen=None, symbol="star", symbolSize=16,
                        symbolBrush="w", symbolPen=pg.mkPen("k", width=1))

        bottom, left = widget.getAxis("bottom"), widget.getAxis("left")
        if len(slice_axes) == 2:
            for axis, values, step, name in (
                (bottom, axes[slice_axes[0]], nx + gap, names[slice_axes[0]]),
                (left, axes[slice_axes[1]], ny + gap, names[slice_axes[1]]),
            ):
                size = nx if axis is bottom else ny
                axis.setTicks([[(i * step + size / 2, f"{v:.3g}") for i, v in enumerate(values)]])
                axis.setLabel(name)
        else:
            for axis in (bottom, left):
                axis.setTicks([[]])
                axis.setLabel("")
        widget.setTitle(
            f"inside each tile: {names[a]} → [{axes[a][0]:.3g}, {axes[a][-1]:.3g}], "
            f"{names[b]} ↑ [{axes[b][0]:.3g}, {axes[b][-1]:.3g}]",
            color=LABEL_COLOR[:3], size="10pt",
        )

    def _tile_indices(self, col: int, row: int):
        """Positions of the extra variables in a grid tile, None if the tile is empty."""
        slice_axes = self._data["slice_axes"]
        if len(slice_axes) == 2:
            return col, row
        _, _, _, cols, rows = self._tiles
        index = (rows - 1 - row) * cols + col  # first slice at the top left, reading order
        return (index,) if index < len(self._data["axes"][slice_axes[0]]) else None

    def _on_grid_hover(self, position):
        """Shows the coordinates and the function value under the cursor in the slice grid."""
        if not self._grid_mode() or self._tiles is None:
            return
        nx, ny, gap, cols, rows = self._tiles
        point = self.grid_widget.getViewBox().mapSceneToView(position)
        col, i = divmod(int(math.floor(point.x())), nx + gap)
        row, j = divmod(int(math.floor(point.y())), ny + gap)
        if not (0 <= col < cols and 0 <= row < rows and i < nx and j < ny):
            return
        indices = self._tile_indices(col, row)
        if indices is None:
            return
        d = self._data
        a, b = d["plot_axes"]
        coordinates = {a: d["axes"][a][i], b: d["axes"][b][j]}
        coordinates.update((s, d["axes"][s][k]) for s, k in zip(d["slice_axes"], indices))
        text = ", ".join(f"{name} = {coordinates[n]:.4g}" for n, name in enumerate(d["names"]))
        self.statusBar().showMessage(f"{text}: f = {d['z'][self._slice_key(indices)][i, j]:.6g}")

    def _visible_optimum(self):
        """The known minimum if it should be shown and lies within the plotted ranges."""
        optimum = self._data["optimum"]
        if optimum is None or not self.show_optimum.isChecked():
            return None
        point, _ = optimum
        if all(axis[0] <= v <= axis[-1] for v, axis in zip(point, self._data["axes"])):
            return optimum
        return None

    @staticmethod
    def _optimum_marker(base: np.ndarray, height: float, value: float):
        """White dot at the minimum with a flag above it, visible even behind hills."""
        top = base + [0, 0, height * 0.35 + 0.3]
        point = gl.GLScatterPlotItem(pos=base[None], color=OPTIMUM_COLOR, size=14, pxMode=True)
        pole = gl.GLLinePlotItem(pos=np.array([base, top]), color=OPTIMUM_COLOR, width=2, glOptions="additive")
        label = gl.GLTextItem(pos=top, text=f" min f = {value:.4g}", color=(255, 255, 255, 255))
        return [point, pole, label]

    @staticmethod
    def _axes(x, y, xs, ys, names, zmin, zmax, height):
        """Floor grid, z axis in the corner and range labels."""
        grid = gl.GLGridItem()
        grid.setSize(xs[-1] - xs[0], ys[-1] - ys[0])
        grid.setSpacing((xs[-1] - xs[0]) / 10, (ys[-1] - ys[0]) / 10)
        grid.setColor((255, 255, 255, 50))
        grid.translate(0, 0, -0.002)

        x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
        z_axis = gl.GLLinePlotItem(pos=np.array([[x0, y1, 0], [x0, y1, height]]), color=LABEL_COLOR, width=1)
        pad = 0.25
        labels = [
            ((0, y0 - pad, 0), f"{names[0]}: {x[0]:.3g} … {x[-1]:.3g}"),
            ((x1 + pad, 0, 0), f"{names[1]}: {y[0]:.3g} … {y[-1]:.3g}"),
            ((x0, y1, 0), f"  f = {zmin:.3g}"),
            ((x0, y1, height), f"  f = {zmax:.3g}"),
        ]
        texts = [gl.GLTextItem(pos=pos, text=text, color=LABEL_COLOR) for pos, text in labels]
        return [grid, z_axis, *texts]

    def reset_view(self):
        if self._data is None:
            return
        if self._data["dimension"] > SLICED_DIMENSIONS:
            self.hyperslice.focus_start()
        elif self._grid_mode():
            self.grid_widget.autoRange(padding=0.02)
        elif self._data["dimension"] >= 2:
            height = BASE_HEIGHT * self.z_scale.value() / 100
            self.gl_view.setCameraPosition(
                pos=QtGui.QVector3D(0, 0, height / 2), distance=BOX_SIZE * 1.6, elevation=28, azimuth=-55
            )
        else:
            self.plot_widget.autoRange()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
