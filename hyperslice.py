"""HyperSlice view (van Wijk, van Liere, 1993) for functions of many variables.

All panels are slices through one common focus point:
  - below the diagonal, panel (i, j) is a heatmap of f over the variables x_i, x_j
    with all the other variables fixed at the focus point;
  - on the diagonal, panel (i, i) is the graph of f along x_i alone.
Dragging inside a panel moves the focus point, and every panel is recomputed. Only the
visible slices are evaluated, so the cost grows as N^2 rather than exponentially with N.
"""
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

SAMPLE_SIZE = 100_000     # random points used to estimate the color scale
MIN_RESOLUTION = 16       # points along a panel side
MAX_RESOLUTION = 128
LEFT, TOP, RIGHT, BOTTOM = 34, 8, 8, 24  # margins around the matrix, px
GAP = 3                   # distance between panels, px
LUT_SIZE = 1024
BACKGROUND = QtGui.QColor(30, 30, 30)
LABEL_COLOR = QtGui.QColor(220, 220, 220)
FRAME_COLOR = QtGui.QColor(90, 90, 90)
FOCUS_COLOR = QtGui.QColor(255, 255, 255, 170)
CURVE_COLOR = QtGui.QColor(235, 235, 235)


class HyperSliceView(QtWidgets.QWidget):
    statusChanged = QtCore.pyqtSignal(str)
    rangeChanged = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self._evaluate = None
        self._names = ()
        self._images = {}
        self._values = {}
        self._drag = None
        self._hover_active = False
        self._span = (0.0, 1.0)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._recompute)

    # ---------- setup ----------

    def set_function(self, evaluate, title: str, names, ranges, optimum, max_resolution: int):
        """Raises ValueError if the function cannot be evaluated or is undefined everywhere."""
        self._lower = np.array([lo for lo, _ in ranges], dtype=float)
        self._upper = np.array([hi for _, hi in ranges], dtype=float)
        rng = np.random.default_rng(0)  # the same scale on every run
        points = rng.uniform(self._lower, self._upper, (SAMPLE_SIZE, len(names)))
        sample = np.asarray(evaluate(points), dtype=float)
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            raise ValueError("The function is undefined at every sampled point of the range")

        self._evaluate = evaluate
        self._title = title
        self._names = names
        self._optimum = optimum
        self._max_resolution = max_resolution
        self._sample = sample
        self._focus = self.start_point()
        self._drag = None

    def set_appearance(self, cmap, clip: bool, show_optimum: bool):
        """Applies the display options and recomputes the panels."""
        self._lut = cmap.getLookupTable(nPts=LUT_SIZE, alpha=True, mode="byte")
        self._clip = clip
        self._show_optimum = show_optimum
        if clip:
            low, high = np.percentile(self._sample, [1, 99])
        else:
            low, high = self._sample.min(), self._sample.max()
            if self._optimum is not None:
                low = min(low, self._optimum[1])
        if high - low < 1e-12:  # constant function
            low, high = low - 1.0, high + 1.0
        self._span = float(low), float(high)
        self.rangeChanged.emit(*self._span)
        self._recompute()

    def start_point(self) -> np.ndarray:
        """The known minimum if it lies inside the domain, the domain center otherwise."""
        point = self._optimum_in_domain()
        return point.copy() if point is not None else (self._lower + self._upper) / 2

    def _optimum_in_domain(self):
        if self._optimum is None:
            return None
        point = self._optimum[0]
        return point if np.all((self._lower <= point) & (point <= self._upper)) else None

    def has_optimum(self) -> bool:
        return self._optimum_in_domain() is not None

    def set_focus(self, point: np.ndarray):
        self._focus = np.clip(point, self._lower, self._upper)
        self._recompute()

    def focus_center(self):
        self.set_focus((self._lower + self._upper) / 2)

    def focus_start(self):
        self.set_focus(self.start_point())

    # ---------- computation ----------

    def _panel_size(self) -> float:
        count = len(self._names)
        return max(4.0, min(self.width() - LEFT - RIGHT, self.height() - TOP - BOTTOM) / count - GAP)

    def _resolution(self) -> int:
        side = int(self._panel_size() * self.devicePixelRatioF())
        return int(np.clip(min(side, self._max_resolution), MIN_RESOLUTION, MAX_RESOLUTION))

    def _recompute(self):
        if self._evaluate is None:
            return
        self._timer.stop()
        count = len(self._names)
        res = self._resolution()
        self._axes = [np.linspace(lo, hi, res) for lo, hi in zip(self._lower, self._upper)]
        pairs = [(i, j) for j in range(count) for i in range(j)]
        size_2d, size_1d = res * res, res

        # All panels go into one batch: evaluation in the DLL or numpy is much cheaper per point this way.
        points = np.empty((len(pairs) * size_2d + count * size_1d + 1, count))
        points[:] = self._focus
        position = 0
        for i, j in pairs:
            points[position:position + size_2d, i] = np.repeat(self._axes[i], res)
            points[position:position + size_2d, j] = np.tile(self._axes[j], res)
            position += size_2d
        for i in range(count):
            points[position:position + size_1d, i] = self._axes[i]
            position += size_1d

        try:
            values = np.asarray(self._evaluate(points), dtype=float)
        except ValueError as exc:
            self.statusChanged.emit(str(exc))
            return
        self._focus_value = float(values[-1])
        bad = ~np.isfinite(values[:-1])
        self._bad = int(bad.sum())
        filled = values[:-1].copy()
        filled[bad] = self._span[0]  # undefined points are colored as the minimum, as in the other views

        if not self._clip and not bad.all():
            # The scale comes from a random sample, so slices may exceed it: widen it, it never shrinks while dragging.
            low, high = min(self._span[0], filled.min()), max(self._span[1], filled.max())
            if (low, high) != self._span:
                self._span = float(low), float(high)
                self.rangeChanged.emit(*self._span)
        low, high = self._span
        indices = (np.clip((filled - low) / (high - low), 0, 1) * (LUT_SIZE - 1)).astype(int)
        colors = self._lut[indices]

        self._values, self._images = {}, {}
        position = 0
        for i, j in pairs:
            self._values[i, j] = filled[position:position + size_2d].reshape(res, res)
            # rows of the image run top to bottom, x_j grows upwards
            rgba = colors[position:position + size_2d].reshape(res, res, 4).transpose(1, 0, 2)[::-1]
            self._images[i, j] = self._to_image(rgba)
            position += size_2d
        for i in range(count):
            self._values[i, i] = filled[position:position + size_1d]
            position += size_1d

        self._report_focus()
        self.update()

    @staticmethod
    def _to_image(rgba: np.ndarray) -> QtGui.QImage:
        data = np.ascontiguousarray(rgba, dtype=np.uint8)
        height, width, _ = data.shape
        return QtGui.QImage(data.data, width, height, 4 * width, QtGui.QImage.Format.Format_RGBA8888).copy()

    def _report_focus(self):
        if self._hover_active:
            return
        low, high = self._span
        value = f"{self._focus_value:.6g}" if np.isfinite(self._focus_value) else "undefined"
        message = f"{self._title}: f at the focus point = {value}; color scale {low:.4g} … {high:.4g}"
        if self._bad:
            message += f"; undefined at {self._bad} points (colored as the minimum)"
        self.statusChanged.emit(message)

    # ---------- drawing ----------

    def _panel_rect(self, col: int, row: int) -> QtCore.QRectF:
        size = self._panel_size()
        return QtCore.QRectF(LEFT + col * (size + GAP), TOP + row * (size + GAP), size, size)

    def _fractions(self, col: int, row: int, point: np.ndarray) -> tuple[float, float]:
        """Position of a point inside panel (col, row), both from 0 to 1; v grows upwards."""
        span = self._upper - self._lower
        u = (point[col] - self._lower[col]) / span[col]
        v = (point[row] - self._lower[row]) / span[row]
        return float(u), float(v)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if self._evaluate is None or not self._values:
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        count = len(self._names)
        optimum = self._optimum_in_domain() if self._show_optimum else None
        mismatch = None
        if optimum is not None:
            mismatch = np.abs(self._focus - optimum) > 1e-9 * (self._upper - self._lower)

        for row in range(count):
            for col in range(row + 1):
                rect = self._panel_rect(col, row)
                if col == row:
                    self._draw_profile(painter, rect, col, optimum, mismatch)
                else:
                    self._draw_slice(painter, rect, col, row, optimum, mismatch)
                painter.setPen(FRAME_COLOR)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)

        painter.setPen(LABEL_COLOR)
        for i, name in enumerate(self._names):
            cell = self._panel_rect(i, i)
            painter.drawText(
                QtCore.QRectF(cell.x(), TOP + count * (cell.height() + GAP) - GAP, cell.width(), BOTTOM),
                QtCore.Qt.AlignmentFlag.AlignCenter, name,
            )
            painter.drawText(
                QtCore.QRectF(0, cell.y(), LEFT - 4, cell.height()),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, name,
            )
        self._draw_info(painter)

    def _draw_slice(self, painter, rect, col, row, optimum, mismatch):
        painter.drawImage(rect, self._images[col, row])
        u, v = self._fractions(col, row, self._focus)
        x, y = rect.left() + u * rect.width(), rect.bottom() - v * rect.height()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QtGui.QPen(FOCUS_COLOR, 1))
        painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
        painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
        if optimum is None:
            return
        u, v = self._fractions(col, row, optimum)
        center = QtCore.QPointF(rect.left() + u * rect.width(), rect.bottom() - v * rect.height())
        on_slice = mismatch.sum() - mismatch[col] - mismatch[row] == 0
        self._draw_marker(painter, center, on_slice)

    def _draw_profile(self, painter, rect, index, optimum, mismatch):
        low, high = self._span
        heights = np.clip((self._values[index, index] - low) / (high - low), 0, 1)
        xs = rect.left() + np.linspace(0, 1, len(heights)) * rect.width()
        ys = rect.bottom() - heights * rect.height()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QtGui.QPen(CURVE_COLOR, 1.2))
        painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in zip(xs, ys)]))

        u = (self._focus[index] - self._lower[index]) / (self._upper[index] - self._lower[index])
        x = rect.left() + u * rect.width()
        painter.setPen(QtGui.QPen(FOCUS_COLOR, 1))
        painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
        if optimum is not None and mismatch.sum() - mismatch[index] == 0:
            u = (optimum[index] - self._lower[index]) / (self._upper[index] - self._lower[index])
            t = np.clip((self._optimum[1] - low) / (high - low), 0, 1)
            self._draw_marker(painter, QtCore.QPointF(rect.left() + u * rect.width(), rect.bottom() - t * rect.height()), True)

    @staticmethod
    def _draw_marker(painter, center, on_slice: bool):
        """Filled: the minimum lies on this slice. Ring: only its projection onto these two axes."""
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        if on_slice:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
            painter.setBrush(QtGui.QColor(255, 255, 255))
            painter.drawEllipse(center, 4.0, 4.0)
        else:
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2.5))
            painter.drawEllipse(center, 4.0, 4.0)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.2))
            painter.drawEllipse(center, 4.0, 4.0)

    def _draw_info(self, painter):
        """Focus coordinates and the legend, in the free space above the diagonal."""
        count = len(self._names)
        first = count // 2 + 1  # this column and those to the right are free in rows up to first - 1
        corner = self._panel_rect(first, 0)
        area = QtCore.QRectF(corner.x() + 8, corner.y(), self.width() - RIGHT - corner.x() - 8,
                             self._panel_rect(0, first - 1).bottom() - corner.y())
        lines = [f"{name} = {value:.5g}" for name, value in zip(self._names, self._focus)]
        value = f"{self._focus_value:.6g}" if np.isfinite(self._focus_value) else "undefined"
        lines += [f"f = {value}"]
        if self._optimum is not None:
            lines += [f"f* = {self._optimum[1]:.6g}"]
        lines += ["", "Drag inside a panel to move", "the focus point (crosshairs)."]
        if self._show_optimum and self.has_optimum():
            lines += ["● minimum on this slice", "○ its projection on the axes"]
        painter.setPen(LABEL_COLOR)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.drawText(area, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, "\n".join(lines))

    # ---------- interaction ----------

    def _panel_at(self, position: QtCore.QPointF):
        if self._evaluate is None:
            return None
        size = self._panel_size()
        col = int((position.x() - LEFT) // (size + GAP))
        row = int((position.y() - TOP) // (size + GAP))
        if not 0 <= col <= row < len(self._names):
            return None
        return (col, row) if self._panel_rect(col, row).contains(position) else None

    def _panel_fractions(self, panel, position: QtCore.QPointF) -> tuple[float, float]:
        rect = self._panel_rect(*panel)
        u = np.clip((position.x() - rect.left()) / rect.width(), 0, 1)
        v = np.clip((rect.bottom() - position.y()) / rect.height(), 0, 1)
        return float(u), float(v)

    def _drag_to(self, position: QtCore.QPointF):
        col, row = self._drag
        u, v = self._panel_fractions(self._drag, position)
        self._focus[col] = self._lower[col] + u * (self._upper[col] - self._lower[col])
        if col != row:
            self._focus[row] = self._lower[row] + v * (self._upper[row] - self._lower[row])
        if not self._timer.isActive():
            self._timer.start()  # mouse moves arriving before the next recomputation are merged

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self._drag = self._panel_at(event.position())
        if self._drag is not None:
            self._hover_active = False
            self._drag_to(event.position())

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self._drag_to(event.position())
            return
        panel = self._panel_at(event.position())
        if panel is None or not self._values:
            self.leaveEvent(None)
            return
        self._hover_active = True
        self.statusChanged.emit(self._describe_point(panel, event.position()))

    def leaveEvent(self, event):
        if self._hover_active:
            self._hover_active = False
            if self._values:
                self._report_focus()

    def _describe_point(self, panel, position: QtCore.QPointF) -> str:
        """Coordinates and the value of the grid point under the cursor; the rest of the point is the focus."""
        col, row = panel
        u, v = self._panel_fractions(panel, position)
        res = len(self._axes[0])
        i, j = int(round(u * (res - 1))), int(round(v * (res - 1)))
        if col == row:
            value = self._values[col, col][i]
            text = f"{self._names[col]} = {self._axes[col][i]:.4g}"
        else:
            value = self._values[col, row][i, j]
            text = (f"{self._names[col]} = {self._axes[col][i]:.4g}, "
                    f"{self._names[row]} = {self._axes[row][j]:.4g}")
        return f"{text}, the others at the focus point: f = {value:.6g}"

    def resizeEvent(self, event):
        if self._evaluate is not None and self._values and not self._timer.isActive():
            self._timer.start()
