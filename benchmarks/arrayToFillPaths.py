"""Benchmark for arraytoline.arrayToFillPaths().

arrayToFillPaths() chops a filled curve into a list of QPainterPaths, one
per chunk of `chunksize` points, so the cost of building the paths and the cost
of rasterizing them both depend on chunksize and pull in opposite directions:

  * small chunksize -> many small QPainterPaths. Each one is cheap to build and
    cheap for the rasterizer to scan-convert, but the per-path overhead is paid
    tens of thousands of times.
  * large chunksize -> one or two huge QPainterPaths. Almost no per-path
    overhead, but the rasterizer has to walk a very long edge list per fill and
    the implicit closing edges grow too.

The interesting quantity is therefore the *total* cost of creating the paths and
painting them into a QImage, which is what this benchmark times. Only
connect='all' is exercised, i.e. a single run of points with no non-finite
values, so the finite-mask and segment-splitting machinery stays out of the
measurement.

Noise appears in the y data to make the fill paths more jagged and therefore more
expensive to rasterize. The noise is scaled to the y range of the sine curve, so a
noise of 0.2 means the y values are perturbed by a standard deviation of 20% of the
sine amplitude. The noise is applied after the sine curve is generated, so it does
not affect the x values or the number of points.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

rng = np.random.default_rng(0)

# point count of the test curve
NUM_POINTS = 20_000

# number of full sine cycles spanned over the x range
NUM_CYCLES = 20

# chunksizes to compare.
# The smaller ones probe the per-path overhead and the larger ones probe the
# scan-conversion cost. 
CHUNKSIZES = [2, 10, 50, 100, 125, 150, 175, 200, 250, 500, 1_000, 5_000, 10_000, 20_000]

# size of the QImage the fill paths get rasterized into
IMAGE_SIZE = (800, 600)

# room left around the data, standing in for the axes
MARGIN = 20.0


def sine_curve(num_points: int = NUM_POINTS, num_cycles: int = NUM_CYCLES):
    x = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    y = np.sin(2.0 * np.pi * num_cycles * x)
    return x, y


def fit_transform(data_rect, image_rect):
    """QTransform mapping *data_rect* onto *image_rect*.

    *data_rect* is (xmin, xmax, ymin, ymax) in data coordinates (y pointing
    up), *image_rect* is (left, right, top, bottom) in pixels (y pointing
    down), which mirrors the device transform a PlotCurveItem gets from its
    ViewBox.
    """
    xmin, xmax, ymin, ymax = data_rect
    left, right, top, bottom = image_rect

    scale_x = (right - left) / (xmax - xmin)
    scale_y = -(bottom - top) / (ymax - ymin)  # negative flips y

    # QTransform post-multiplies, so translating *before* scaling means the
    # data is scaled first and the translation is applied to the result.
    transform = QtGui.QTransform()
    transform.translate(left - scale_x * xmin, top - scale_y * ymax)
    transform.scale(scale_x, scale_y)
    return transform


class TimeArrayToFillPaths:
    unit = "seconds"
    param_names = ["chunksize", "noise"]
    params = [CHUNKSIZES, [0.0, 0.2]]

    def setup(self, chunksize, noise):
        self.x, self.y = sine_curve()
        self.y += noise * rng.standard_normal(size=self.y.shape)

        self.brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        width, height = IMAGE_SIZE
        self.transform = fit_transform(
            (float(self.x[0]), float(self.x[-1]),
             float(self.y.min()), float(self.y.max())),
            (MARGIN, width - MARGIN, MARGIN, height - MARGIN),
        )

    def time_test(self, chunksize, noise):
        image = QtGui.QImage(*IMAGE_SIZE, QtGui.QImage.Format.Format_RGB32)
        image.fill(0)

        painter = QtGui.QPainter(image)
        try:
            # explicitly off: this is pyqtgraph's default for curves and
            # leaving it to the config option would make the benchmark depend
            # on the ambient configuration.
            painter.setRenderHint(painter.RenderHint.Antialiasing, False)
            painter.setTransform(self.transform)

            fill_path_list = pg.arraytoline.arrayToFillPaths(
                self.x, self.y, "all", 0.0, chunksize
            )
            for path in fill_path_list:
                painter.fillPath(path, self.brush)
        finally:
            painter.end()