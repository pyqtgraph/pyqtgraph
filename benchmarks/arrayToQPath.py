import numpy as np

import pyqtgraph as pg

rng = np.random.default_rng(12345)

class _TimeSuite:
    param_names = ["Size", "Connection Type"]
    params = ([10_000, 100_000, 1_000_000], ['all', 'finite', 'pairs', 'array'])


    def setup(self, nelems, connect):
        self.xdata = np.arange(nelems, dtype=np.float64)
        self.ydata = rng.standard_normal(nelems, dtype=np.float64)
        if connect == 'array':
            self.connect_array = np.ones(nelems, dtype=bool)
        if self.have_nonfinite:
            self.ydata[::5000] = np.nan

    def time_test(self, nelems, connect):
        if connect == 'array':
            connect = self.connect_array
        pg.arrayToQPath(self.xdata, self.ydata, connect=connect)

class TimeSuiteAllFinite(_TimeSuite):
    def __init__(self):
        super().__init__()
        self.have_nonfinite = False

class TimeSuiteWithNonFinite(_TimeSuite):
    def __init__(self):
        super().__init__()
        self.have_nonfinite = True


# _arraytoqpath_finite() has two independent code paths, selected by the
# *method* keyword argument:
#
#   'per_segment' builds a QPolygonF per finite segment and joins them with
#                  QPainterPath.addPolygon(). It ignores the enableExperimental
#                  config option entirely.
#   'bulk'        packs the vertices into a single buffer up front. The
#                  enableExperimental config option decides whether that buffer
#                  is written straight into the path's own storage (True) or
#                  handed over through a QDataStream round trip (False).
#
# The method names are deliberately agnostic about enableExperimental, since
# that option only ever affects the 'bulk' path.
#
# Each entry is (method, enableExperimental).
FINITE_METHODS = [
    ('per_segment', False),
    ('bulk', False),
    ('bulk', True),
]

# Column labels, in display order. asv encodes param values into the result
# filenames, so keep these plain and free of whitespace.
METHOD_NAMES = [
    'polygon_per_segment',
    'bulk_qdatastream',
    'bulk_inplace',
]

# label -> (method, enableExperimental)
METHOD_PARAMS = dict(zip(METHOD_NAMES, FINITE_METHODS))

NAN_SPACINGS = [3, 5, 10, 15, 20, 25, 30, 50, 100, 250, 500, 1_000, 10_000]


class TimeNanSpacing:
    param_names = ["nan_spacing", "method"]
    params = (NAN_SPACINGS, METHOD_NAMES)

    num_points = 100_000

    def setup(self, nan_spacing, method):
        self.qpath_method, self.enable_experimental = METHOD_PARAMS[method]
        pg.setConfigOption('enableExperimental', self.enable_experimental)

        self.xdata = np.arange(self.num_points, dtype=np.float64)
        self.ydata = rng.standard_normal(self.num_points, dtype=np.float64)
        self.ydata[::nan_spacing] = np.nan

    def teardown(self, *args, **kwargs):
        # toggle the option back off
        pg.setConfigOption('enableExperimental', False)

    def time_test(self, nan_spacing, method):
        pg.arraytoline._arraytoqpath_finite(
            self.xdata, self.ydata, method=self.qpath_method
        )
