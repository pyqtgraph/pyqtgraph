import os
if "BINDER_SERVICE_HOST" in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from .GraphicsView import GraphicsLayoutWidget
from .GraphicsView import PlotWidget

