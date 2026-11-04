import warnings

from .plotConfigTemplate import Ui_Form  # noqa: F401

warnings.warn(
    "pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_generic has been "
    "renamed to pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate. This "
    "compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
