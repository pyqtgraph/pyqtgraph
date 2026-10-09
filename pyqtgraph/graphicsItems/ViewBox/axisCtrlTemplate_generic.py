import warnings

from .axisCtrlTemplate import Ui_Form  # noqa: F401

warnings.warn(
    "pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_generic has been "
    "renamed to pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate. This "
    "compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
