import warnings

from .FlowchartCtrlTemplate import Ui_Form  # noqa: F401

warnings.warn(
    "pyqtgraph.flowchart.FlowchartCtrlTemplate_generic has been renamed to "
    "pyqtgraph.flowchart.FlowchartCtrlTemplate. This compatibility shim will "
    "be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
