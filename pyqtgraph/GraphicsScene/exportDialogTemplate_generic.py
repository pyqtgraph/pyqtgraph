import warnings

from .exportDialogTemplate import Ui_Form  # noqa: F401

warnings.warn(
    "pyqtgraph.GraphicsScene.exportDialogTemplate_generic has been renamed "
    "to pyqtgraph.GraphicsScene.exportDialogTemplate. This compatibility "
    "shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
