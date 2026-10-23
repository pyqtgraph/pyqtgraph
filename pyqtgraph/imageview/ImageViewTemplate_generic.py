import warnings

from .ImageViewTemplate import Ui_Form  # noqa: F401

warnings.warn(
    "pyqtgraph.imageview.ImageViewTemplate_generic has been renamed to "
    "pyqtgraph.imageview.ImageViewTemplate. This compatibility shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
