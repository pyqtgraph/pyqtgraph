import warnings

from VideoTemplate import Ui_MainWindow  # noqa: F401

warnings.warn(
    "VideoTemplate_generic has been renamed to VideoTemplate. This "
    "compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
