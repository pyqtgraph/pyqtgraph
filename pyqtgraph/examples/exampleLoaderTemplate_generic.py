import warnings

from exampleLoaderTemplate import Ui_Form  # noqa: F401

warnings.warn(
    "exampleLoaderTemplate_generic has been renamed to exampleLoaderTemplate. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
