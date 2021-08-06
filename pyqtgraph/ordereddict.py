import collections
import warnings


class OrderedDict(collections.OrderedDict):
    def __init__(self, *args, **kwds):
        warnings.warn(
            "OrderedDict is in the standard library for supported versions of Python. Will be removed in 0.13",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwds)
