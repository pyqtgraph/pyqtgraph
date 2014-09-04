"""
Helper functions that smooth out the differences between python 2 and 3.
"""
import sys


if sys.version_info < (3,):
    def asUnicode(x):
        if isinstance(x, str):
            return x.decode("utf-8")
        else:
            return unicode(x)
    string_types = basestring
else:
    def asUnicode(x):
        return str(x)
    string_types = str
