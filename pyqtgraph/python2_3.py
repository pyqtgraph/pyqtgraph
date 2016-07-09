"""
Helper functions that smooth out the differences between python 2 and 3.
"""
import sys

def asUnicode(x):
    if sys.version_info[0] == 2:
        if isinstance(x, unicode):
            return x
        elif isinstance(x, str):
            return x.decode('UTF-8')
        else:
            return unicode(x)
    else:
        return str(x)


if sys.version_info[0] == 3:
    basestring = str
    xrange = range
else:
    import __builtin__
    basestring = __builtin__.basestring
    xrange = __builtin__.xrange
