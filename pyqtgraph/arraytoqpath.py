__all__ = ['arrayToQPath']

import numpy as np
import numpy.typing as npt

from . import Qt
from .Qt import QtGui

def _compute_backfill_indices(isfinite):
    # the presence of inf/nans result in an empty QPainterPath being generated
    # this behavior started in Qt 5.12.3 and was introduced in this commit
    # https://github.com/qt/qtbase/commit/c04bd30de072793faee5166cff866a4c4e0a9dd7
    # We therefore replace non-finite values

    # credit: Divakar https://stackoverflow.com/a/41191127/643629
    mask = ~isfinite
    idx = np.arange(len(isfinite))
    idx[mask] = -1
    np.maximum.accumulate(idx, out=idx)
    first = np.searchsorted(idx, 0)
    if first < len(isfinite):
        # Replace all non-finite entries from beginning of arr with the first finite one
        idx[:first] = first
        return idx
    else:
        return None

def _compute_finite_segments(finite_mask) -> tuple[npt.NDArray[np.intp], ...]:
    # from a boolean mask of finite positions, generate two ndarrays.
    # first array contains the starting indices of the finite segments.
    # second array contains the length of the finite segments.
    # a "run" consists of at least 2 elements.
    nonfinite_locs = np.nonzero(~finite_mask)[0]
    # pretend that there's a nonfinite before and after the array
    nonfinite_locs = np.concatenate(([-1], nonfinite_locs, [len(finite_mask)]))
    sidx = nonfinite_locs[:-1] + 1      # start index of segment
    slen = np.diff(nonfinite_locs) - 1  # length of segment
    mask = slen >= 2
    sidx = sidx[mask]
    slen = slen[mask]
    return sidx, slen

def _arrayToQPath_all(x, y, finiteCheck):
    n = x.shape[0]
    if n == 0:
        return QtGui.QPainterPath()

    finite_idx = None
    if finiteCheck:
        isfinite = np.isfinite(x) & np.isfinite(y)
        if not isfinite.all():
            finite_idx = isfinite.nonzero()[0]
            n = len(finite_idx)

    if n < 2:
        return QtGui.QPainterPath()

    chunksize = 10000
    numchunks = (n + chunksize - 1) // chunksize
    minchunks = 3

    if numchunks < minchunks:
        # too few chunks, batching would be a pessimization
        polybuf = Qt.internals.QPolygonBuffer(n)
        arr = polybuf.ndarray()

        if finite_idx is None:
            arr[:, 0] = x
            arr[:, 1] = y
        else:
            arr[:, 0] = x[finite_idx]
            arr[:, 1] = y[finite_idx]

        return polybuf.to_qpainterpath()

    # at this point, we have numchunks >= minchunks

    path = QtGui.QPainterPath()
    path.reserve(n)
    polybuf = Qt.internals.QPolygonBuffer(chunksize)
    for idx in range(numchunks):
        sl = slice(idx*chunksize, min((idx+1)*chunksize, n))
        currsize = sl.stop - sl.start
        polybuf.resize(currsize)
        subarr = polybuf.ndarray()
        if finite_idx is None:
            subarr[:, 0] = x[sl]
            subarr[:, 1] = y[sl]
        else:
            fiv = finite_idx[sl]  # view
            subarr[:, 0] = x[fiv]
            subarr[:, 1] = y[fiv]
        path.connectPath(polybuf.to_qpainterpath())
    return path


def _arrayToQPath_finite(x, y, isfinite=None, *, method=None):
    n = x.shape[0]
    if n == 0:
        return QtGui.QPainterPath()

    if isfinite is None:
        isfinite = np.isfinite(x) & np.isfinite(y)

    sidx, slen = _compute_finite_segments(isfinite)

    num_points = slen.sum()
    if num_points == 0:
        return QtGui.QPainterPath()

    if method is None or method not in ['qpolygonf', 'qpainterpath']:
        if num_points >= 15 * len(slen):
            method = 'qpolygonf'
        else:
            method = 'qpainterpath'

    if method == 'qpolygonf':
        path = QtGui.QPainterPath()
        path.reserve(num_points)

        # create a single polygon able to hold the largest chunk
        polybuf = Qt.internals.QPolygonBuffer(max(slen))

        for i, l in zip(sidx, slen):
            polybuf.resize(l)
            subarr = polybuf.ndarray()
            subarr[:, 0] = x[i:i+l]
            subarr[:, 1] = y[i:i+l]
            path.addPolygon(polybuf.qpolygon())

        return path
    else:
        # move_draw = np.concatenate([np.r_[False, np.ones(l - 1, dtype=bool)] for l in slen])
        move_ind = np.cumsum(slen) - slen
        # data_ind = np.concatenate([np.arange(i, i + l) for i, l in zip(sidx, slen)])
        data_ind = np.arange(num_points) + np.repeat(sidx - move_ind, slen)

        qpath_buffer = Qt.internals.QPainterPathBuffer(num_points)
        arr = qpath_buffer.ndarray()

        arr['c'] = 1
        arr['c'][move_ind] = 0
        arr['x'] = x[data_ind]
        arr['y'] = y[data_ind]

        return qpath_buffer.to_qpainterpath()


def _arrayToQPath_pairs(x, y, finiteCheck):
    # ensure that we have an even number of elements
    n = len(x) // 2 * 2
    x = x[:n]
    y = y[:n]

    if finiteCheck:
        isfinite = np.isfinite(x) & np.isfinite(y)
        if not np.all(isfinite):
            mask = isfinite
            # remove pair if at least one point within pair is non-finite
            mask.reshape((-1, 2))[:] = (mask[0::2] & mask[1::2])[:, np.newaxis]
            x = x[mask]
            y = y[mask]
            n = len(x)

    if n == 0:
        return QtGui.QPainterPath()

    qpath_buffer = Qt.internals.QPainterPathBuffer(n)
    arr = qpath_buffer.ndarray()
    arr['c'][0::2] = 0
    arr['c'][1::2] = 1
    arr['x'] = x
    arr['y'] = y
    return qpath_buffer.to_qpainterpath()

def arrayToQPath(x, y, connect='all', finiteCheck=True):
    """
    Convert an array of x,y coordinates to QPainterPath as efficiently as
    possible. The *connect* argument may be 'all', indicating that each point
    should be connected to the next; 'pairs', indicating that each pair of
    points should be connected, or an array of int32 values (0 or 1) indicating
    connections.
    
    Parameters
    ----------
    x : np.ndarray
        x-values to be plotted of shape (N,)
    y : np.ndarray
        y-values to be plotted, must be same length as `x` of shape (N,)
    connect : {'all', 'pairs', 'finite', (N,) ndarray}, optional
        Argument detailing how to connect the points in the path. `all` will 
        have sequential points being connected.  `pairs` generates lines
        between every other point.  `finite` only connects points that are
        finite.  If an ndarray is passed, containing int32 values of 0 or 1,
        only values with 1 will connect to the previous point.  Def
    finiteCheck : bool, default True
        When false, the check for finite values will be skipped, which can
        improve performance. If nonfinite values are present in `x` or `y`,
        an empty QPainterPath will be generated.
    
    Returns
    -------
    QPainterPath
        QPainterPath object to be drawn
    
    Raises
    ------
    ValueError
        Raised when the connect argument has an invalid value placed within.

    Notes
    -----
    A QPainterPath is generated through one of two ways.  When the connect
    parameter is 'all', a QPolygonF object is created, and
    ``QPainterPath.addPolygon()`` is called.  For other connect parameters
    a ``QDataStream`` object is created and the QDataStream >> QPainterPath
    operator is used to pass the data.  The memory format is as follows

    .. code-block:
        numVerts(i4)
        0(i4)   x(f8)   y(f8)    <-- 0 means this vertex does not connect
        1(i4)   x(f8)   y(f8)    <-- 1 means this vertex connects to the previous vertex
        ...
        cStart(i4)   fillRule(i4)
    
    see: https://github.com/qt/qtbase/blob/dev/src/gui/painting/qpainterpath.cpp

    All values are big endian--pack using struct.pack('>d') or struct.pack('>i')
    This binary format may change in future versions of Qt
    """

    n = x.shape[0]
    if n == 0:
        return QtGui.QPainterPath()

    connect_array = None
    if isinstance(connect, np.ndarray):
        # make connect argument contain only str type
        connect_array, connect = connect, 'array'

    if connect == 'all':
        return _arrayToQPath_all(x, y, finiteCheck)

    elif connect == 'finite':
        isfinite = np.isfinite(x) & np.isfinite(y)
        if np.all(isfinite):
            return _arrayToQPath_all(x, y, finiteCheck=False)
        else:
            return _arrayToQPath_finite(x, y, isfinite)

    elif connect == 'pairs':
        return _arrayToQPath_pairs(x, y, finiteCheck)

    elif connect == 'array':
        if finiteCheck:
            isfinite = np.isfinite(x) & np.isfinite(y)
            if not np.all(isfinite):
                backfill_idx = _compute_backfill_indices(isfinite)
                x = x[backfill_idx]
                y = y[backfill_idx]

        qpath_buffer = Qt.internals.QPainterPathBuffer(n)
        arr = qpath_buffer.ndarray()
        arr['x'] = x
        arr['y'] = y

        # Let's call a point with either x or y being nan is an invalid point.
        # A point will anyway not connect to an invalid point regardless of the
        # 'c' value of the invalid point. Therefore, we should set 'c' to 0 for
        # the next point of an invalid point.
        arr['c'][:1] = 0  # the first vertex has no previous vertex to connect
        arr['c'][1:] = connect_array[:-1]

        return qpath_buffer.to_qpainterpath()
    else:
        raise ValueError('connect argument must be "all", "pairs", "finite", or array')
