__all__ = ['arrayToQPath']

import numpy as np
import numpy.typing as npt

from . import Qt
from . import getConfigOption
from .Qt import QtCore, QtGui

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

def _compute_finite_segments(finite_mask) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
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

def _compute_finite_mask(x, y):
    x_integer = isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer)
    y_integer = isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)

    if x_integer and y_integer:
        return np.ones(x.shape, dtype=bool)
    elif x_integer:
        return np.isfinite(y)
    elif y_integer:
        return np.isfinite(x)
    else:
        return np.isfinite(x) & np.isfinite(y)

def _arraytoqpath_all(x, y, finiteCheck):
    num_points = x.shape[0]
    if num_points == 0:
        return QtGui.QPainterPath()

    finite_idx = None
    if finiteCheck:
        finite_mask = _compute_finite_mask(x, y)
        if not finite_mask.all():
            finite_idx = finite_mask.nonzero()[0]
            num_points = len(finite_idx)

    if num_points < 2:
        return QtGui.QPainterPath()

    # calculate number of chunks based on nominal size
    chunksize = min(num_points, 10000)
    num_chunks = (num_points + chunksize - 1) // chunksize

    # too few chunks, chunking would be a pessimization
    if num_chunks < 3:
        num_chunks = 1

    # distribute the points evenly across the chunks
    chunksize = num_points // num_chunks
    remainder = num_points % num_chunks

    fullpath = None
    subpath = None
    polybuf = Qt.internals.QPolygonBuffer(chunksize + int(remainder != 0))

    offset = 0
    while offset < num_points:
        currsize = chunksize + int(remainder != 0)
        if remainder:
            remainder -= 1

        polybuf.resize(currsize)
        subarr = polybuf.ndarray()
        sl = slice(offset, offset + currsize)
        if finite_idx is None:
            subarr[:, 0] = x[sl]
            subarr[:, 1] = y[sl]
        else:
            fiv = finite_idx[sl]    # view
            subarr[:, 0] = x[fiv]
            subarr[:, 1] = y[fiv]

        if fullpath is None:
            fullpath = QtGui.QPainterPath()
            fullpath.reserve(num_points)
            fullpath.addPolygon(polybuf.qpolygon())
        else:
            if subpath is None:
                subpath = QtGui.QPainterPath()
            else:
                subpath.clear()
            subpath.reserve(currsize)
            subpath.addPolygon(polybuf.qpolygon())
            fullpath.connectPath(subpath)

        offset += currsize
    return fullpath


def _arraytoqpath_finite(x, y, finite_mask=None, *, method=None):
    """
    Build a QPainterPath from *x*, *y*, breaking it at non-finite points.

    Parameters
    ----------
    x, y : np.ndarray
        coordinates of shape (N,)
    finite_mask : np.ndarray, optional
        boolean mask of shape (N,) selecting the finite points. Computed from
        *x* and *y* when not supplied.
    method : {'per_segment', 'bulk'}, optional
        how the vertices are transferred into the QPainterPath.

        'per_segment'
            Call ``QPainterPath.addPolygon()`` once per contiguous finite
            segment, staging each one through a reusable QPolygonF. Cost scales
            with the number of segments.
        'bulk'
            Gather every finite segment into a single buffer in one pass and
            hand it to the path at once. Cost scales with the total point
            count. This path is further specialised by the ``enableExperimental``
            config option, which selects between writing directly into the
            path's own storage and a QDataStream round trip.

        When ``None`` (the default) a method is chosen heuristically.
    """
    n = x.shape[0]
    if n == 0:
        return QtGui.QPainterPath()

    if finite_mask is None:
        finite_mask = _compute_finite_mask(x, y)
    sidx, slen = _compute_finite_segments(finite_mask)

    num_points = int(slen.sum())
    if num_points == 0:
        return QtGui.QPainterPath()

    if method is None:
        experimental = getConfigOption('enableExperimental')
        # tuning values from running benchmarks/arrayToQPath.py
        if num_points > len(slen) * (100 if experimental else 15):
            method = 'per_segment'
        else:
            method = 'bulk'
    elif method not in ('per_segment', 'bulk'):
        raise ValueError(
            f"method must be one of 'per_segment', 'bulk'; got {method!r}"
        )

    if method == 'per_segment':
        path = QtGui.QPainterPath()
        path.reserve(num_points)

        # create a single polygon able to hold the largest chunk
        polybuf = Qt.internals.QPolygonBuffer(max(slen))

        # converting tolist() is measurably faster in benchmarks/arrayToQPath.py
        for i, l in zip(sidx.tolist(), slen.tolist()):
            polybuf.resize(l)
            subarr = polybuf.ndarray()
            subarr[:, 0] = x[i:i+l]
            subarr[:, 1] = y[i:i+l]
            path.addPolygon(polybuf.qpolygon())

        return path
    else:
        move_ind = np.cumsum(slen) - slen
        data_ind = np.arange(num_points) + np.repeat(sidx - move_ind, slen)

        qpath_buffer = Qt.internals.QPainterPathBuffer(num_points)
        arr = qpath_buffer.ndarray()

        arr['c'] = 1
        arr['c'][move_ind] = 0
        arr['x'] = x[data_ind]
        arr['y'] = y[data_ind]

        return qpath_buffer.to_qpainterpath()


def _arraytoqpath_pairs(x, y, finiteCheck):
    # ensure that we have an even number of elements
    n = len(x) // 2 * 2
    x = x[:n]
    y = y[:n]

    if finiteCheck:
        mask = _compute_finite_mask(x, y)
        if not np.all(mask):
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

def _arraytoqpath_array(x, y, connect_array, finiteCheck):
    num_points = x.shape[0]
    if num_points == 0:
        return QtGui.QPainterPath()

    if finiteCheck:
        finite_mask = _compute_finite_mask(x, y)
        if not np.all(finite_mask):
            backfill_idx = _compute_backfill_indices(finite_mask)
            if backfill_idx is None:    # all non-finite
                return QtGui.QPainterPath()
            x = x[backfill_idx]
            y = y[backfill_idx]

    qpath_buffer = Qt.internals.QPainterPathBuffer(num_points)
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

    connect_kind, connect_array = connect, None
    del connect
    if isinstance(connect_kind, np.ndarray):
        # make connect argument contain only str type
        connect_kind, connect_array = "array", connect_kind

    match connect_kind:
        case "all":
            return _arraytoqpath_all(x, y, finiteCheck)

        case "finite":
            finite_mask = _compute_finite_mask(x, y)
            if not np.all(finite_mask):
                return _arraytoqpath_finite(x, y, finite_mask)
            else:
                return _arraytoqpath_all(x, y, finiteCheck=False)

        case "pairs":
            return _arraytoqpath_pairs(x, y, finiteCheck)

        case "array":
            return _arraytoqpath_array(x, y, connect_array, finiteCheck)

    raise ValueError('connect argument must be "all", "pairs", "finite", or array')

def arrayToLineSegments(x, y, connect, finiteCheck, out=None):
    if out is None:
        out = Qt.internals.PrimitiveArray(QtCore.QLineF, 4)

    # analogue of arrayToQPath taking the same parameters
    if len(x) < 2:
        out.resize(0)
        return out

    connect_array = None
    if isinstance(connect, np.ndarray):
        # the last element is not used
        connect_array, connect = np.asarray(connect[:-1], dtype=bool), 'array'

    all_finite = True
    if finiteCheck or connect == 'finite':
        mask = _compute_finite_mask(x, y)
        all_finite = np.all(mask)

    if connect == 'all':
        if not all_finite:
            # remove non-finite points, if any
            x = x[mask]
            y = y[mask]

    elif connect == 'finite':
        if all_finite:
            connect = 'all'
        else:
            # each non-finite point affects the segment before and after
            connect_array = mask[:-1] & mask[1:]

    elif connect == 'pairs':
        if not all_finite:
            # ensure that we have an even number of elements
            npairs = len(x) // 2
            mask = mask[:npairs*2]
            # remove pair if at least one point within pair is non-finite
            mask.reshape((-1, 2))[:] = (mask[0::2] & mask[1::2])[:, np.newaxis]
            x = x[:npairs*2][mask]
            y = y[:npairs*2][mask]

    elif connect == 'array':
        if not all_finite:
            # replicate the behavior of arrayToQPath
            backfill_idx = _compute_backfill_indices(mask)
            if backfill_idx is None:    # all non-finite
                out.resize(0)
                return out
            x = x[backfill_idx]
            y = y[backfill_idx]

    if connect == 'all':
        nsegs = len(x) - 1
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1]
            memory[:, 2] = x[1:]
            memory[:, 1] = y[:-1]
            memory[:, 3] = y[1:]

    elif connect == 'pairs':
        nsegs = len(x) // 2
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory = memory.reshape((-1, 2))
            memory[:, 0] = x[:nsegs * 2]
            memory[:, 1] = y[:nsegs * 2]

    elif connect_array is not None:
        # the following are handled here
        # - 'array'
        # - 'finite' with non-finite elements
        nsegs = np.count_nonzero(connect_array)
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1][connect_array]
            memory[:, 2] = x[1:][connect_array]
            memory[:, 1] = y[:-1][connect_array]
            memory[:, 3] = y[1:][connect_array]

    else:
        nsegs = 0
        out.resize(nsegs)

    return out
