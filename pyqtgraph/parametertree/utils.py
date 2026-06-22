"""JSON serialization utilities for pyqtgraph Parameter trees.

Design notes
------------
``saveState()`` is the canonical serializer: every built-in Parameter subclass
already overrides it to return JSON-primitive values (e.g. ``ColorParameter``
→ RGBA tuple, ``FontParameter`` → string, ``CalendarParameter`` → string,
``PenParameter`` → tuple of primitives, ``QtEnumParameter`` → string name).
This module only needs to handle the residual types that ``saveState()`` still
emits but that the standard ``json`` module cannot encode:

- **tuple** — JSON has no tuple type; plain ``json`` silently converts tuples
  to lists, which breaks ``PenParameter`` (``mkPen`` checks
  ``isinstance(v, tuple)``) and any parameter using tuple-valued opts such as
  ``limits``.  Tuples are preserved with a ``{"__tuple__": [...]}`` sentinel.

- **numpy ndarray** — encoded as ``{"__ndarray__": [...], "dtype": "<dtype>"}``
  so that shape and element type survive a round-trip.

- **numpy scalar types** (``np.integer``, ``np.floating``, ``np.bool_``) —
  silently promoted to the matching Python primitive; no sentinel needed.

- **set / frozenset** — serialized as a sorted list.  All built-in opts that
  use a set (e.g. ``ctrlActions``) only require ``in`` membership checks, so
  restoring as a list is transparent.

- **QIcon** — serialized as ``null``.  A ``QIcon`` object has no lossless JSON
  representation; only a plain string file path (which the standard encoder
  handles directly) survives a round-trip.  ``QIcon.StandardPixmap`` integer
  values are already JSON-native and need no special handling.

- **ColorMap** — two forms depending on whether the map is named:

  1. *Named colormaps* (``colormap.name`` is set) are stored as
     ``{"__colormap__": {"name": "<name>"}}``, reconstructed via
     ``pyqtgraph.colormap.get()``.  A viridis LUT shrinks from ~5 kB to a
     few bytes.
  2. *Anonymous colormaps* fall back to the full
     ``{"__colormap__": {pos, color, mapping_mode, name}}`` form.

If a ``saveState()`` override returns any other non-serializable object (e.g.
a ``QDate`` from a custom parameter type that forgot to convert it), the
encoder raises ``TypeError`` as usual.  The correct fix is always in the
parameter type's ``saveState()``, not here.

Public helpers
--------------
``ParameterJsonEncoder``   — ``JSONEncoder`` subclass; use with ``json.dumps``
``_decode_hook``           — ``object_hook`` for ``json.loads``
"""
from __future__ import annotations

from json import JSONDecoder, JSONEncoder
from typing import Any

import numpy as np

from ..colormap import ColorMap
from ..Qt import QtGui
import pyqtgraph.colormap as pgcm


# ---------------------------------------------------------------------------
# Pre-encoding helper: mark tuples before json.dumps sees the structure
# ---------------------------------------------------------------------------

def _mark_special(o: Any) -> Any:
    """Recursively replace every ``tuple`` with ``{"__tuple__": [...]}``.

    This traversal happens *before* ``JSONEncoder.encode`` so that the
    sentinel dicts are visible to the standard encoder.  ``dict`` values and
    ``list`` elements are also walked so that tuples nested at any depth are
    found.  Non-container objects (including non-serializable ones like numpy
    arrays) are returned unchanged and will later hit ``default()``.
    """
    if isinstance(o, tuple):
        return {'__tuple__': [_mark_special(e) for e in o]}
    if isinstance(o, (set, frozenset)):
        # JSON has no set type; restore as a sorted list (all callers use `in`, so list is fine)
        return sorted(o)
    if isinstance(o, dict):
        return {k: _mark_special(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_mark_special(e) for e in o]
    return o


# ---------------------------------------------------------------------------
# Decoding hook
# ---------------------------------------------------------------------------

def _decode_hook(dct: dict) -> Any:
    """``object_hook`` for :func:`json.loads`.

    The JSON decoder calls this bottom-up (innermost dicts first), so nested
    sentinels are already reconstructed when an outer sentinel is reached —
    no explicit recursion is required here.

    Reconstructs:

    - ``{"__tuple__": [...]}``           → ``tuple``
    - ``{"__ndarray__": [...], ...}``    → ``numpy.ndarray``
    - ``{"__colormap__": {"name": ...}}``→ named :class:`~pyqtgraph.ColorMap`
    - ``{"__colormap__": {...}}``        → anonymous :class:`~pyqtgraph.ColorMap`
    """
    if '__tuple__' in dct:
        # Elements may already be reconstructed tuples/arrays from inner calls
        return tuple(dct['__tuple__'])

    if '__ndarray__' in dct:
        arr = np.array(dct['__ndarray__'])
        dtype = dct.get('dtype')
        if dtype:
            arr = arr.astype(dtype, copy=False)
        return arr

    if '__colormap__' in dct:
        attrs = dct['__colormap__']
        name = attrs.get('name')
        if name and 'pos' not in attrs:
            # Named colormap — reconstruct from the registry
            return pgcm.get(name)
        # pos and color have already been decoded as ndarrays by this point
        return ColorMap(
            pos=attrs['pos'],
            color=attrs['color'],
            mapping=attrs['mapping_mode'],
            name=name,
        )

    return dct


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ParameterJsonEncoder(JSONEncoder):
    """``JSONEncoder`` subclass for pyqtgraph Parameter tree serialization.

    Usage::

        json_str = ParameterJsonEncoder(indent=2).encode(param.saveState())
        state    = json.loads(json_str, object_hook=_decode_hook)

    The encoder handles the small set of types that ``saveState()`` may still
    emit after each Parameter type has already converted its Qt objects to
    primitives.  See the module docstring for the full list.
    """

    def encode(self, o: Any) -> str:
        """Pre-process *o* to mark tuples, then delegate to the standard encoder."""
        return super().encode(_mark_special(o))

    def iterencode(self, o: Any, _one_shot: bool = False):
        """Pre-process *o* to mark tuples before chunk-based encoding (used by ``indent``)."""
        return super().iterencode(_mark_special(o), _one_shot)

    def default(self, o: Any) -> Any:
        """Handle non-serializable objects not covered by ``saveState()`` overrides."""
        if isinstance(o, np.ndarray):
            return {'__ndarray__': o.tolist(), 'dtype': str(o.dtype)}

        # numpy scalar types — promote to Python primitives, no sentinel needed
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)

        if isinstance(o, ColorMap):
            if o.name:
                # Named colormaps: store only the name — far more compact
                return {'__colormap__': {'name': o.name}}
            pos, color = o.getStops()
            # pos and color are ndarrays; they will re-enter default() below
            return {
                '__colormap__': {
                    'pos': pos,
                    'color': color,
                    'mapping_mode': o.mapping_mode,
                    'name': o.name,
                }
            }

        # QIcon has no lossless JSON representation; only string file paths
        # (handled natively) survive a round-trip.  Serialize as null so that
        # saveState() / restoreState() still works for Python-to-Python use.
        if isinstance(o, QtGui.QIcon):
            return None

        return super().default(o)

    def decode(self, json_str: str) -> Any:
        """Decode a JSON string produced by this encoder back to Python objects."""
        return JSONDecoder(object_hook=_decode_hook).decode(json_str)
