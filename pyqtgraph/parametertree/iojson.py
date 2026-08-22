"""JSON serialization for pyqtgraph Parameter trees.

Intended as a companion to ``saveState()`` / ``restoreState()``: the same
round-trip guarantee, stored as a human-readable JSON file instead of an
in-memory dict.

Design
------
``saveState()`` is the canonical serializer.  Every built-in Parameter
subclass already overrides it to return JSON-primitive values::

    ColorParameter    → (r, g, b, a) tuple
    FontParameter     → font-description string
    CalendarParameter → date string
    PenParameter      → tuple of (color, width, style, capStyle, joinStyle, cosmetic)
    QtEnumParameter   → enum-member name string

This module only needs to handle the small set of types that ``saveState()``
still emits but that the standard ``json`` module cannot encode — see
:class:`~pyqtgraph.parametertree.utils.ParameterJsonEncoder` for the full list.

Two workflows
-------------
**Full-structure round-trip** — clone or migrate a tree:

    json_str = parameter_to_json(param)
    clone    = parameter_from_json(json_str)

**User-settings workflow** — save only user-modified values, reload into an
existing tree while preserving widget connections and signal handlers:

    # Save (only values the user changed)
    parameter_to_json_file(param, 'settings.json', filter='user')

    # Load back into the same tree later
    parameter_restore_from_json_file(param, 'settings.json')

Public API
----------
``parameter_to_json(param, indent=None, filter=None)``       → ``str``
``parameter_from_json(json_str)``                            → ``Parameter``
``parameter_restore_from_json(param, json_str)``
``parameter_to_json_file(param, path, *, overwrite=True, indent=2, encoding=None, filter=None)``
``parameter_from_json_file(path, encoding=None)``            → ``Parameter``
``parameter_restore_from_json_file(param, path, encoding=None)``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Union

from .Parameter import Parameter
from .utils import ParameterJsonEncoder, _decode_hook

__all__ = [
    'parameter_to_json',
    'parameter_from_json',
    'parameter_restore_from_json',
    'parameter_to_json_file',
    'parameter_from_json_file',
    'parameter_restore_from_json_file',
]

# RFC 8259 mandates UTF-8 for JSON interchange; used as the default when no
# encoding is explicitly requested.
_JSON_ENCODING = 'utf-8'

_Filter = Literal['user'] | None


# ---------------------------------------------------------------------------
# String API
# ---------------------------------------------------------------------------

def parameter_to_json(
    param: Parameter,
    indent: int | None = None,
    filter: _Filter = None,
) -> str:
    """Serialize a Parameter tree to a JSON string.

    Calls :meth:`~pyqtgraph.parametertree.Parameter.saveState` internally so
    that each Parameter subclass's own ``saveState()`` override handles Qt-
    object conversion before the encoder sees the data.

    Parameters
    ----------
    param:
        Root of the Parameter tree (or any sub-tree node).
    indent:
        ``None`` (default) produces a compact single-line string; an integer
        such as ``2`` produces human-readable indented output.
    filter:
        Passed to ``saveState()``.  ``None`` (default) saves the full tree
        structure including type, limits, and all opts — suitable for cloning
        or migrating a tree.  ``'user'`` saves only user-settable values,
        producing a compact settings file that can be reloaded into an
        existing tree via :func:`parameter_restore_from_json`.

    Returns
    -------
    str
        A JSON string representing the parameter tree.
    """
    return ParameterJsonEncoder(indent=indent).encode(param.saveState(filter=filter))


def parameter_from_json(json_str: str) -> Parameter:
    """Reconstruct a Parameter tree from a full-structure JSON string.

    Parameters
    ----------
    json_str:
        A JSON string produced by :func:`parameter_to_json` (with
        ``filter=None``) or read from a file written by
        :func:`parameter_to_json_file`.

    Returns
    -------
    Parameter
        A fully reconstructed Parameter tree.

    See Also
    --------
    parameter_restore_from_json : load user-settings into an existing tree.
    """
    state = json.loads(json_str, object_hook=_decode_hook)
    return Parameter.create(**state)


def parameter_restore_from_json(param: Parameter, json_str: str) -> None:
    """Restore parameter values from a JSON string into an *existing* tree.

    Uses :meth:`~pyqtgraph.parametertree.Parameter.restoreState` so that all
    widget connections and signal handlers on *param* are preserved.  Intended
    for loading user settings saved with ``filter='user'``, but also works
    with full-structure JSON.

    Parameters
    ----------
    param:
        The existing Parameter tree to update.
    json_str:
        A JSON string produced by :func:`parameter_to_json`.
    """
    state = json.loads(json_str, object_hook=_decode_hook)
    param.restoreState(state)


# ---------------------------------------------------------------------------
# File API
# ---------------------------------------------------------------------------

def parameter_to_json_file(
    param: Parameter,
    path: Union[str, Path],
    *,
    overwrite: bool = True,
    indent: int = 2,
    encoding: str | None = None,
    filter: _Filter = None,
) -> None:
    """Serialize a Parameter tree to a ``.json`` file.

    Parameters
    ----------
    param:
        Root of the Parameter tree (or any sub-tree node).
    path:
        Destination path.  The ``.json`` extension is enforced regardless of
        whatever suffix *path* already carries.
    overwrite:
        When ``False`` and the destination file already exists, raises
        :exc:`FileExistsError` instead of overwriting.
    indent:
        Indentation used in the output file (default ``2``).  Pass ``None``
        for a compact single-line file.
    encoding:
        Text encoding for the file.  Defaults to ``'utf-8'`` (RFC 8259).
    filter:
        Passed to ``saveState()``.  ``'user'`` saves only user-settable values
        (recommended for settings files); ``None`` saves the full structure.
    """
    path = Path(path).with_suffix('.json')
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} already exists")
    path.write_text(
        ParameterJsonEncoder(indent=indent).encode(param.saveState(filter=filter)),
        encoding=encoding or _JSON_ENCODING,
    )


def parameter_from_json_file(
    path: Union[str, Path],
    encoding: str | None = None,
) -> Parameter:
    """Reconstruct a Parameter tree from a full-structure ``.json`` file.

    Parameters
    ----------
    path:
        Path to the ``.json`` file written by :func:`parameter_to_json_file`
        with ``filter=None``.
    encoding:
        Text encoding of the file.  Defaults to ``'utf-8'`` (RFC 8259).

    Returns
    -------
    Parameter
        A fully reconstructed Parameter tree.

    See Also
    --------
    parameter_restore_from_json_file : load user-settings into an existing tree.
    """
    text = Path(path).read_text(encoding=encoding or _JSON_ENCODING)
    state = json.loads(text, object_hook=_decode_hook)
    return Parameter.create(**state)


def parameter_restore_from_json_file(
    param: Parameter,
    path: Union[str, Path],
    encoding: str | None = None,
) -> None:
    """Restore parameter values from a ``.json`` file into an *existing* tree.

    Uses :meth:`~pyqtgraph.parametertree.Parameter.restoreState` so that all
    widget connections and signal handlers on *param* are preserved.  Intended
    for loading user settings saved with ``filter='user'``, but also works
    with full-structure JSON.

    Parameters
    ----------
    param:
        The existing Parameter tree to update.
    path:
        Path to the ``.json`` file written by :func:`parameter_to_json_file`.
    encoding:
        Text encoding of the file.  Defaults to ``'utf-8'`` (RFC 8259).
    """
    text = Path(path).read_text(encoding=encoding or _JSON_ENCODING)
    state = json.loads(text, object_hook=_decode_hook)
    param.restoreState(state)
