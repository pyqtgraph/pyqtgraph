"""
Deprecation source for pyqtgraph's renamed public names.

Adding an entry here is enough to:

- keep the old name working at the top level (``pyqtgraph.OldName``), via
  ``__getattr__`` in ``pyqtgraph/__init__.py``;
- keep the old module path working, via a shim module calling
  ``redirect_module`` (only needed if the file itself also moved);
- cover the rename in ``tests/test_deprecated_names.py``, which is
  parametrized directly off this list.
"""

from ._deprecation import RenamedSymbol

RENAMED_SYMBOLS = [
    RenamedSymbol(
        old_name="ItemSample",
        new_name="SampleItem",
        new_module="pyqtgraph.graphicsItems.LegendItem",
        # no old_module: the class was renamed in place, the file didn't move.
    ),
    RenamedSymbol(
        old_name="ItemGroup",
        new_name="GroupItem",
        old_module="pyqtgraph.graphicsItems.ItemGroup",
        new_module="pyqtgraph.graphicsItems.GroupItem",
    ),
]
