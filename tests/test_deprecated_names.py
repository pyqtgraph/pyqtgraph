"""Tests for backward-compatible aliases of renamed public names.

Parametrized directly off ``pyqtgraph._deprecated_names.RENAMED_SYMBOLS`` -
adding a rename there is enough for it to be covered here automatically.
"""
import importlib

import pytest

import pyqtgraph as pg
from pyqtgraph._deprecated_names import RENAMED_SYMBOLS


@pytest.mark.parametrize("symbol", RENAMED_SYMBOLS, ids=lambda s: s.old_name)
def test_old_name_accessible_at_top_level(symbol):
    # pyqtgraph.OldName must still work (via __getattr__ in __init__.py)
    # and warn, and resolve to the exact same object as the new name -
    # not a copy - so isinstance/identity checks keep working.
    with pytest.warns(DeprecationWarning, match=f"{symbol.old_name} has been renamed to {symbol.new_name}"):
        old = getattr(pg, symbol.old_name)
    new_module = importlib.import_module(symbol.new_module)
    assert old is getattr(new_module, symbol.new_name)


@pytest.mark.parametrize("symbol", RENAMED_SYMBOLS, ids=lambda s: s.old_name)
def test_new_name_accessible_at_top_level(symbol):
    # The rename must also be usable going forward: pyqtgraph.NewName
    # should work with no warning, straight off the normal import chain.
    new_module = importlib.import_module(symbol.new_module)
    assert getattr(pg, symbol.new_name) is getattr(new_module, symbol.new_name)


@pytest.mark.parametrize(
    "symbol",
    # Only symbols whose file actually moved (old_module set) have a shim
    # module to import - an in-place rename like ItemSample has nothing at
    # its old module path to test here.
    [s for s in RENAMED_SYMBOLS if s.old_module is not None],
    ids=lambda s: s.old_name,
)
def test_old_module_path_still_importable(symbol):
    # Importing the old module path (e.g. pyqtgraph.graphicsItems.ItemGroup)
    # must still work and warn once, since redirect_module runs at import time.
    with pytest.warns(DeprecationWarning, match=f"{symbol.old_module} has been renamed"):
        old_module = importlib.import_module(symbol.old_module)
    new_module = importlib.import_module(symbol.new_module)
    # The shim re-exports the renamed symbol under both its old and new name.
    assert getattr(old_module, symbol.old_name) is getattr(new_module, symbol.new_name)
    assert getattr(old_module, symbol.new_name) is getattr(new_module, symbol.new_name)
