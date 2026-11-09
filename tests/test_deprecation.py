"""Unit tests for the generic rename-deprecation machinery in
pyqtgraph._deprecation, independent of pyqtgraph's own registry
(see tests/test_deprecated_names.py for that)."""
import sys
import types

import pytest

from pyqtgraph._deprecation import (
    RenamedSymbol,
    redirect_module,
    renamed_attr_getattr,
)


@pytest.fixture
def new_module():
    # Stands in for the module a symbol was renamed into (e.g. GroupItem.py),
    # registered in sys.modules so importlib.import_module can find it.
    mod = types.ModuleType("dummy_new_module")
    mod.NewName = object()
    mod.OtherPublic = object()
    mod._Private = object()
    sys.modules[mod.__name__] = mod
    yield mod
    del sys.modules[mod.__name__]


def test_renamed_attr_getattr_resolves_old_name(new_module):
    symbols = [RenamedSymbol(old_name="OldName", new_name="NewName", new_module=new_module.__name__)]
    __getattr__ = renamed_attr_getattr("dummy_module", symbols)

    # Using old_name name should emit a warning and resolve to new_name
    with pytest.warns(DeprecationWarning, match="OldName has been renamed to NewName"):
        result = __getattr__("OldName")
    assert result is new_module.NewName


def test_renamed_attr_getattr_unknown_name_raises_attributeerror(new_module):
    # A name that isn't in the registry must behave like a normal missing
    # attribute, not silently redirect anywhere.
    symbols = [RenamedSymbol(old_name="OldName", new_name="NewName", new_module=new_module.__name__)]
    __getattr__ = renamed_attr_getattr("dummy_module", symbols)

    with pytest.raises(AttributeError):
        __getattr__("SomethingElse")


def test_renamed_attr_getattr_duplicate_old_name_raises():
    # Two entries can't claim the same old_name - renamed_attr_getattr
    # wouldn't know which new_name to redirect it to.
    symbols = [
        RenamedSymbol(old_name="OldName", new_name="A", new_module="m1"),
        RenamedSymbol(old_name="OldName", new_name="B", new_module="m2"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        renamed_attr_getattr("dummy_module", symbols)


def test_redirect_module_warns_and_reexports(new_module):
    symbols = [
        RenamedSymbol(
            old_name="OldName",
            new_name="NewName",
            old_module="dummy_old_module",
            new_module=new_module.__name__,
        )
    ]
    # Stands in for globals() of the shim module being redirected.
    namespace = {"__name__": "dummy_old_module"}

    # Warns as soon as the shim is imported, not just when an attribute is
    # looked up on it (a bare `import` wouldn't otherwise touch anything).
    with pytest.warns(DeprecationWarning, match="dummy_old_module has been renamed"):
        redirect_module(namespace, "dummy_old_module", symbols)

    # Old name resolves to the actual new object (not a copy/wrapper), so
    # isinstance/identity checks still work across old and new names.
    assert namespace["OldName"] is new_module.NewName
    # New name is also re-exported, so the shim can be used under either name.
    assert namespace["NewName"] is new_module.NewName
    # The rest of the new module's public API is re-exported too, so the
    # shim doesn't silently drop anything the old file used to expose.
    assert namespace["OtherPublic"] is new_module.OtherPublic
    # Private names must not leak into the shim.
    assert "_Private" not in namespace
    # __all__ lists only the old name(s), mirroring what the old module
    # itself used to export - so `from old_module import *` still behaves
    # the way it always did.
    assert namespace["__all__"] == ["OldName"]


def test_redirect_module_no_matches_raises():
    # old_module_name doesn't match any symbol's old_module - calling
    # redirect_module from the wrong file (or before registering it) is a
    # programming error, not a silent no-op.
    symbols = [RenamedSymbol(old_name="OldName", new_name="NewName", new_module="m")]
    with pytest.raises(ValueError):
        redirect_module({}, "some_other_module", symbols)


def test_redirect_module_duplicate_old_name_raises(new_module):
    # Two symbols moved out of the same old_module can't both claim the
    # same old_name.
    symbols = [
        RenamedSymbol(
            old_name="OldName",
            new_name="NewName",
            old_module="dummy_old_module",
            new_module=new_module.__name__,
        ),
        RenamedSymbol(
            old_name="OldName",
            new_name="OtherPublic",
            old_module="dummy_old_module",
            new_module=new_module.__name__,
        ),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        redirect_module({"__name__": "dummy_old_module"}, "dummy_old_module", symbols)
