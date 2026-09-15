"""
Generic machinery for deprecating renamed classes, functions, and modules.

Driven entirely by a list of `RenamedSymbol` entries the caller supplies
(typically its own registrymodule).
Two situations:
1. `renamed_attr_getattr` - a rename that keeps the same module: the old
   name works via ``__getattr__`` (PEP 562), warning only when looked up.
2. `redirect_module` - a rename that also moves the file: the old module
   becomes a shim that warns on import and re-exports the new module's
   public API under both names.
"""

import importlib
import warnings
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RenamedSymbol:
    """One renamed public symbol. Always constructed with keyword arguments.

    old_name / new_name : the symbol's name before and after the rename.
    old_module : dotted path of a compatibility shim left at the previous
        module path, or None if only the name (not the file) moved.
    new_module : dotted path of the module that now defines it.
    """
    old_name: str
    new_name: str
    old_module: str | None = None
    new_module: str


def _warn_renamed(old, new):
    warnings.warn(
        f"{old} has been renamed to {new} and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


def renamed_attr_getattr(module_name, symbols):
    """Build a module-level ``__getattr__`` serving old names from ``symbols``.

    Define at module scope: ``__getattr__ = renamed_attr_getattr(__name__, symbols)``.
    Raises ``ValueError`` if two entries share an ``old_name``.
    """
    by_old_name = {}
    for s in symbols:
        if s.old_name in by_old_name:
            raise ValueError(
                f"duplicate RenamedSymbol.old_name {s.old_name!r} in registry "
                f"passed to renamed_attr_getattr({module_name!r}, ...)"
            )
        by_old_name[s.old_name] = s

    def __getattr__(name):
        symbol = by_old_name.get(name)
        if symbol is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        _warn_renamed(name, symbol.new_name)
        new_module = importlib.import_module(symbol.new_module)
        return getattr(new_module, symbol.new_name)

    return __getattr__


def redirect_module(namespace, old_module_name, symbols):
    """Turn the calling module into a compatibility shim.

    Call once, at the top of a module whose contents moved elsewhere:
    ``redirect_module(globals(), __name__, symbols)``. Matches every entry
    whose ``old_module`` equals ``old_module_name`` (there can be several,
    if multiple classes moved out of one file), warns once on import, then
    re-exports the matched new module(s)' public API plus each symbol under
    its old name.

    ``__all__`` lists only the old names, mirroring what the old module
    itself used to export - other public names stay reachable via attribute
    access or explicit import. Raises ``ValueError`` on a duplicate
    ``old_name``; a same-named attribute shared by two different
    ``new_module``s resolves to whichever is later in ``symbols``.
    """
    matches = [s for s in symbols if s.old_module == old_module_name]
    if not matches:
        raise ValueError(f"no RenamedSymbol registered with old_module={old_module_name!r}")

    old_names = [s.old_name for s in matches]
    if len(set(old_names)) != len(old_names):
        raise ValueError(
            f"duplicate RenamedSymbol.old_name among entries for old_module={old_module_name!r}"
        )

    new_modules = list(dict.fromkeys(s.new_module for s in matches))  # first-seen order
    _warn_renamed(old_module_name, ", ".join(new_modules))

    for symbol in matches:
        new_module = importlib.import_module(symbol.new_module)
        namespace.update(
            (k, v) for k, v in vars(new_module).items() if not k.startswith('_')
        )
        namespace[symbol.old_name] = getattr(new_module, symbol.new_name)
    namespace['__all__'] = old_names
