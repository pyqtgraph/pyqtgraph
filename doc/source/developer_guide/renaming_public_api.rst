Renaming a public class or function
====================================

PyQtGraph is gradually cleaning up inconsistent naming across its public
API (see `issue #3628 <https://github.com/pyqtgraph/pyqtgraph/issues/3628>`_).
Renaming something that users may already import means the old name has to
keep working - with a ``DeprecationWarning`` - until it's removed in a
future release.

This is handled by a small generic mechanism in
:mod:`pyqtgraph._deprecation`, driven entirely by the registry in
``pyqtgraph/_deprecated_names.py``. To rename a public symbol:

1. Rename the class/function itself, and update every internal usage in
   pyqtgraph to the new name.
2. Add a :class:`~pyqtgraph._deprecation.RenamedSymbol` entry to
   ``RENAMED_SYMBOLS`` in ``pyqtgraph/_deprecated_names.py``, giving the
   old name, new name, and the module that now defines it.
3. If the file itself also moved (not just the name), leave a shim behind
   at the old module path: a file containing only

   .. code-block:: python

       from .._deprecated_names import RENAMED_SYMBOLS
       from .._deprecation import redirect_module

       redirect_module(globals(), __name__, RENAMED_SYMBOLS)

   This keeps ``import pyqtgraph.graphicsItems.OldModule`` working, warns
   once on import, and re-exports the new module's public API under both
   the old and new names.
4. If the rename happened in place (same module, only the name changed),
   add the same three-line ``__getattr__`` hookup at the bottom of that
   module instead, using :func:`~pyqtgraph._deprecation.renamed_attr_getattr`:

   .. code-block:: python

       from .._deprecated_names import RENAMED_SYMBOLS
       from .._deprecation import renamed_attr_getattr

       __getattr__ = renamed_attr_getattr(__name__, RENAMED_SYMBOLS)

That's it - adding the entry to ``RENAMED_SYMBOLS`` is what makes the old
name resolve at the top level (``pyqtgraph.OldName``, via the
``__getattr__`` already wired up in ``pyqtgraph/__init__.py``) and what
pulls the rename into ``tests/test_deprecated_names.py``, which is
parametrized directly off the registry - no test needs to be written by
hand for a new rename.
