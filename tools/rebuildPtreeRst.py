import os.path
import textwrap

from pyqtgraph.parametertree.Parameter import PARAM_TYPES, _PARAM_ITEM_TYPES


def mkDocs(typeList):
    typeNames = sorted([typ.__name__ for typ in typeList])
    typDocs = [
    f"""\
    .. autoclass:: {name}
       :members:
    """
    for name in typeNames]
    indented = '\n'.join(typDocs)
    # There will be two newlines at the end, so remove one
    return textwrap.dedent(indented)[:-1]

types = set(PARAM_TYPES.values())
items = [typ.itemClass for typ in PARAM_TYPES.values() if typ.itemClass is not None] \
            + [item for item in _PARAM_ITEM_TYPES.values()]
items = set(items)

doc = f"""\
..
  This file is auto-generated from pyqtgraph/tools/rebuildPtreeRst.py. Do not modify by hand! Instead, rerun the
  generation script with `python pyqtgraph/tools/rebuildPtreeRst.py`.

Built-in Parameter Types
========================

.. currentmodule:: pyqtgraph.parametertree.parameterTypes

Parameters
----------

{mkDocs(types)}

ParameterItems
--------------

{mkDocs(items)}
"""

here = os.path.dirname(__file__)
rstFilename = os.path.join(here, '..', 'doc', 'source', 'parametertree', 'parametertypes.rst')
with open(rstFilename, 'w') as ofile:
    ofile.write(doc)
