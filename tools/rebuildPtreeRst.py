import os.path
import textwrap

from pyqtgraph.parametertree.Parameter import PARAM_TYPES, _PARAM_ITEM_TYPES


def mkDocs(typeList):
    typDocs = [
    f"""\
    .. autoclass:: {typ.__module__}.{typ.__name__}
       :members:\
    """
    for typ in typeList]
    indented = '\n\n'.join(typDocs)
    return textwrap.dedent(indented)

types = set(PARAM_TYPES.values())
items = [typ.itemClass for typ in PARAM_TYPES.values() if typ.itemClass is not None] \
            + [item for item in _PARAM_ITEM_TYPES.values()]
items = set(items)

doc = f"""\
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