import pyqtgraph.parametertree as pt
from pyqtgraph.parametertree.utils import compare_parameters

import pytest


def test_compare_parameters():

    # Test all types
    from pyqtgraph.examples import _buildParamTypes
    created1 = _buildParamTypes.makeAllParamTypes()
    created2 = _buildParamTypes.makeAllParamTypes()

    assert compare_parameters(created1, created2)

    # Test with different types
    p1 = pt.Parameter.create(name='params', type='group', children=[dict(name='a', type='int', value=1)])
    p2 = pt.Parameter.create(name='params', type='group', children=[dict(name='a', type='float', value=1)])
    assert not compare_parameters(p1, p2)

    # Test with different classes
    p1 = pt.Parameter.create(name='params', type='group', children=[dict(name='a', type='int', value=1), dict(name='b', type='int', value=2)])
    p2 = pt.Parameter.create(name='params', type='group', children=[dict(name='a', type='float', value=1)])
    assert not compare_parameters(p1, p2)

