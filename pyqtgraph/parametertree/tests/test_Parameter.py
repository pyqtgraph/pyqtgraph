# -*- coding: utf-8 -*-
import pytest
from pyqtgraph.parametertree import Parameter


def test_parameter_hasdefault():
    opts = {'name': 'param', 'type': int, 'value': 1}

    # default unspecified
    p = Parameter(**opts)
    assert not p.hasDefault()

    p.setDefault(1)
    assert p.hasDefault()
    assert p.defaultValue() == 1

    # default specified
    p = Parameter(default=0, **opts)
    assert p.hasDefault()
    assert p.defaultValue() == 0


@pytest.mark.parametrize('passdefault', [True, False])
def test_parameter_hasdefault_none(passdefault):
    # test that Parameter essentially ignores defualt=None, same as not passing
    # a default at all
    opts = {'name': 'param', 'type': int, 'value': 0}
    if passdefault:
        opts['default'] = None

    p = Parameter(**opts)
    assert not p.hasDefault()
    assert p.defaultValue() is None

    p.setDefault(None)
    assert not p.hasDefault()

def test_unpack_parameter():
    # test that **unpacking correctly returns child name/value maps
    params = [
        dict(name='a', type='int', value=1),
        dict(name='b', type='str', value='2'),
        dict(name='c', type='float', value=3.0),
    ]
    p = Parameter.create(name='params', type='group', children=params)
    result = dict(**p)

    assert 'a' in result
    assert result['a'] == 1
    assert 'b' in result
    assert result['b'] == '2'
    assert 'c' in result
    assert result['c'] == 3.0
