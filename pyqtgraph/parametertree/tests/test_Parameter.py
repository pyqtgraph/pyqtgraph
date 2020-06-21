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
