# -*- coding: utf-8 -*-
import pytest
from functools import wraps
from pyqtgraph.parametertree import Parameter


def test_parameter_hasdefault():
    opts = {'name': 'param', 'type': int, 'value': 1}

    # default unspecified
    p = Parameter(**opts)
    assert p.hasDefault()
    assert p.defaultValue() == opts["value"]

    p.setDefault(2)
    assert p.hasDefault()
    assert p.defaultValue() == 2

    # default specified
    p = Parameter(default=0, **opts)
    assert p.hasDefault()
    assert p.defaultValue() == 0

    # default specified as None
    p = Parameter(default=None, **opts)
    assert not p.hasDefault()

def test_add_child():
    p = Parameter.create(name='test', type='group', children=[
        dict(name='ch1', type='bool', value=True),
        dict(name='ch2', type='bool', value=False),
    ])
    with pytest.raises(ValueError):
        p.addChild(dict(name='ch1', type='int', value=0))
    existing = p.child('ch1')
    ch = p.addChild(dict(name='ch1', type='int', value=0), existOk=True)
    assert ch is existing

    ch = p.addChild(dict(name='ch1', type='int', value=0), autoIncrementName=True)
    assert ch.name() == 'ch3'

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

def test_interact():
    Parameter.RUN_DEFAULT = Parameter.RUN_BUTTON
    value = None
    def retain(func):
        """Retain result for post-call analysis"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal value
            value = func(*args, **kwargs)
            return value
        return wrapper

    @retain
    def a(x, y=5):
        return x, y

    with pytest.raises(ValueError):
        Parameter.interact(a)

    host = Parameter.interact(a, x=10)
    for child in 'x', 'y':
        assert child in host.names

    host = Parameter.interact(a, x=10, y={'type': 'list', 'limits': [5, 10]})
    testParam = host.child('y')
    assert testParam.type() == 'list'
    assert testParam.opts['limits'] == [5,10]

    myval = 5
    host = Parameter.interact(a, deferred=dict(x=lambda: myval))
    assert 'x' not in host.names
    host.child('Run').activate()
    assert value == (5, 5)
    myval = 10
    host.child('Run').activate()
    assert value == (10, 5)

    p = Parameter
    host = Parameter.interact(a, x=10, y=50, ignores=['x'], runOpts=(p.RUN_CHANGED, p.RUN_CHANGING))
    for child in 'x', 'Run':
        assert child not in host.names

    host['y'] = 20
    assert value == (10, 20)
    host.child('y').sigValueChanging.emit(host.child('y'), 100)
    assert value == (10, 100)

    oldFmt = p.RUN_TITLE_FORMAT
    try:
        p.RUN_TITLE_FORMAT = lambda name: name.upper()
        host = p.interact(a, x={'title': 'different', 'value': 5})
        titles = [p.title() for p in host]
        for ch in 'different', 'Y':
            assert ch in titles
    finally:
        p.RUN_TITLE_FORMAT = oldFmt

    oldDflt = p.RUN_DEFAULT
    try:
        p.RUN_DEFAULT = p.RUN_CHANGED
        host = p.interact(a, x=5)
        host['y'] = 20
        assert value == (5, 20)
    finally:
        p.RUN_DEFAULT = oldDflt

    @retain
    def kwargTest(a, b=5, **c):
        return a + b - c.get('test', None)
    host = p.interact(kwargTest, a=10, test=3)
    for ch in 'a', 'b', 'test':
        assert ch in host.names
    host.child('Run').activate()
    assert value == 12

    host = Parameter.create(name='test deco', type='group')
    @host.interact_decorator()
    @retain
    def a(x=5):
        return x
    assert 'a' in host.names
    assert 'x' in host.child('a').names
    host.child('a', 'Run').activate()
    assert value == 5

    @host.interact_decorator(nest=False, runOpts=p.RUN_CHANGED)
    @retain
    def b(y=6):
        return y
    assert 'b' not in host.names
    assert 'y' in host.names
    host['y'] = 7
    assert value == 7

    def raw(x=5):
        return x

    @retain
    def override(**kwargs):
        return raw(**kwargs)
    host = p.interact(raw, runFunc=override, runOpts=p.RUN_CHANGED)
    assert 'x' in host.names
    host['x'] = 100
    assert value == 100
