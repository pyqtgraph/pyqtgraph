# -*- coding: utf-8 -*-
from contextlib import ExitStack

import pytest
from functools import wraps
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter as GP
from pyqtgraph.parametertree.interactive import interact, RunOpts, InteractiveFunction


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
    RunOpts.setOpts(runOpts=RunOpts.ON_BUTTON)
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
        interact(a)

    host = interact(a, x=10)
    for child in 'x', 'y':
        assert child in host.names

    host = interact(a, x=10, y={'type': 'list', 'limits': [5, 10]})
    testParam = host.child('y')
    assert testParam.type() == 'list'
    assert testParam.opts['limits'] == [5,10]

    myval = 5
    a_interact = InteractiveFunction(a, deferred=dict(x=lambda: myval))
    host = interact(a_interact)
    assert 'x' not in host.names
    host.child('Run').activate()
    assert value == (5, 5)
    myval = 10
    host.child('Run').activate()
    assert value == (10, 5)

    host = interact(a, x=10, y=50, ignores=['x'], runOpts=(RunOpts.ON_CHANGED, RunOpts.ON_CHANGING))
    for child in 'x', 'Run':
        assert child not in host.names

    host['y'] = 20
    assert value == (10, 20)
    host.child('y').sigValueChanging.emit(host.child('y'), 100)
    assert value == (10, 100)

    with RunOpts.optsContext(title=str.upper):
        host = interact(a, x={'title': 'different', 'value': 5})
        titles = [p.title() for p in host]
        for ch in 'different', 'Y':
            assert ch in titles

    with RunOpts.optsContext(runOpts=RunOpts.ON_CHANGED):
        host = interact(a, x=5)
        host['y'] = 20
        assert value == (5, 20)

    @retain
    def kwargTest(a, b=5, **c):
        return a + b - c.get('test', None)
    host = interact(kwargTest, a=10, test=3)
    for ch in 'a', 'b', 'test':
        assert ch in host.names
    host.child('Run').activate()
    assert value == 12

    host = GP.create(name='test deco', type='group')
    @host.interactDecorator()
    @retain
    def a(x=5):
        return x
    assert 'a' in host.names
    assert 'x' in host.child('a').names
    host.child('a', 'Run').activate()
    assert value == 5

    @host.interactDecorator(nest=False, runOpts=RunOpts.ON_CHANGED)
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
    host = interact(raw, runFunc=override, runOpts=RunOpts.ON_CHANGED)
    assert 'x' in host.names
    host['x'] = 100
    assert value == 100

    RunOpts.setOpts(runOpts=None)

def test_onlyRun():
    def a():
        return 5
    assert not isinstance(interact(a, runOpts=RunOpts.ON_BUTTON), GP)

def test_interactiveFunc():
    value = 0
    def myfunc(a=5):
        nonlocal value
        value = a
        return a
    interactive = InteractiveFunction(myfunc)
    host = interact(interactive, runOpts=[])

    host['a'] = 7
    assert interactive.runFromButton() == 7

    interactive.disconnect()
    interactive.runFromButton(a=10)
    assert value == 7

    interactive.reconnect()
    interactive.runFromButton(a=10)
    assert value == 10

def test_badOptsContext():
    with pytest.raises(KeyError):
        with RunOpts.optsContext(bad=4):
            pass
