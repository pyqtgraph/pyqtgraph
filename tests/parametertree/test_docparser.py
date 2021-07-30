import sys

import numpy as np
import pytest

from pyqtgraph.parametertree.parameterTypes import GroupParameter as GP

# ------
# Define various functions whose signatures all mean the same thing and who should be parsed without fault
# -----
# Also define simple decorator to ease test registration
TO_RUN = []

def addToSuite(func):
    TO_RUN.append(func)
    return func


@addToSuite
def normalFmt_oneLineDoc(x=5.0, y=6.0):
    """
    Prints the values of x and y.

    [x.options]
    tip='The X parameter'
    limits=[0, 10]
    step=0.1

    [y.options]
    tip='The Y parameter'
    step=0.1
    """
    return x, y


@addToSuite
def normalFmt_noDoc(x=5.0, y=6.0):
    """
    [x.options]
    tip='The X parameter'
    limits=[0, 10]
    step=0.1

    [y.options]
    tip='The Y parameter'
    step=0.1
    """
    return x, y


@addToSuite
def normalFmt_noDoc_shouldOverrideSig(x=3, y=4):
    """
    [x.options]
    tip='The X parameter'
    limits=[0, 10]
    step=0.1
    type=float
    value=5

    [y.options]
    tip='The Y parameter'
    step=0.1
    value=6
    type=float
    """
    return x, y


@addToSuite
def numpyFmt(x=5.0, y=6.0):
    """
    Prints the values of x and y.

    Parameters
    ----------
    x: float
        The X parameter.

        [x.options]
        limits=[0, 10]
        step=0.1
        tip='The X parameter'

    y: float
        The Y parameter.

        [y.options]
        step=0.1
        tip='The Y parameter'
    """
    return x, y


@addToSuite
def numpyFmt_noHeader(x=5.0, y=6.0):
    """
    Prints the values of x and y.

    Parameters
    ----------
    x: float
        The X parameter.
        limits=[0, 10]
        step=0.1
        tip='The X parameter'

    y: float
        The Y parameter.
        step=0.1
        tip='The Y parameter'
    """
    return x, y


@addToSuite
def rstFmt(x=5.0, y=6.0):
    """
    Prints the values of x and y.

    :param x: The X parameter
    [x.options]
    limits=[0, 10]
    step=0.1
    tip='The X parameter'

    :param y: The Y parameter
    [y.options]
    step=0.1
    tip='The Y parameter'
    """
    return x, y


@addToSuite
def rstFmt_noHeaders(x=5.0, y=6.0):
    """
    Prints the values of x and y.

    :param x: The X parameter
    limits=[0, 10]
    step=0.1

    :param y: The Y parameter
    step=0.1
    """
    return x, y


@pytest.mark.parametrize('xyFunc', [rstFmt_noHeaders])
def test_docparsing(xyFunc):
    def argCollector(x, y):
        assert x, y == (5.0, 6.0)
    param = GP.interact(xyFunc, runOpts=GP.RUN_BUTTON, runFunc=argCollector)
    param.child('Run').activate()
    for name in 'y', 'x':
        ch = param.child(name)
        assert ch.opts['tip'] == f'The {name.upper()} parameter'
        assert ch.opts['type'] == 'float'
        assert ch.opts['step'] == 0.1
    # ch is 'x'
    assert ch.opts['limits'] == [0, 10]


def test_docstringFailure():
    def a(x=3):
        """
        [x.options]
        value = 5
        tip='a tip that won\'t appear'
        [x.options]
        value = 6
        """
        # Should still parse, but won't set the value
        return x
    param = GP.interact(a)
    assert param['x'] == 3
    assert 'tip' not in param.child('x').opts


def test_userOverride():
    def a(x=3):
        """
        [x.options]
        value = 6
        """
        return x
    for extra in 9, {'value': 9}:
        param = GP.interact(a, x=extra)
        assert param['x'] == 9


def test_evalFallback():
    def a(x='6'):
        """
        [x.options]
        value = [5badname]
        """
    param = GP.interact(a)
    assert param['x'] == '[5badname]'


def test_commonNumpyFuncs():
    for param in (
            GP.interact(np.ones, shape=[1,2], dtype={'type': 'str'}, ignores=['like']),
            GP.interact(np.linspace, start=1, stop=2, dtype={'type': 'str'})
    ):
        for ch in param.children():
            assert ch.opts['tip']


def test_noDocstringParserFallback(monkeypatch):
    monkeypatch.setitem(sys.modules, 'docstring_parser', None)
    for ch in GP.interact(rstFmt_noHeaders):
        assert 'tip' not in ch.opts