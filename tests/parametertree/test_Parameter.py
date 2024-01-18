from functools import wraps

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.parametertree import (
    InteractiveFunction,
    Interactor,
    Parameter,
    RunOptions,
    interact,
)
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.parameterTypes import GroupParameter as GP
from pyqtgraph.Qt import QtGui

pg.mkQApp()

def test_parameter_hasdefault():
    opts = {"name": "param", "type": int, "value": 1}

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


def test_parameter_pinning_and_defaults():
    p = Parameter(
        name="param", type=int, value=1, default=1, pinValueToDefault=True, treatInitialValueAsModified=False
    )
    assert p.valueModifiedSinceResetToDefault() is False
    p.setValue(2)
    assert p.valueModifiedSinceResetToDefault() is True
    p.setToDefault()
    assert p.valueModifiedSinceResetToDefault() is False
    p.setValue(3)
    p.setValue(1)
    assert p.valueModifiedSinceResetToDefault() is True
    p.setToDefault()
    p.setDefault(2)
    assert p.valueModifiedSinceResetToDefault() is False
    assert p.value() == 2

    p = Parameter(name="param", type=int, value=1, default=1)
    assert p.valueModifiedSinceResetToDefault() is False

    p = Parameter(name="param", type=int, value=1, default=2, treatInitialValueAsModified=False)
    assert p.valueModifiedSinceResetToDefault() is True

    p = Parameter(name="param", type=int, default=1, treatInitialValueAsModified=False)
    assert p.valueModifiedSinceResetToDefault() is True

    p = Parameter(name="param", type=int, value=1, default=1, treatInitialValueAsModified=True)
    assert p.valueModifiedSinceResetToDefault() is True


def test_add_child():
    p = Parameter.create(
        name="test",
        type="group",
        children=[
            dict(name="ch1", type="bool", value=True),
            dict(name="ch2", type="bool", value=False),
        ],
    )
    with pytest.raises(ValueError):
        p.addChild(dict(name="ch1", type="int", value=0))
    existing = p.child("ch1")
    ch = p.addChild(dict(name="ch1", type="int", value=0), existOk=True)
    assert ch is existing

    ch = p.addChild(dict(name="ch1", type="int", value=0), autoIncrementName=True)
    assert ch.name() == "ch3"


def test_unpack_parameter():
    # test that **unpacking correctly returns child name/value maps
    params = [
        dict(name="a", type="int", value=1),
        dict(name="b", type="str", value="2"),
        dict(name="c", type="float", value=3.0),
    ]
    p = Parameter.create(name="params", type="group", children=params)
    result = dict(**p)

    assert "a" in result
    assert result["a"] == 1
    assert "b" in result
    assert result["b"] == "2"
    assert "c" in result
    assert result["c"] == 3.0


def test_interact():
    interactor = Interactor(runOptions=RunOptions.ON_ACTION)
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
        interactor(a)

    host = interactor(a, x=10)
    for child in "x", "y":
        assert child in host.names

    host = interactor(a, x=10, y={"type": "list", "limits": [5, 10]})
    testParam = host.child("y")
    assert testParam.type() == "list"
    assert testParam.opts["limits"] == [5, 10]

    myval = 5
    a_interact = InteractiveFunction(a, closures=dict(x=lambda: myval))
    host = interactor(a_interact)
    assert "x" not in host.names
    host.activate()
    assert value == (5, 5)
    myval = 10
    host.activate()
    assert value == (10, 5)

    host = interactor(
        a,
        x=10,
        y=50,
        ignores=["x"],
        runOptions=(RunOptions.ON_CHANGED, RunOptions.ON_CHANGING),
    )
    for child in "x", "Run":
        assert child not in host.names

    host["y"] = 20
    assert value == (10, 20)
    host.child("y").sigValueChanging.emit(host.child("y"), 100)
    assert value == (10, 100)

    with interactor.optsContext(titleFormat=str.upper):
        host = interactor(a, x={"title": "different", "value": 5})
        titles = [p.title() for p in host]
        for ch in "different", "Y":
            assert ch in titles

    with interactor.optsContext(titleFormat="Group only"):
        host = interactor(a, x=1)
        assert host.title() == "Group only"
        assert [p.title() is None for p in host]

    with interactor.optsContext(runOptions=RunOptions.ON_CHANGED):
        host = interactor(a, x=5)
        host["y"] = 20
        assert value == (5, 20)
        assert "Run" not in host.names

    @retain
    def kwargTest(a, b=5, **c):
        return a + b - c.get("test", None)

    host = interactor(kwargTest, a=10, test=3)
    for ch in "a", "b", "test":
        assert ch in host.names
    host.activate()
    assert value == 12

    host = GP.create(name="test deco", type="group")
    interactor.setOpts(parent=host)

    @interactor.decorate()
    @retain
    def a(x=5):
        return x

    assert "a" in host.names
    assert "x" in host.child("a").names
    host.child("a").activate()
    assert value == 5

    @interactor.decorate(nest=False, runOptions=RunOptions.ON_CHANGED)
    @retain
    def b(y=6):
        return y

    assert "b" not in host.names
    assert "y" in host.names
    host["y"] = 7
    assert value == 7

    def raw(x=5):
        return x

    @retain
    def override(**kwargs):
        return raw(**kwargs)

    host = interactor(wraps(raw)(override), runOptions=RunOptions.ON_CHANGED)
    assert "x" in host.names
    host["x"] = 100
    assert value == 100


def test_run():
    def a():
        """"""

    interactor = Interactor(runOptions=RunOptions.ON_ACTION)

    defaultRunBtn = Parameter.create(**interactor.runActionTemplate, name="Run")
    group = interactor(a)
    assert group.makeTreeItem(0).button.text() == defaultRunBtn.name()

    template = dict(defaultName="Test", type="action")
    with interactor.optsContext(runActionTemplate=template):
        x = interactor(a)
    assert x.makeTreeItem(0).button.text() == "Test"

    parent = Parameter.create(name="parent", type="group")
    test2 = interactor(a, parent=parent, nest=False)
    assert (
        len(test2) == 1
        and test2[0].name() == a.__name__
        and test2[0].parent() is parent
    )

    test2 = interactor(a, nest=False)
    assert len(test2) == 1 and not test2[0].parent()


def test_no_func_group():
    def inner(a=5, b=6):
        return a + b

    out = interact(inner, nest=False)
    assert isinstance(out, list)


def test_tips():
    def a():
        """a simple tip"""

    interactor = Interactor()

    group = interactor(a, runOptions=RunOptions.ON_ACTION)
    assert group.opts["tip"] == a.__doc__ and group.type() == "_actiongroup"

    params = interactor(a, runOptions=RunOptions.ON_ACTION, nest=False)
    assert len(params) == 1 and params[0].opts["tip"] == a.__doc__

    def a2(x=5):
        """
        A long docstring with a newline
        followed by more text won't result in a tooltip
        """

    param = interactor(a2)
    assert "tip" not in param.opts


def test_interactiveFunc():
    value = 0

    def myfunc(a=5):
        nonlocal value
        value = a
        return a

    interactive = InteractiveFunction(myfunc)
    host = interact(interactive, runOptions=[])

    host["a"] = 7
    assert interactive.runFromAction() == 7

    interactive.disconnect()
    interactive.runFromAction(a=10)
    assert value == 7

    interactive.reconnect()
    interactive.runFromAction(a=10)
    assert value == 10

    assert not interactive.setDisconnected(True)
    assert interactive.setDisconnected(False)

    host = interact(interactive, runOptions=RunOptions.ON_CHANGED)
    interactive.disconnect()
    host["a"] = 20
    assert value == 10


def test_badOptsContext():
    with pytest.raises(KeyError):
        Interactor(bad=4)


def test_updateParamDuringRun():
    counter = 0

    @InteractiveFunction
    def func(a=1, ignored=2):
        nonlocal counter
        counter += a

    param = interact(func, ignores=["ignored"])
    func.parametersNeedRunKwargs = True

    func(a=3, ignored=4)
    # Ensure "test" was only run once
    assert counter == 3
    assert param["a"] == 3
    assert func.extra["ignored"] == 4

    func.parametersNeedRunKwargs = False
    func(a=1)
    assert counter == 4
    assert param["a"] == 3


def test_remove_params():
    class RetainVal:
        a = 1

    @InteractiveFunction
    def inner(a=4):
        RetainVal.a = a

    host = interact(inner, runOptions=RunOptions.ON_CHANGED)
    host["a"] = 5
    assert RetainVal.a == 5

    inner.removeParameters()
    host["a"] = 6
    assert RetainVal.a == 5


def test_interactive_reprs():
    inter = Interactor()
    assert str(inter.getOpts()) in repr(inter)

    ifunc = InteractiveFunction(lambda x=5: x, closures=dict(x=lambda: 10))
    assert "closures=['x']" in repr(ifunc)


def test_rm_without_clear_cache():
    class RetainVal:
        a = 1

    host = Parameter.create(name="host", type="group")
    interactor = Interactor(parent=host, nest=False)

    @interactor.decorate(a=9)
    def inner(a=4):
        RetainVal.a = a

    inner.removeParameters(clearCache=False)
    host["a"] = 6
    assert RetainVal.a == 1

    inner()
    assert RetainVal.a == 9

    inner.removeParameters(clearCache=True)
    inner()
    assert RetainVal.a == 4


def test_decorate_already_interactive():
    @InteractiveFunction
    def inner(a=4):
        return a

    in1 = inner
    in2 = interact.decorate()(inner)
    assert in1 is in2


def test_update_non_param_kwarg():
    class RetainVal:
        a = 1

    @InteractiveFunction
    def a(x=3, **kwargs):
        RetainVal.a = sum(kwargs.values()) + x
        return RetainVal.a

    a.parametersNeedRunKwargs = True

    host = interact(a)
    assert a(y=10) == 13
    assert len(host.names) == 1 and host["x"] == 3

    assert a() == 3

    # Code path where "propagateParamChanges" shouldn't be reconnected if
    # it's already disconnected
    a.disconnect()
    assert a(y=10) == 13
    host["x"] = 5
    assert RetainVal.a == 13

    # But the cache should still be up-to-date
    assert a() == 5


def test_hookup_extra_params():
    @InteractiveFunction
    def a(x=5, **kwargs):
        return x + sum(kwargs.values())

    interact(a)

    p2 = Parameter.create(name="p2", type="int", value=3)
    a.hookupParameters([p2], clearOld=False)

    assert a() == 8


def test_class_interact():
    parent = Parameter.create(name="parent", type="group")
    interactor = Interactor(parent=parent, nest=False)

    def outside_class_deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    class A:
        def a(self, x=5):
            return x

        @classmethod
        def b(cls, y=5):
            return y

        @outside_class_deco
        def c(self, z=5):
            return z

    a = A()
    ai = interactor.decorate()(a.a)
    assert ai() == a.a()

    bi = interactor.decorate()(A.b)
    assert bi() == A.b()

    ci = interactor.decorate()(a.c)
    assert ci() == a.c()


def test_args_interact():
    @interact.decorate()
    def a(*args):
        """"""

    assert not (a.parameters or a.extra)
    a()


def test_interact_with_icon():
    randomPixmap = QtGui.QPixmap(64, 64)
    randomPixmap.fill(QtGui.QColor("red"))

    parent = Parameter.create(name="parent", type="group")

    @interact.decorate(
        runActionTemplate=dict(icon=randomPixmap),
        parent=parent,
        runOptions=RunOptions.ON_ACTION,
    )
    def a():
        """"""

    groupItem = parent.child("a").itemClass(parent.child("a"), 1)
    buttonPixmap = groupItem.button.icon().pixmap(randomPixmap.size())

    # hold references to the QImages
    images = [ pix.toImage() for pix in (randomPixmap, buttonPixmap) ]

    imageBytes = [ fn.ndarray_from_qimage(img) for img in images ]
    assert np.array_equal(*imageBytes)


def test_interact_ignore_none_child():
    class InteractorSubclass(Interactor):
        def resolveAndHookupParameterChild(
            self, functionGroup, childOpts, interactiveFunction
        ):
            if childOpts["type"] not in PARAM_TYPES:
                # Optionally add to `extra` instead
                return None
            return super().resolveAndHookupParameterChild(
                functionGroup, childOpts, interactiveFunction
            )

    interactor = InteractorSubclass()
    out = interactor(lambda a=None: a, runOptions=[])
    assert "a" not in out.names


def test_interact_existing_parent():
    lastValue = None

    def a():
        nonlocal lastValue
        lastValue = 5

    parent = Parameter.create(name="parent", type="group")
    outParam = interact(a, parent=parent)
    assert outParam in parent.names.values()
    outParam.activate()
    assert lastValue == 5
