import pytest
from functools import wraps
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter as GP
from pyqtgraph.parametertree import RunOpts, InteractiveFunction, Interactor, interact


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
    interactor = Interactor(runOpts=RunOpts.ON_ACTION)
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
    host.child("Run").activate()
    assert value == (5, 5)
    myval = 10
    host.child("Run").activate()
    assert value == (10, 5)

    host = interactor(
        a, x=10, y=50, ignores=["x"], runOpts=(RunOpts.ON_CHANGED, RunOpts.ON_CHANGING)
    )
    for child in "x", "Run":
        assert child not in host.names

    host["y"] = 20
    assert value == (10, 20)
    host.child("y").sigValueChanging.emit(host.child("y"), 100)
    assert value == (10, 100)

    with interactor.optsContext(title=str.upper):
        host = interactor(a, x={"title": "different", "value": 5})
        titles = [p.title() for p in host]
        for ch in "different", "Y":
            assert ch in titles

    with interactor.optsContext(title="Group only"):
        host = interactor(a, x=1)
        assert host.title() == "Group only"
        assert [p.title() is None for p in host]

    with interactor.optsContext(runOpts=RunOpts.ON_CHANGED):
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
    host.child("Run").activate()
    assert value == 12

    host = GP.create(name="test deco", type="group")
    interactor.setOpts(parent=host)

    @interactor.decorate()
    @retain
    def a(x=5):
        return x

    assert "a" in host.names
    assert "x" in host.child("a").names
    host.child("a", "Run").activate()
    assert value == 5

    @interactor.decorate(nest=False, runOpts=RunOpts.ON_CHANGED)
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

    host = interactor(wraps(raw)(override), runOpts=RunOpts.ON_CHANGED)
    assert "x" in host.names
    host["x"] = 100
    assert value == 100


def test_run():
    def a():
        """"""

    interactor = Interactor(runOpts=RunOpts.ON_ACTION)

    defaultRunBtn = Parameter.create(**interactor.runActionTemplate, name="Run")
    btn = interactor(a)
    assert btn.type() == defaultRunBtn.type()

    template = dict(defaultName="Test", type="action")
    with interactor.optsContext(runActionTemplate=template):
        x = interactor(a)
    assert x.name() == "Test"

    parent = Parameter.create(name="parent", type="group")
    test2 = interactor(a, parent=parent, nest=False)
    assert test2.parent() is parent

    test2 = interactor(a, nest=False)
    assert not test2.parent()

def test_no_func_group():
    def inner(a=5, b=6):
        return a + b

    out = interact(inner, nest=False)
    assert isinstance(out, list)


def test_tips():
    def a():
        """a simple tip"""

    interactor = Interactor()

    btn = interactor(a, runOpts=RunOpts.ON_ACTION)
    assert btn.opts["tip"] == a.__doc__

    def a2(x=5):
        """a simple tip"""

    def a3(x=5):
        """
        A long docstring with a newline
        followed by more text won't result in a tooltip
        """

    param = interactor(a2, runOpts=RunOpts.ON_ACTION)
    assert param.opts["tip"] == a2.__doc__ and param.type() == "group"

    param = interactor(a3)
    assert "tip" not in param.opts


def test_interactiveFunc():
    value = 0

    def myfunc(a=5):
        nonlocal value
        value = a
        return a

    interactive = InteractiveFunction(myfunc)
    host = interact(interactive, runOpts=[])

    host["a"] = 7
    assert interactive.runFromAction() == 7

    interactive.disconnect()
    interactive.runFromAction(a=10)
    assert value == 7

    interactive.reconnect()
    interactive.runFromAction(a=10)
    assert value == 10


def test_badOptsContext():
    with pytest.raises(KeyError):
        Interactor(bad=4)


def test_updateParamDuringRun():
    counter = 0

    @InteractiveFunction
    def func(a=1):
        nonlocal counter
        counter += a

    param = interact(func)
    func.parametersNeedRunKwargs = True

    func(a=3)
    # Ensure "test" was only run once
    assert counter == 3
    assert param["a"] == 3

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

    host = interact(inner)
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
    host = interact(a)

    p2 = Parameter.create(name="p2", type="int", value=3)
    a.hookupParameters([p2], clearOld=False)

    assert a() == 8

