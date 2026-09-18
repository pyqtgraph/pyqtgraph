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
    interact, ParameterTree,
)
from pyqtgraph.parametertree.Parameter import PARAM_TYPES, coalesceTreeChanges
from pyqtgraph.parametertree.parameterTypes import GroupParameter as GP
from pyqtgraph.Qt import QtGui

pg.mkQApp()

def test_parameter_hasdefault():
    opts = {"name": "param", "type": 'int', "value": 1}

    # default unspecified
    p = Parameter.create(**opts)
    assert p.hasDefault()

    # default specified
    p = Parameter.create(default=0, **opts)
    assert p.hasDefault()
    assert p.defaultValue() == 0

    # default specified as None
    p = Parameter.create(default=None, **opts)
    assert not p.hasDefault()
    p.setDefault(2)
    assert p.hasDefault()
    assert p.defaultValue() == 2


def test_parameter_getValues():
    params = [
        {"name": "a", "type": "int", "value": 1},
        {"name": "b", "type": "float"},
        {"name": "c", "type": "group", 'children': [
            {"name": "d", "type": "bool"},
            {"name": "e", "type": "int", "value": 2},
        ]},
    ]
    p = Parameter.create(name="param", type='group',
                         children=params)
    p.getValues()


def test_parameter_no_value():
    p = Parameter.create(name="param", type='group',)
    assert p.value() is None

    p = Parameter.create(name='param', type='float',)
    assert p.value() is None


def test_parameter_defaults_and_pristineness():
    # init with identical value and default
    p = Parameter.create(name="param", type='int', value=1, default=1)
    assert p.valueModifiedSinceResetToDefault() is True
    # init with different value and default
    p = Parameter.create(name="param", type='int', value=1, default=2)
    assert p.valueModifiedSinceResetToDefault() is True
    # init with value only
    p = Parameter.create(name="param", type='int', value=1)
    assert p.valueModifiedSinceResetToDefault() is True
    # init with default only
    p = Parameter.create(name="param", type='int', default=1)
    assert p.valueModifiedSinceResetToDefault() is False

    # initially value is pristine since only a default was given
    assert p.value() == 1
    # update default, and allow the value to track since it is pristine
    p.setDefault(2, updatePristineValues=True)
    assert p.value() == 2
    assert p.valueModifiedSinceResetToDefault() is False
    # update default but do not allow the value to track
    p.setDefault(3)  # by default, updatePristineValues=False
    assert p.value() == 2
    assert p.valueModifiedSinceResetToDefault() is True        
    # update default again, explicitly requesting updatePristineValues=False
    p.setToDefault()
    assert p.valueModifiedSinceResetToDefault() is False
    p.setDefault(4, updatePristineValues=False)
    assert p.value() == 3
    assert p.valueModifiedSinceResetToDefault() is True
    # update value directly, causing dirty state
    p.setToDefault()
    assert p.valueModifiedSinceResetToDefault() is False
    p.setValue(5)
    assert p.valueModifiedSinceResetToDefault() is True
    p.setDefault(6, updatePristineValues=True)
    assert p.value() == 5
    assert p.valueModifiedSinceResetToDefault() is True
    # test setting value to same as default value does not result in pristine state
    p.setToDefault()
    assert p.valueModifiedSinceResetToDefault() is False
    p.setValue(0)
    p.setValue(p.defaultValue())
    assert p.valueModifiedSinceResetToDefault() is True
    # test setToDefault
    p.setToDefault()
    assert p.valueModifiedSinceResetToDefault() is False
    assert p.value() == p.defaultValue()
    p.setDefault(7, updatePristineValues=True)
    assert p.value() == 7

    # if the value is coerced away from the raw default, the parameter remains modified
    p = Parameter.create(name="param", type='int', value=1, default=2.5)
    p.setToDefault()
    assert p.value() == 2
    assert p.valueModifiedSinceResetToDefault() is True

    # init with neither value nor default
    p = Parameter.create(name="param", type='int')
    assert p.valueModifiedSinceResetToDefault() is False
    p.setDefault(8)
    assert p.valueModifiedSinceResetToDefault() is False
    assert p.value() == 8


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


class TestTreeChangeBlocker:

    def _makeNestedTree(self):
        self.root = Parameter.create(name="root", type="group", children=[
            dict(name="group", type="group", children=[
                dict(name="p", type="list", limits=["a", "b", "c"], value="a"),
            ]),
        ])
        self.group = self.root.child("group")
        self.param = self.group.child("p")

    def _recordChanges(self, param):
        events = []
        param.sigTreeStateChanged.connect(lambda emitter, changes: events.append(changes))
        self.events = events
        return events

    def _create_tree(self):
        self.tree = ParameterTree()
        self.tree.setParameters(self.root)

    @pytest.fixture(autouse=True)
    def setup_tree(self, qtbot):
        """
        Runs before every test.
         'autouse=True' means you don't have to explicitly pass it to tests.
        """
        self._makeNestedTree()
        self._recordChanges(self.root)
        self._create_tree()
        qtbot.addWidget(self.tree)

        self.tree.show()

        yield

        self.tree.close()

    def test_treeChangeBlocker_default_behavior_unchanged(self):
        # with no keep/dedupe/emitter, treeChangeBlocker should behave exactly as before:
        # one signal from self carrying every real change, uncoalesced.

        with self.root.treeChangeBlocker():
            self.param.setValue("b")
            self.param.setValue("c")

        assert len(self.events) == 1
        assert self.events[0] == [(self.param, "value", "b"), (self.param, "value", "c")]

    def test_treeChangeBlocker_keep_filters_change_types_value(self):

        with self.root.treeChangeBlocker(keep={'value'}):
            self.root.child('group', 'p').setLimits(["a", "b", "c", "d"])
            self.root.child('group', 'p').setValue("b")

        assert len(self.events) == 1
        assert self.events[0] == [(self.param, "value", "b")]

    def test_treeChangeBlocker_keep_filters_change_types_limits(self):

        with self.root.treeChangeBlocker(keep={'limits'}):
            self.root.child('group', 'p').setLimits(["a", "b", "c", "d"])
            self.root.child('group', 'p').setValue("b")

        assert len(self.events) == 1
        assert self.events[0] == [(self.param, 'limits', ["a", "b", "c", "d"])]

    def test_treeChangeBlocker_keep_empty_is_silent(self):

        with self.root.treeChangeBlocker(keep=set()):
            self.param.setValue("b")
        assert self.events == []

    def test_treeChangeBlocker_dedupe_collapses_repeated_changes(self):
        with self.param.treeChangeBlocker(dedupe=True):
            self.param.setValue("b")
            self.param.setValue("c")
            self.param.setValue("a")

        assert len(self.events) == 1
        assert self.events[0] == [(self.param, "value", "a")]

    def test_treeChangeBlocker_dedupe_keeps_distinct_change_types(self):
        p = self.param
        with p.treeChangeBlocker(dedupe=True):
            p.setValue("b")
            p.setValue("c")
            p.setLimits(["a", "b", "c", "d"])

        assert len(self.events) == 1
        changes = dict((changeType, data) for (_, changeType, data) in self.events[0])
        assert changes == {"value": "c", "limits": ["a", "b", "c", "d"]}

    def test_treeChangeBlocker_dedupe_merges_options_payloads(self):
        p = self.param
        with p.treeChangeBlocker(dedupe=True):
            p.setOpts(readonly=True)
            p.show(False)

        assert len(self.events) == 1
        assert self.events[0] == [(p, "options", {"readonly": True, "visible": False})]


    def test_treeChangeBlocker_dedupe_same_option_key_keeps_last_value(self):
        p = self.param
        with p.treeChangeBlocker(dedupe=True):
            p.setOpts(readonly=True)
            p.setOpts(readonly=False)

        assert len(self.events) == 1
        assert self.events[0] == [(p, "options", {"readonly": False})]

    def test_treeChangeBlocker_emitter_routes_signal(self):
        self.root.sigTreeStateChanged.disconnect()

        rootEvents = self._recordChanges(self.root)
        groupEvents = self._recordChanges(self.group)
        pEvents = self._recordChanges(self.param)

        with self.param.treeChangeBlocker(dedupe=True, emitter=self.root):
            self.param.setValue("b")
            self.param.setValue("c")

        # the signal is raised on `root`, and nothing at all reaches `group` or `p`'s
        # own listeners -- the cascade is stopped at the source, not just masked at root.
        assert groupEvents == []
        assert pEvents == []
        assert len(rootEvents) == 1
        assert rootEvents[0] == [(self.param, "value", "c")]


    def test_coalesceTreeChanges_keep_filters_by_type(self):
        p = object()
        changes = [(p, "value", 1), (p, "limits", [1, 2])]
        result = coalesceTreeChanges(changes, keep={"value"})
        assert result == [(p, "value", 1)]


    def test_coalesceTreeChanges_dedupe_keeps_last_scalar(self):
        p = object()
        changes = [(p, "value", 1), (p, "value", 2), (p, "value", 3)]
        result = coalesceTreeChanges(changes, dedupe=True)
        assert result == [(p, "value", 3)]


    def test_coalesceTreeChanges_dedupe_merges_dict_payloads(self):
        p = object()
        changes = [(p, "options", {"readonly": True}), (p, "options", {"visible": False})]
        result = coalesceTreeChanges(changes, dedupe=True)
        assert result == [(p, "options", {"readonly": True, "visible": False})]
        
        
# ---------------------------------------------------------------------------
# Tests for Parameter.setValue() blockSignal / blockSlots behaviour
# (regression for #3305, alternative to #3489)
# ---------------------------------------------------------------------------

def test_setValue_blockSlots_single_callable_blocks_only_that_slot():
    """
    blockSlots=<callable> must temporarily disconnect *only* that slot while
    still emitting sigValueChanged to every other connected listener.
    """
    p = Parameter.create(name="param", type="float", value=0.0)

    blocked_received = []
    other_received = []

    def blocked_slot(param, value):
        blocked_received.append(value)

    def other_slot(param, value):
        other_received.append(value)

    p.sigValueChanged.connect(blocked_slot)
    p.sigValueChanged.connect(other_slot)

    p.setValue(1.0, blockSlots=blocked_slot)

    assert blocked_received == [], (
        "Blocked slot must not receive the signal when passed as blockSlots"
    )
    assert other_received == [1.0], (
        "Other connected slots must still receive sigValueChanged"
    )

    # After the call the slot must be reconnected; next plain setValue fires both
    p.setValue(2.0)
    assert blocked_received == [2.0], (
        "Blocked slot must be reconnected after the setValue call"
    )
    assert other_received == [1.0, 2.0]


def test_setValue_blockSlots_non_callable_raises_typeerror():
    """
    Passing a non-callable (or a list containing one) as blockSlots must
    raise a clear TypeError instead of failing deep inside Qt's disconnect(),
    and must not disconnect any of the other, valid slots in the list.
    """
    p = Parameter.create(name="param", type="float", value=0.0)

    with pytest.raises(TypeError):
        p.setValue(1.0, blockSlots="not callable")

    received = []

    def real_slot(param, value):
        received.append(value)

    p.sigValueChanged.connect(real_slot)

    with pytest.raises(TypeError):
        p.setValue(2.0, blockSlots=[real_slot, "not callable"])

    # real_slot must still be connected: validation must fail before any
    # slot in the list gets disconnected
    p.setValue(3.0)
    assert received == [3.0]


def test_setValue_blockSlots_list_blocks_multiple_slots():
    """
    blockSlots accepts a list/tuple of callables, all of which should be
    disconnected during emission and reconnected afterward.
    """
    p = Parameter.create(name="param", type="float", value=0.0)

    received_a, received_b, received_c = [], [], []

    def slot_a(param, value):
        received_a.append(value)

    def slot_b(param, value):
        received_b.append(value)

    def slot_c(param, value):
        received_c.append(value)

    p.sigValueChanged.connect(slot_a)
    p.sigValueChanged.connect(slot_b)
    p.sigValueChanged.connect(slot_c)

    p.setValue(1.0, blockSlots=[slot_a, slot_b])

    assert received_a == []
    assert received_b == []
    assert received_c == [1.0]

    p.setValue(2.0)
    assert received_a == [2.0]
    assert received_b == [2.0]
    assert received_c == [1.0, 2.0]


def test_setValue_blockSlots_ignores_unconnected_slot():
    """
    A callable passed to blockSlots that is not (yet) connected to
    sigValueChanged must be silently ignored instead of raising, and must
    not be connected afterward as a side effect.
    """
    p = Parameter.create(name="param", type="float", value=0.0)

    received = []
    unconnected_received = []

    def slot(param, value):
        received.append(value)

    def unconnected_slot(param, value):
        unconnected_received.append(value)

    p.sigValueChanged.connect(slot)

    # unconnected_slot was never connected; passing it must not raise
    p.setValue(1.0, blockSlots=[slot, unconnected_slot])

    assert received == [], "slot must be blocked as requested"
    assert unconnected_received == [], (
        "unconnected_slot was never connected, so it must not receive the signal"
    )

    # slot must be reconnected; unconnected_slot must remain unconnected
    p.setValue(2.0)
    assert received == [2.0]
    assert unconnected_received == [], (
        "unconnected_slot must not be connected as a side effect of setValue"
    )


def test_setValue_blockSignal_true_suppresses_signal_entirely():
    """
    blockSignal=True must prevent sigValueChanged from being emitted at all,
    even when blockSlots is also given.
    """
    p = Parameter.create(name="param", type="float", value=0.0)

    received = []

    def slot(param, value):
        received.append(value)

    p.sigValueChanged.connect(slot)

    p.setValue(99.0, blockSignal=True)

    assert received == [], (
        "sigValueChanged must not be emitted when blockSignal=True"
    )
    # Value must still be stored despite the signal being blocked
    assert p.value() == 99.0


def test_setValue_blockSignal_none_emits_normally():
    """
    The default blockSignal=None (falsy) must emit sigValueChanged as usual.
    """
    p = Parameter.create(name="param", type="float", value=0.0)

    received = []

    def slot(param, value):
        received.append(value)

    p.sigValueChanged.connect(slot)

    p.setValue(42.0)          # blockSignal defaults to None
    p.setValue(43.0, blockSignal=None)
    p.setValue(44.0, blockSignal=False)

    assert received == [42.0, 43.0, 44.0], (
        "sigValueChanged must fire for all falsy blockSignal values"
    )


def test_setValue_blockSignal_callable_is_deprecated_but_still_works():
    """
    Passing a callable as blockSignal (the old, pre-#3489 usage pattern) must
    still work for backward compatibility, but must raise a DeprecationWarning
    and behave the same as passing it via blockSlots.
    """
    p = Parameter.create(name="param", type="float", value=0.0)

    blocked_received = []
    other_received = []

    def blocked_slot(param, value):
        blocked_received.append(value)

    def other_slot(param, value):
        other_received.append(value)

    p.sigValueChanged.connect(blocked_slot)
    p.sigValueChanged.connect(other_slot)

    with pytest.deprecated_call():
        p.setValue(1.0, blockSignal=blocked_slot)

    assert blocked_received == []
    assert other_received == [1.0]

    p.setValue(2.0)
    assert blocked_received == [2.0]
    assert other_received == [1.0, 2.0]
