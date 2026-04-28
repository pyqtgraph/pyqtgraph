"""Tests for custom Parameter subclass save/restore fidelity (issue #3430)."""
import warnings

import pytest
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, registerParameterType
from pyqtgraph.parametertree.Parameter import PARAM_NAMES, PARAM_TYPES
from pyqtgraph.functions import eq


@pytest.fixture(autouse=True)
def _restore_param_registry():
    """Isolate each test: restore PARAM_TYPES/PARAM_NAMES to their pre-test state."""
    saved_types = dict(PARAM_TYPES)
    saved_names = dict(PARAM_NAMES)
    yield
    PARAM_TYPES.clear()
    PARAM_TYPES.update(saved_types)
    PARAM_NAMES.clear()
    PARAM_NAMES.update(saved_names)


def _classes(p):
    """Recursive class fingerprint: [type, [children...]]."""
    return [type(p), [_classes(c) for c in p.children()]]


def test_custom_subclass_survives_round_trip():
    """A registered custom subclass must be re-created (not its base) after restoreState."""
    class MyGroup(pTypes.GroupParameter):
        def __init__(self, **opts):
            opts['type'] = 'mygroup'
            super().__init__(**opts)

    registerParameterType('mygroup', MyGroup)

    original = MyGroup(name='root', children=[
        dict(name='x', type='int', value=3),
    ])
    state = original.saveState()
    restored = Parameter.create(**state)

    assert type(restored) is MyGroup, (
        f"Expected MyGroup after restoreState, got {type(restored).__name__}"
    )
    assert eq(state, restored.saveState())
    assert _classes(original) == _classes(restored)


def test_unregistered_subclass_warns():
    """A subclass that reuses a built-in type name should emit UserWarning."""
    class BadSub(pTypes.GroupParameter):
        def __init__(self, **opts):
            opts['type'] = 'group'   # reuses built-in type
            super().__init__(**opts)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        BadSub(name='bad')

    assert any(issubclass(warning.category, UserWarning) for warning in w), \
        "Expected a UserWarning for type/class mismatch"


def test_registered_subclass_no_warning():
    """A properly registered subclass must not produce any warning."""
    class GoodSub(pTypes.GroupParameter):
        def __init__(self, **opts):
            opts['type'] = 'goodsub'
            super().__init__(**opts)

    registerParameterType('goodsub', GoodSub)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        GoodSub(name='good')

    assert not any(issubclass(warning.category, UserWarning) for warning in w), \
        "Unexpected UserWarning for correctly registered subclass"
