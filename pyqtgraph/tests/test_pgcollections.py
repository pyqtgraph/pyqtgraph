import pytest
from pyqtgraph.pgcollections import ProtectedDict, CaselessDict

protected = None
labels = None
targets = None


def test_ProtectedDict_iterkeys():
    for k in protected.iterkeys():
        assert k in labels

def test_ProtectedDict_iteritems():
    for k, v in protected.iteritems():
        assert k in labels
        assert v in targets

def test_ProtectedDict_itervalues():
    for v in protected.itervalues():
        assert v in targets


protected_dict_functions = [test_ProtectedDict_iterkeys,
                            test_ProtectedDict_iteritems,
                            test_ProtectedDict_itervalues]


@pytest.mark.parameterize("func", protected_dict_functions)
def setup_function(func):
    global protected, labels, targets
    string = 'yep, this is a string!'
    tup = (['a', 'b', 'c'])
    dct = {'a': 1, 'b': 2, 'c': 3}
    lst = ['a', 'b', 'c', 'd']
    labels = ['a_string', 'tuple', 'dict', 'list']
    targets = [string, tup, dct, lst]
    read_only_dict = dict((label, target) for label, target in
                           zip(labels, targets))
    protected = ProtectedDict(read_only_dict)


@pytest.mark.parameterize("func", protected_dict_functions)
def teardown_function(func):
    protected = None
    labels = None
    targets = None


def test_CaselessDict_update():
    dct1 = {'a': 1, 'b': 2, 'c': 3}
    dct2 = {'a': 10, 'b': 20, 'c': 30}

    caseless_dct1 = CaselessDict(dct1)
    caseless_dct1.update(dct2)

    assert dct2 == dict(caseless_dct1)
