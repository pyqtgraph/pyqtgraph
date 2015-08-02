from pyqtgraph.debug import pretty

def test_pretty():
    # simply a smoke test for now
    test_dict = {'str': 'cow jumped over the moon',
                 'list': ['dog', 'chases', 'cat'],
                 'dct': {'foo': 'bar'},
                 'tup': ('spam', 'spam', 'spam'),
                 'set': set([2.7, 3.3, 3.4])}
    pretty(test_dict)