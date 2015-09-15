from pyqtgraph.util.lru_cache import LRUCache

def testLRU():
    lru = LRUCache(2, 1)
    # check twice
    checkLru(lru)
    checkLru(lru)

def checkLru(lru):
    lru[1] = 1
    lru[2] = 2
    lru[3] = 3

    assert len(lru) == 2
    assert set([2, 3]) == set(lru.keys())
    assert set([2, 3]) == set(lru.values())

    lru[2] = 2
    assert set([2, 3]) == set(lru.values())

    lru[1] = 1
    set([2, 1]) == set(lru.values())

    #Iterates from the used in the last access to others based on access time.
    assert [(2, 2), (1, 1)] == list(lru.iteritems(accessTime=True))
    lru[2] = 2
    assert [(1, 1), (2, 2)] == list(lru.iteritems(accessTime=True))

    del lru[2]
    assert [(1, 1), ] == list(lru.iteritems(accessTime=True))

    lru[2] = 2
    assert [(1, 1), (2, 2)] == list(lru.iteritems(accessTime=True))

    _a = lru[1]
    assert [(2, 2), (1, 1)] == list(lru.iteritems(accessTime=True))

    _a = lru[2]
    assert [(1, 1), (2, 2)] == list(lru.iteritems(accessTime=True))

    assert lru.get(2) == 2
    assert lru.get(3) == None
    assert [(1, 1), (2, 2)] == list(lru.iteritems(accessTime=True))

    lru.clear()
    assert [] == list(lru.iteritems())


if __name__ == '__main__':
    testLRU()
