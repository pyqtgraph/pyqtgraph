import warnings
warnings.warn(
    "No longer used in pyqtgraph. Will be removed in 0.13",
    DeprecationWarning, stacklevel=2
)

import operator
import sys
import itertools


class LRUCache(object):
    '''
    This LRU cache should be reasonable for short collections (until around 100 items), as it does a
    sort on the items if the collection would become too big (so, it is very fast for getting and
    setting but when its size would become higher than the max size it does one sort based on the
    internal time to decide which items should be removed -- which should be Ok if the resizeTo
    isn't too close to the maxSize so that it becomes an operation that doesn't happen all the
    time).
    '''

    def __init__(self, maxSize=100, resizeTo=70):
        '''
        ============== =========================================================
        **Arguments:**
        maxSize        (int) This is the maximum size of the cache. When some 
                       item is added and the cache would become bigger than 
                       this, it's resized to the value passed on resizeTo.
        resizeTo       (int) When a resize operation happens, this is the size 
                       of the final cache.
        ============== =========================================================
        '''
        assert resizeTo < maxSize
        self.maxSize = maxSize
        self.resizeTo = resizeTo
        self._counter = 0
        self._dict = {}
        self._nextTime = itertools.count(0).__next__

    def __getitem__(self, key):
        item = self._dict[key]
        item[2] = self._nextTime()
        return item[1]

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        item = self._dict.get(key)
        if item is None:
            if len(self._dict) + 1 > self.maxSize:
                self._resizeTo()
            
            item = [key, value, self._nextTime()]
            self._dict[key] = item
        else:
            item[1] = value
            item[2] = self._nextTime()
            
    def __delitem__(self, key):
        del self._dict[key]
        
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
        
    def clear(self):
        self._dict.clear()
 
    def values(self):
        return [i[1] for i in self._dict.values()]

    def keys(self):
        return [x[0] for x in self._dict.values()]

    def _resizeTo(self):
        ordered = sorted(self._dict.values(), key=operator.itemgetter(2))[:self.resizeTo]
        for i in ordered:
            del self._dict[i[0]]

    def items(self, accessTime=False):
        '''
        :param bool accessTime:
            If True sorts the returned items by the internal access time.
        '''
        if accessTime:
            for x in sorted(self._dict.values(), key=operator.itemgetter(2)):
                yield x[0], x[1]
        else:
            for x in self._dict.items():
                yield x[0], x[1]
