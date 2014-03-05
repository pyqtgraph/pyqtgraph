import operator
import sys
import itertools


_IS_PY3 = sys.version_info[0] == 3

class LRUCache(object):
    '''
    This LRU cache should be reasonable for short collections (until around 100 items), as it does a
    sort on the items if the collection would become too big (so, it is very fast for getting and
    setting but when its size would become higher than the max size it does one sort based on the
    internal time to decide which items should be removed -- which should be Ok if the resize_to
    isn't too close to the max_size so that it becomes an operation that doesn't happen all the
    time).
    '''

    def __init__(self, max_size=100, resize_to=70):
        '''
        :param int max_size:
            This is the maximum size of the cache. When some item is added and the cache would become
            bigger than this, it's resized to the value passed on resize_to.
            
        :param int resize_to:
            When a resize operation happens, this is the size of the final cache.
        '''
        assert resize_to < max_size
        self.max_size = max_size
        self.resize_to = resize_to
        self._counter = 0
        self._dict = {}
        if _IS_PY3:
            self._next_time = itertools.count(0).__next__
        else:
            self._next_time = itertools.count(0).next

    def __getitem__(self, key):
        item = self._dict[key]
        item[2] = self._next_time()
        return item[1]

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        item = self._dict.get(key)
        if item is None:
            if len(self._dict) + 1 > self.max_size:
                self._resize_to()
            
            item = [key, value, self._next_time()]
            self._dict[key] = item
        else:
            item[1] = value
            item[2] = self._next_time()
            
    def __delitem__(self, key):
        del self._dict[key]
        
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
        
    def clear(self):
        self._dict.clear()
 
    if _IS_PY3:
        def values(self):
            return [i[1] for i in self._dict.values()]
        
        def keys(self):
            return [x[0] for x in self._dict.values()]
        
        def _resize_to(self):
            ordered = sorted(self._dict.values(), key=operator.itemgetter(2))[:self.resize_to]
            for i in ordered:
                del self._dict[i[0]]
                
        def iteritems(self, access_time=False):
            '''
            :param bool access_time:
                If True sorts the returned items by the internal access time.
            '''
            if access_time:
                for x in sorted(self._dict.values(), key=operator.itemgetter(2)):
                    yield x[0], x[1]
            else:
                for x in self._dict.items():
                    yield x[0], x[1]
                    
    else:
        def values(self):
            return [i[1] for i in self._dict.itervalues()]
        
        def keys(self):
            return [x[0] for x in self._dict.itervalues()]
            
        
        def _resize_to(self):
            ordered = sorted(self._dict.itervalues(), key=operator.itemgetter(2))[:self.resize_to]
            for i in ordered:
                del self._dict[i[0]]
                
        def iteritems(self, access_time=False):
            '''
            :param bool access_time:
                If True sorts the returned items by the internal access time.
            '''
            if access_time:
                for x in sorted(self._dict.itervalues(), key=operator.itemgetter(2)):
                    yield x[0], x[1]
            else:
                for x in self._dict.iteritems():
                    yield x[0], x[1]
