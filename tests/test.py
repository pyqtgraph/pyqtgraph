import unittest
import os, sys
## make sure this instance of pyqtgraph gets imported first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## all tests should be defined with this class so we have the option to tweak it later.
class TestCase(unittest.TestCase):
        
    def testLRU(self):
        from pyqtgraph.lru_cache import LRUCache
        lru = LRUCache(2, 1)

        def CheckLru():
            lru[1] = 1
            lru[2] = 2
            lru[3] = 3

            self.assertEqual(2, len(lru))
            self.assertSetEqual(set([2, 3]), set(lru.keys()))
            self.assertSetEqual(set([2, 3]), set(lru.values()))

            lru[2] = 2
            self.assertSetEqual(set([2, 3]), set(lru.values()))
            
            lru[1] = 1
            self.assertSetEqual(set([2, 1]), set(lru.values()))

            #Iterates from the used in the last access to others based on access time.
            self.assertEqual([(2, 2), (1, 1)], list(lru.iteritems(access_time=True)))
            lru[2] = 2
            self.assertEqual([(1, 1), (2, 2)], list(lru.iteritems(access_time=True)))

            del lru[2]
            self.assertEqual([(1, 1), ], list(lru.iteritems(access_time=True)))

            lru[2] = 2
            self.assertEqual([(1, 1), (2, 2)], list(lru.iteritems(access_time=True)))

            _a = lru[1]
            self.assertEqual([(2, 2), (1, 1)], list(lru.iteritems(access_time=True)))

            _a = lru[2]
            self.assertEqual([(1, 1), (2, 2)], list(lru.iteritems(access_time=True)))

            self.assertEqual(lru.get(2), 2)
            self.assertEqual(lru.get(3), None)
            self.assertEqual([(1, 1), (2, 2)], list(lru.iteritems(access_time=True)))

            lru.clear()
            self.assertEqual([], list(lru.iteritems()))

        CheckLru()

        # Check it twice...
        CheckLru()

if __name__ == '__main__':
    unittest.main()