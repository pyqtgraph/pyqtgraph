import unittest
import os, sys
## make sure this instance of pyqtgraph gets imported first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## all tests should be defined with this class so we have the option to tweak it later.
class TestCase(unittest.TestCase):
    pass