import math
import pickle

from pyqtgraph import SRTTransform


def test_SRTTransform():
    a = SRTTransform({'scale': 2, 'angle': math.pi / 2})
    b = pickle.loads(pickle.dumps(a))
    assert a == b


def test_SRTTransform3D():
    a = SRTTransform({'scale': 2, 'angle': math.pi / 2})
    b = pickle.loads(pickle.dumps(a))
    assert a == b
