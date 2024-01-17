import math
import pickle

from pyqtgraph import SRTTransform
from pyqtgraph.parametertree import Parameter


def test_SRTTransform():
    a = SRTTransform({'scale': 2, 'angle': math.pi / 2})
    b = pickle.loads(pickle.dumps(a))
    assert a == b


def test_SRTTransform3D():
    a = SRTTransform({'scale': 2, 'angle': math.pi / 2})
    b = pickle.loads(pickle.dumps(a))
    assert a == b


if __name__ == '__main__':
    test_SRTTransform()
    test_SRTTransform3D()
