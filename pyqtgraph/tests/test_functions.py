import pyqtgraph as pg
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

np.random.seed(12345)

def testSolve3D():
    p1 = np.array([[0,0,0,1],
                   [1,0,0,1],
                   [0,1,0,1],
                   [0,0,1,1]], dtype=float)
    
    # transform points through random matrix
    tr = np.random.normal(size=(4, 4))
    tr[3] = (0,0,0,1)
    p2 = np.dot(tr, p1.T).T[:,:3]
    
    # solve to see if we can recover the transformation matrix.
    tr2 = pg.solve3DTransform(p1, p2)
    
    assert_array_almost_equal(tr[:3], tr2[:3])


def test_mapCoordinates():
    data = np.array([[ 0.,  1.,  2.],
                     [ 2.,  3.,  5.],
                     [ 7.,  7.,  4.]])
    
    x = np.array([[  0.3,   0.6],
                  [  1. ,   1. ],
                  [  0.5,   1. ],
                  [  0.5,   2.5],
                  [ 10. ,  10. ]])
    
    result = pg.mapCoordinates(data, x)
    
    import scipy.ndimage
    spresult = scipy.ndimage.map_coordinates(data, x.T, order=1)
    
    assert_array_almost_equal(result, spresult)
    

