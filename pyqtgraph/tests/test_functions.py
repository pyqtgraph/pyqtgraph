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


def test_interpolateArray():
    data = np.array([[ 1.,   2.,   4.  ],
                     [ 10.,  20.,  40. ],
                     [ 100., 200., 400.]])
    
    x = np.array([[  0.3,   0.6],
                  [  1. ,   1. ],
                  [  0.5,   1. ],
                  [  0.5,   2.5],
                  [ 10. ,  10. ]])
    
    result = pg.interpolateArray(data, x)
    
    #import scipy.ndimage
    #spresult = scipy.ndimage.map_coordinates(data, x.T, order=1)
    spresult = np.array([  5.92,  20.  ,  11.  ,   0.  ,   0.  ])  # generated with the above line
    
    assert_array_almost_equal(result, spresult)
    
    # test mapping when x.shape[-1] < data.ndim
    x = np.array([[  0.3,   0],
                  [  0.3,   1],
                  [  0.3,   2]])
    
    r1 = pg.interpolateArray(data, x)
    r2 = pg.interpolateArray(data, x[0,:1])
    assert_array_almost_equal(r1, r2)
    
    
    # test mapping 2D array of locations
    x = np.array([[[0.5, 0.5], [0.5, 1.0], [0.5, 1.5]],
                  [[1.5, 0.5], [1.5, 1.0], [1.5, 1.5]]])
    
    r1 = pg.interpolateArray(data, x)
    #r2 = scipy.ndimage.map_coordinates(data, x.transpose(2,0,1), order=1)
    r2 = np.array([[   8.25,   11.  ,   16.5 ],  # generated with the above line
                   [  82.5 ,  110.  ,  165.  ]])

    assert_array_almost_equal(r1, r2)
    
def test_subArray():
    a = np.array([0, 0, 111, 112, 113, 0, 121, 122, 123, 0, 0, 0, 211, 212, 213, 0, 221, 222, 223, 0, 0, 0, 0])
    b = pg.subArray(a, offset=2, shape=(2,2,3), stride=(10,4,1))
    c = np.array([[[111,112,113], [121,122,123]], [[211,212,213], [221,222,223]]])
    assert np.all(b == c)
    
    # operate over first axis; broadcast over the rest
    aa = np.vstack([a, a/100.]).T
    cc = np.empty(c.shape + (2,))
    cc[..., 0] = c
    cc[..., 1] = c / 100.
    bb = pg.subArray(aa, offset=2, shape=(2,2,3), stride=(10,4,1))
    assert np.all(bb == cc)
    
    
    
if __name__ == '__main__':
    test_interpolateArray()