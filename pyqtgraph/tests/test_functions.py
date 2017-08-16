import pyqtgraph as pg
import numpy as np
import sys
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

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


def test_interpolateArray_order0():
    check_interpolateArray(order=0)


def test_interpolateArray_order1():
    check_interpolateArray(order=1)


def check_interpolateArray(order):
    def interpolateArray(data, x):
        result = pg.interpolateArray(data, x, order=order)
        assert result.shape == x.shape[:-1] + data.shape[x.shape[-1]:]
        return result
    
    data = np.array([[ 1.,   2.,   4.  ],
                     [ 10.,  20.,  40. ],
                     [ 100., 200., 400.]])
    
    # test various x shapes
    interpolateArray(data, np.ones((1,)))
    interpolateArray(data, np.ones((2,)))
    interpolateArray(data, np.ones((1, 1)))
    interpolateArray(data, np.ones((1, 2)))
    interpolateArray(data, np.ones((5, 1)))
    interpolateArray(data, np.ones((5, 2)))
    interpolateArray(data, np.ones((5, 5, 1)))
    interpolateArray(data, np.ones((5, 5, 2)))
    with pytest.raises(TypeError):
        interpolateArray(data, np.ones((3,)))
    with pytest.raises(TypeError):
        interpolateArray(data, np.ones((1, 3,)))
    with pytest.raises(TypeError):
        interpolateArray(data, np.ones((5, 5, 3,)))
    
    x = np.array([[  0.3,   0.6],
                  [  1. ,   1. ],
                  [  0.501,   1. ],   # NOTE: testing at exactly 0.5 can yield different results from map_coordinates
                  [  0.501,   2.501],  # due to differences in rounding
                  [ 10. ,  10. ]])
    
    result = interpolateArray(data, x)
    # make sure results match ndimage.map_coordinates
    import scipy.ndimage
    spresult = scipy.ndimage.map_coordinates(data, x.T, order=order)
    #spresult = np.array([  5.92,  20.  ,  11.  ,   0.  ,   0.  ])  # generated with the above line
    
    assert_array_almost_equal(result, spresult)
    
    # test mapping when x.shape[-1] < data.ndim
    x = np.array([[  0.3,   0],
                  [  0.3,   1],
                  [  0.3,   2]])
    r1 = interpolateArray(data, x)
    x = np.array([0.3])  # should broadcast across axis 1
    r2 = interpolateArray(data, x)
    
    assert_array_almost_equal(r1, r2)
    
    
    # test mapping 2D array of locations
    x = np.array([[[0.501, 0.501], [0.501, 1.0], [0.501, 1.501]],
                  [[1.501, 0.501], [1.501, 1.0], [1.501, 1.501]]])
    
    r1 = interpolateArray(data, x)
    r2 = scipy.ndimage.map_coordinates(data, x.transpose(2,0,1), order=order)
    #r2 = np.array([[   8.25,   11.  ,   16.5 ],  # generated with the above line
                   #[  82.5 ,  110.  ,  165.  ]])

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
    
    
def test_rescaleData():
    dtypes = map(np.dtype, ('ubyte', 'uint16', 'byte', 'int16', 'int', 'float'))
    for dtype1 in dtypes:
        for dtype2 in dtypes:
            data = (np.random.random(size=10) * 2**32 - 2**31).astype(dtype1)
            for scale, offset in [(10, 0), (10., 0.), (1, -50), (0.2, 0.5), (0.001, 0)]:
                if dtype2.kind in 'iu':
                    lim = np.iinfo(dtype2)
                    lim = lim.min, lim.max
                else:
                    lim = (-np.inf, np.inf)
                s1 = np.clip(float(scale) * (data-float(offset)), *lim).astype(dtype2)
                s2 = pg.rescaleData(data, scale, offset, dtype2)
                assert s1.dtype == s2.dtype
                if dtype2.kind in 'iu':
                    assert np.all(s1 == s2)
                else:
                    assert np.allclose(s1, s2)


def test_makeARGB():
    # Many parameters to test here:
    #  * data dtype (ubyte, uint16, float, others)
    #  * data ndim (2 or 3)
    #  * levels (None, 1D, or 2D)
    #  * lut dtype
    #  * lut size
    #  * lut ndim (1 or 2)
    #  * useRGBA argument
    # Need to check that all input values map to the correct output values, especially
    # at and beyond the edges of the level range.

    def checkArrays(a, b):
        # because py.test output is difficult to read for arrays
        if not np.all(a == b):
            comp = []
            for i in range(a.shape[0]):
                if a.shape[1] > 1:
                    comp.append('[')
                for j in range(a.shape[1]):
                    m = a[i,j] == b[i,j]
                    comp.append('%d,%d  %s %s  %s%s' % 
                                (i, j, str(a[i,j]).ljust(15), str(b[i,j]).ljust(15),
                                 m, ' ********' if not np.all(m) else ''))
                if a.shape[1] > 1:
                    comp.append(']')
            raise Exception("arrays do not match:\n%s" % '\n'.join(comp))
    
    def checkImage(img, check, alpha, alphaCheck):
        assert img.dtype == np.ubyte
        assert alpha is alphaCheck
        if alpha is False:
            checkArrays(img[..., 3], 255)
        
        if np.isscalar(check) or check.ndim == 3:
            checkArrays(img[..., :3], check)
        elif check.ndim == 2:
            checkArrays(img[..., :3], check[..., np.newaxis])
        elif check.ndim == 1:
            checkArrays(img[..., :3], check[..., np.newaxis, np.newaxis])
        else:
            raise Exception('invalid check array ndim')
        
    # uint8 data tests
    
    im1 = np.arange(256).astype('ubyte').reshape(256, 1)
    im2, alpha = pg.makeARGB(im1, levels=(0, 255))
    checkImage(im2, im1, alpha, False)
    
    im3, alpha = pg.makeARGB(im1, levels=(0.0, 255.0))
    checkImage(im3, im1, alpha, False)

    im4, alpha = pg.makeARGB(im1, levels=(255, 0))
    checkImage(im4, 255-im1, alpha, False)
    
    im5, alpha = pg.makeARGB(np.concatenate([im1]*3, axis=1), levels=[(0, 255), (0.0, 255.0), (255, 0)])
    checkImage(im5, np.concatenate([im1, im1, 255-im1], axis=1), alpha, False)
    

    im2, alpha = pg.makeARGB(im1, levels=(128,383))
    checkImage(im2[:128], 0, alpha, False)
    checkImage(im2[128:], im1[:128], alpha, False)
    

    # uint8 data + uint8 LUT
    lut = np.arange(256)[::-1].astype(np.uint8)
    im2, alpha = pg.makeARGB(im1, lut=lut)
    checkImage(im2, lut, alpha, False)
    
    # lut larger than maxint
    lut = np.arange(511).astype(np.uint8)
    im2, alpha = pg.makeARGB(im1, lut=lut)
    checkImage(im2, lut[::2], alpha, False)
    
    # lut smaller than maxint
    lut = np.arange(128).astype(np.uint8)
    im2, alpha = pg.makeARGB(im1, lut=lut)
    checkImage(im2, np.linspace(0, 127, 256).astype('ubyte'), alpha, False)

    # lut + levels
    lut = np.arange(256)[::-1].astype(np.uint8)
    im2, alpha = pg.makeARGB(im1, lut=lut, levels=[-128, 384])
    checkImage(im2, np.linspace(192, 65.5, 256).astype('ubyte'), alpha, False)
    
    im2, alpha = pg.makeARGB(im1, lut=lut, levels=[64, 192])
    checkImage(im2, np.clip(np.linspace(385.5, -126.5, 256), 0, 255).astype('ubyte'), alpha, False)

    # uint8 data + uint16 LUT
    lut = np.arange(4096)[::-1].astype(np.uint16) // 16
    im2, alpha = pg.makeARGB(im1, lut=lut)
    checkImage(im2, np.arange(256)[::-1].astype('ubyte'), alpha, False)

    # uint8 data + float LUT
    lut = np.linspace(10., 137., 256)
    im2, alpha = pg.makeARGB(im1, lut=lut)
    checkImage(im2, lut.astype('ubyte'), alpha, False)

    # uint8 data + 2D LUT
    lut = np.zeros((256, 3), dtype='ubyte')
    lut[:,0] = np.arange(256)
    lut[:,1] = np.arange(256)[::-1]
    lut[:,2] = 7
    im2, alpha = pg.makeARGB(im1, lut=lut)
    checkImage(im2, lut[:,None,::-1], alpha, False)
    
    # check useRGBA
    im2, alpha = pg.makeARGB(im1, lut=lut, useRGBA=True)
    checkImage(im2, lut[:,None,:], alpha, False)

    
    # uint16 data tests
    im1 = np.arange(0, 2**16, 256).astype('uint16')[:, None]
    im2, alpha = pg.makeARGB(im1, levels=(512, 2**16))
    checkImage(im2, np.clip(np.linspace(-2, 253, 256), 0, 255).astype('ubyte'), alpha, False)

    lut = (np.arange(512, 2**16)[::-1] // 256).astype('ubyte')
    im2, alpha = pg.makeARGB(im1, lut=lut, levels=(512, 2**16-256))
    checkImage(im2, np.clip(np.linspace(257, 2, 256), 0, 255).astype('ubyte'), alpha, False)

    lut = np.zeros(2**16, dtype='ubyte')
    lut[1000:1256] = np.arange(256)
    lut[1256:] = 255
    im1 = np.arange(1000, 1256).astype('uint16')[:, None]
    im2, alpha = pg.makeARGB(im1, lut=lut)
    checkImage(im2, np.arange(256).astype('ubyte'), alpha, False)
    
    
    
    # float data tests
    im1 = np.linspace(1.0, 17.0, 256)[:, None]
    im2, alpha = pg.makeARGB(im1, levels=(5.0, 13.0))
    checkImage(im2, np.clip(np.linspace(-128, 383, 256), 0, 255).astype('ubyte'), alpha, False)
    
    lut = (np.arange(1280)[::-1] // 10).astype('ubyte')
    im2, alpha = pg.makeARGB(im1, lut=lut, levels=(1, 17))
    checkImage(im2, np.linspace(127.5, 0, 256).astype('ubyte'), alpha, False)


    # test sanity checks
    class AssertExc(object):
        def __init__(self, exc=Exception):
            self.exc = exc
        def __enter__(self):
            return self
        def __exit__(self, *args):
            assert args[0] is self.exc, "Should have raised %s (got %s)" % (self.exc, args[0])
            return True
    
    with AssertExc(TypeError):  # invalid image shape
        pg.makeARGB(np.zeros((2,), dtype='float'))
    with AssertExc(TypeError):  # invalid image shape
        pg.makeARGB(np.zeros((2,2,7), dtype='float'))
    with AssertExc():  # float images require levels arg
        pg.makeARGB(np.zeros((2,2), dtype='float'))
    with AssertExc():  # bad levels arg
        pg.makeARGB(np.zeros((2,2), dtype='float'), levels=[1])
    with AssertExc():  # bad levels arg
        pg.makeARGB(np.zeros((2,2), dtype='float'), levels=[1,2,3])
    with AssertExc():  # can't mix 3-channel levels and LUT
        pg.makeARGB(np.zeros((2,2)), lut=np.zeros((10,3), dtype='ubyte'), levels=[(0,1)]*3)
    with AssertExc():  # multichannel levels must have same number of channels as image
        pg.makeARGB(np.zeros((2,2,3), dtype='float'), levels=[(1,2)]*4)
    with AssertExc():  # 3d levels not allowed
        pg.makeARGB(np.zeros((2,2,3), dtype='float'), levels=np.zeros([3, 2, 2]))


def test_eq():
    eq = pg.functions.eq
    
    zeros = [0, 0.0, np.float(0), np.int(0)]
    if sys.version[0] < '3':
        zeros.append(long(0))
    for i,x in enumerate(zeros):
        for y in zeros[i:]:
            assert eq(x, y)
            assert eq(y, x)
    
    assert eq(np.nan, np.nan)
    
    # test 
    class NotEq(object):
        def __eq__(self, x):
            return False
        
    noteq = NotEq()
    assert eq(noteq, noteq) # passes because they are the same object
    assert not eq(noteq, NotEq())


    # Should be able to test for equivalence even if the test raises certain
    # exceptions
    class NoEq(object):
        def __init__(self, err):
            self.err = err
        def __eq__(self, x):
            raise self.err
        
    noeq1 = NoEq(AttributeError())
    noeq2 = NoEq(ValueError())
    noeq3 = NoEq(Exception())
    
    assert eq(noeq1, noeq1)
    assert not eq(noeq1, noeq2)
    assert not eq(noeq2, noeq1)
    with pytest.raises(Exception):
        eq(noeq3, noeq2)

    # test array equivalence
    # note that numpy has a weird behavior here--np.all() always returns True
    # if one of the arrays has size=0; eq() will only return True if both arrays
    # have the same shape.
    a1 = np.zeros((10, 20)).astype('float')
    a2 = a1 + 1
    a3 = a2.astype('int')
    a4 = np.empty((0, 20))
    assert not eq(a1, a2)
    assert not eq(a1, a3)
    assert not eq(a1, a4)

    assert eq(a2, a3)
    assert not eq(a2, a4)
    
    assert not eq(a3, a4)
    
    assert eq(a4, a4.copy())
    assert not eq(a4, a4.T)

    
if __name__ == '__main__':
    test_interpolateArray()