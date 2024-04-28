import sys

import numpy as np
import pytest

from pyqtgraph import setConfigOptions
from pyqtgraph.functions import makeARGB as real_makeARGB

from .makeARGB_test_data import EXPECTED_OUTPUTS, INPUTS, LEVELS, LUTS


try:
    import numba
except ImportError:
    pass

try:
    import cupy
except ImportError:
    pass


def _makeARGB(*args, **kwds):
    img, alpha = real_makeARGB(*args, **kwds)
    if kwds.get('useRGBA'):  # endian independent
        out = img
    elif sys.byteorder == 'little':  # little-endian ARGB32 to B,G,R,A
        out = img
    else:  # big-endian ARGB32 to B,G,R,A
        out = img[..., [3, 2, 1, 0]]
    return out, alpha


@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.parametrize('acceleration', [
        pytest.param('numpy'),
        pytest.param('cupy', marks=pytest.mark.skipif('cupy' not in sys.modules, reason="CuPy not available")),
        pytest.param('numba', marks=pytest.mark.skipif('numba' not in sys.modules, reason="numba not available"))
    ]
)
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize('in_fmt', ["2D", "RGB", "RGBA"])
@pytest.mark.parametrize('level_name', [None, 'SIMPLE', 'RGB', 'RGBA'])
@pytest.mark.parametrize('lut_type', [None, np.uint8, np.uint16])
@pytest.mark.parametrize('scale', [None, 232, 13333])
@pytest.mark.parametrize('use_rgba', [True, False])
def test_makeARGB_against_generated_references(acceleration, dtype, in_fmt, level_name, lut_type, scale, use_rgba):
    if acceleration == "numba":
        setConfigOptions(useCupy=False, useNumba=True)
    elif acceleration == "cupy":
        setConfigOptions(useCupy=True, useNumba=False)
    else:
        setConfigOptions(useCupy=False, useNumba=False)

    if dtype == np.float32 and level_name is None:
        pytest.skip(f"{dtype=} is not compatable with {level_name=}")

    data = INPUTS[(dtype, in_fmt)]
    levels = LEVELS.get(level_name, None)
    lut = LUTS.get(lut_type, None)

    key = (dtype, in_fmt, level_name, lut_type, scale, use_rgba)
    expectation = EXPECTED_OUTPUTS[key]
    if isinstance(expectation, type) and issubclass(expectation, Exception):
        with pytest.raises(expectation) as exc_info:
            _makeARGB(data, lut=lut, levels=levels, scale=scale, useRGBA=use_rgba)
        assert exc_info.type is expectation, f"makeARGB({key!r}) was supposed to raise {expectation}"
    else:
        output, alpha = _makeARGB(data, lut=lut, levels=levels, scale=scale, useRGBA=use_rgba)
        assert (
            output == expectation
        ).all(), f"Incorrect _makeARGB({key!r}) output! Expected:\n{expectation!r}\n  Got:\n{output!r}"
    setConfigOptions(useCupy=False, useNumba=False)


@pytest.mark.parametrize('makeARGB_args,makeARGB_kwargs',
    [
        pytest.param(
            [np.zeros((2,), dtype='float')],
            dict(),
            marks=pytest.mark.xfail(
                raises=TypeError,
                strict=True,
                reason="invalid image shape (ndim=1)"
            )
        ),
        pytest.param(
            [np.zeros((2, 2, 7), dtype='float')],
            dict(),
            marks=pytest.mark.xfail(
                raises=TypeError,
                strict=True,
                reason="invalid_image_shape (ndim=3)"
            )
        ),
        pytest.param(
            [np.zeros((2, 2, 7), dtype='float')],
            dict(),
            marks=pytest.mark.xfail(
                raises=Exception,
                strict=True,
                reason="float images require levels arg"
            )
        ),
        pytest.param(
            [np.zeros((2, 2), dtype='float')],
            dict(levels=[1]),
            marks=pytest.mark.xfail(
                raises=Exception,
                strict=True,
                reason="bad levels arg"
            )
        ),
        pytest.param(
            [np.zeros((2, 2), dtype='float')],
            dict(levels=[1, 2, 3]),
            marks=pytest.mark.xfail(
                raises=Exception,
                strict=True,
                reason="bad levels arg"
            )
        ),
        pytest.param(
            [np.zeros((2, 2))],
            dict(lut=np.zeros((10, 3), dtype='ubyte'), levels=[(0, 1)] * 3),
            marks=pytest.mark.xfail(
                raises=Exception,
                strict=True,
                reason="can't mix 3-channel levels and LUT"
            ),
        ),
        pytest.param(
            [np.zeros((2, 2, 3), dtype='float')],
            dict(levels=[(1, 2)] * 4),
            marks=pytest.mark.xfail(
                raises=Exception,
                strict=True,
                reason="multichannel levels must have same number of channels as image"
            )
        ),
        pytest.param(
            [np.zeros((2, 2, 3), dtype='float')],
            dict(levels=np.zeros([3, 2, 2])),
            marks=pytest.mark.xfail(
                raises=Exception,
                strict=True,
                reason="3d levels not allowed"
            ),
        )
    ]
)
def test_makeARGB_exceptions(makeARGB_args, makeARGB_kwargs):
    _makeARGB(*makeARGB_args, **makeARGB_kwargs)


def test_makeARGB_with_nans():
    # NaNs conversion to 0 is undefined in the C-standard
    # see: https://github.com/pyqtgraph/pyqtgraph/issues/2969#issuecomment-2014924400
    # see: https://stackoverflow.com/questions/10366485/problems-casting-nan-floats-to-int

    # nans in image
    # 2d input image, one pixel is nan
    im1 = np.ones((10, 12))
    im1[3, 5] = np.nan
    im2, alpha = _makeARGB(im1, levels=(0, 1))
    assert alpha
    assert im2[3, 5, 3] == 0  # nan pixel is transparent
    assert im2[0, 0, 3] == 255  # doesn't affect other pixels

    # With masking nans disabled, the nan pixel shouldn't be transparent
    im2, alpha = _makeARGB(im1, levels=(0, 1), maskNans=False)
    assert im2[3, 5, 3] == 255  # nan pixel is transparent

    # 3d RGB input image, any color channel of a pixel is nan
    im1 = np.ones((10, 12, 3))
    im1[3, 5, 1] = np.nan
    im2, alpha = _makeARGB(im1, levels=(0, 1))
    assert alpha
    assert im2[3, 5, 3] == 0  # nan pixel is transparent
    assert im2[0, 0, 3] == 255  # doesn't affect other pixels

    # 3d RGBA input image, any color channel of a pixel is nan
    im1 = np.ones((10, 12, 4))
    im1[3, 5, 1] = np.nan
    im2, alpha = _makeARGB(im1, levels=(0, 1), useRGBA=True)
    assert alpha
    assert im2[3, 5, 3] == 0  # nan pixel is transparent


def checkArrays(a, b):
    # because pytest output is difficult to read for arrays
    if not np.all(a == b):
        comp = []
        for i in range(a.shape[0]):
            if a.shape[1] > 1:
                comp.append('[')
            for j in range(a.shape[1]):
                m = a[i, j] == b[i, j]
                comp.append('%d,%d  %s %s  %s%s' %
                            (i, j, str(a[i, j]).ljust(15), str(b[i, j]).ljust(15),
                             m, ' ********' if not np.all(m) else ''))
            if a.shape[1] > 1:
                comp.append(']')
        raise ValueError("arrays do not match:\n%s" % '\n'.join(comp))


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
        raise ValueError('Invalid check array ndim')


def test_makeARGB_with_human_readable_code():
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


    # uint8 data tests

    im1 = np.arange(256).astype('ubyte').reshape(256, 1)
    im2, alpha = _makeARGB(im1, levels=(0, 255))
    checkImage(im2, im1, alpha, False)

    im3, alpha = _makeARGB(im1, levels=(0.0, 255.0))
    checkImage(im3, im1, alpha, False)

    im4, alpha = _makeARGB(im1, levels=(255, 0))
    checkImage(im4, 255 - im1, alpha, False)

    im5, alpha = _makeARGB(np.concatenate([im1] * 3, axis=1), levels=[(0, 255), (0.0, 255.0), (255, 0)])
    checkImage(im5, np.concatenate([im1, im1, 255 - im1], axis=1), alpha, False)

    im2, alpha = _makeARGB(im1, levels=(128, 383))
    checkImage(im2[:128], 0, alpha, False)
    checkImage(im2[128:], im1[:128], alpha, False)

    # uint8 data + uint8 LUT
    lut = np.arange(256)[::-1].astype(np.uint8)
    im2, alpha = _makeARGB(im1, lut=lut)
    checkImage(im2, lut, alpha, False)

    # lut larger than maxint
    lut = np.arange(511).astype(np.uint8)
    im2, alpha = _makeARGB(im1, lut=lut)
    checkImage(im2, lut[::2], alpha, False)

    # lut smaller than maxint
    lut = np.arange(128).astype(np.uint8)
    im2, alpha = _makeARGB(im1, lut=lut)
    checkImage(im2, np.linspace(0, 127.5, 256, dtype='ubyte'), alpha, False)

    # lut + levels
    lut = np.arange(256)[::-1].astype(np.uint8)
    im2, alpha = _makeARGB(im1, lut=lut, levels=[-128, 384])
    checkImage(im2, np.linspace(191.5, 64.5, 256, dtype='ubyte'), alpha, False)

    im2, alpha = _makeARGB(im1, lut=lut, levels=[64, 192])
    checkImage(im2, np.clip(np.linspace(384.5, -127.5, 256), 0, 255).astype('ubyte'), alpha, False)

    # uint8 data + uint16 LUT
    lut = np.arange(4096)[::-1].astype(np.uint16) // 16
    im2, alpha = _makeARGB(im1, lut=lut)
    checkImage(im2, np.arange(256)[::-1].astype('ubyte'), alpha, False)

    # uint8 data + float LUT
    lut = np.linspace(10., 137., 256)
    im2, alpha = _makeARGB(im1, lut=lut)
    checkImage(im2, lut.astype('ubyte'), alpha, False)

    # uint8 data + 2D LUT
    lut = np.zeros((256, 3), dtype='ubyte')
    lut[:, 0] = np.arange(256)
    lut[:, 1] = np.arange(256)[::-1]
    lut[:, 2] = 7
    im2, alpha = _makeARGB(im1, lut=lut)
    checkImage(im2, lut[:, None, ::-1], alpha, False)

    # check useRGBA
    im2, alpha = _makeARGB(im1, lut=lut, useRGBA=True)
    checkImage(im2, lut[:, None, :], alpha, False)

    # uint16 data tests
    im1 = np.arange(0, 2 ** 16, 256).astype('uint16')[:, None]
    im2, alpha = _makeARGB(im1, levels=(512, 2 ** 16))
    checkImage(im2, np.clip(np.linspace(-2, 253, 256), 0, 255).astype('ubyte'), alpha, False)

    lut = (np.arange(512, 2 ** 16)[::-1] // 256).astype('ubyte')
    im2, alpha = _makeARGB(im1, lut=lut, levels=(512, 2 ** 16 - 256))
    checkImage(im2, np.clip(np.linspace(257, 2, 256), 0, 255).astype('ubyte'), alpha, False)

    lut = np.zeros(2 ** 16, dtype='ubyte')
    lut[1000:1256] = np.arange(256)
    lut[1256:] = 255
    im1 = np.arange(1000, 1256).astype('uint16')[:, None]
    im2, alpha = _makeARGB(im1, lut=lut)
    checkImage(im2, np.arange(256).astype('ubyte'), alpha, False)

    # float data tests
    im1 = np.linspace(1.0, 17.0, 256)[:, None]
    im2, alpha = _makeARGB(im1, levels=(5.0, 13.0))
    checkImage(im2, np.clip(np.linspace(-128, 383, 256), 0, 255).astype('ubyte'), alpha, False)

    lut = (np.arange(1280)[::-1] // 10).astype('ubyte')
    im2, alpha = _makeARGB(im1, lut=lut, levels=(1, 17))
    checkImage(im2, np.linspace(127.5, 0, 256).astype('ubyte'), alpha, False)
