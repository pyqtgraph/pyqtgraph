# Image-based testing borrowed from vispy

"""
Procedure for unit-testing with images:

    Run individual test scripts with the PYQTGRAPH_AUDIT environment variable set:

       $ PYQTGRAPH_AUDIT=1 python pyqtgraph/graphicsItems/tests/test_PlotCurveItem.py

    Any failing tests will display the test results, standard image, and the
    differences between the two. If the test result is bad, then press (f)ail.
    If the test result is good, then press (p)ass and the new image will be
    saved to the test-data directory.

    To check all test results regardless of whether the test failed, set the
    environment variable PYQTGRAPH_AUDIT_ALL=1.
"""

import time
import os
import sys
import inspect
import warnings
import numpy as np

from pathlib import Path

from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph import functions as fn
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph import ImageItem, TextItem


tester = None

# Convenient stamp used for ensuring image orientation is correct
axisImg = [
    "            1         1 1        ",
    "          1 1         1 1 1 1    ",
    "            1   1 1 1 1 1 1 1 1 1",
    "            1         1 1 1 1    ",
    "    1     1 1 1       1 1        ",
    "  1   1                          ",
    "  1   1                          ",
    "    1                            ",
    "                                 ",
    "    1                            ",
    "    1                            ",
    "    1                            ",
    "1 1 1 1 1                        ",
    "1 1 1 1 1                        ",
    "  1 1 1                          ",
    "  1 1 1                          ",
    "    1                            ",
    "    1                            ",
]
axisImg = np.array([map(int, row[::2].replace(' ', '0')) for row in axisImg])



def getTester():
    global tester
    if tester is None:
        tester = ImageTester()
    return tester


def getImageFromWidget(widget):

    # just to be sure the widget size is correct (new window may be resized):
    QtGui.QApplication.processEvents()

    qimg = QtGui.QImage(widget.size(), QtGui.QImage.Format.Format_ARGB32)
    qimg.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(qimg)
    widget.render(painter)
    painter.end()

    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    return fn.qimage_to_ndarray(qimg).copy()


def assertImageApproved(image, standardFile, message=None, **kwargs):
    """Check that an image test result matches a pre-approved standard.

    If the result does not match, then the user can optionally invoke a GUI
    to compare the images and decide whether to fail the test or save the new
    image as the standard.

    Run the test with the environment variable PYQTGRAPH_AUDIT=1 to bring up
    the auditing GUI.

    Parameters
    ----------
    image : (h, w, 4) ndarray
    standardFile : str
        The name of the approved test image to check against. This file name
        is relative to the root of the pyqtgraph test-data repository and will
        be automatically fetched.
    message : str
        A string description of the image. It is recommended to describe
        specific features that an auditor should look for when deciding whether
        to fail a test.

    Extra keyword arguments are used to set the thresholds for automatic image
    comparison (see ``assertImageMatch()``).
    """
    if isinstance(image, QtGui.QWidget):
        # just to be sure the widget size is correct (new window may be resized):
        QtGui.QApplication.processEvents()

        graphstate = scenegraphState(image, standardFile)
        image = getImageFromWidget(image)

    if message is None:
        code = inspect.currentframe().f_back.f_code
        message = "%s::%s" % (code.co_filename, code.co_name)

    # Make sure we have a test data repo available
    dataPath = getTestDataDirectory()

    # Read the standard image if it exists
    stdFileName = os.path.join(dataPath, standardFile + '.png')
    if not os.path.isfile(stdFileName):
        stdImage = None
    else:
        qimg = QtGui.QImage(stdFileName)
        qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        stdImage = fn.qimage_to_ndarray(qimg).copy()
        del qimg

    # If the test image does not match, then we go to audit if requested.
    try:
        if stdImage is None:
            raise Exception("No reference image saved for this test.")
        if image.shape[2] != stdImage.shape[2]:
            raise Exception("Test result has different channel count than standard image"
                            "(%d vs %d)" % (image.shape[2], stdImage.shape[2]))
        if image.shape != stdImage.shape:
            # Allow im1 to be an integer multiple larger than im2 to account
            # for high-resolution displays
            ims1 = np.array(image.shape).astype(float)
            ims2 = np.array(stdImage.shape).astype(float)
            sr = ims1 / ims2 if ims1[0] > ims2[0] else ims2 / ims1
            if (sr[0] != sr[1] or not np.allclose(sr, np.round(sr)) or
               sr[0] < 1):
                raise TypeError("Test result shape %s is not an integer factor"
                                    " different than standard image shape %s." %
                                (ims1, ims2))
            sr = np.round(sr).astype(int)
            image = fn.downsample(image, sr[0], axis=(0, 1)).astype(image.dtype)

        assertImageMatch(image, stdImage, **kwargs)

        if bool(os.getenv('PYQTGRAPH_PRINT_TEST_STATE', False)):
            print(graphstate)

        if os.getenv('PYQTGRAPH_AUDIT_ALL') == '1':
            raise Exception("Image test passed, but auditing due to PYQTGRAPH_AUDIT_ALL evnironment variable.")
    except Exception:
        if os.getenv('PYQTGRAPH_AUDIT') == '1' or os.getenv('PYQTGRAPH_AUDIT_ALL') == '1':
            sys.excepthook(*sys.exc_info())
            getTester().test(image, stdImage, message)
            stdPath = os.path.dirname(stdFileName)
            print('Saving new standard image to "%s"' % stdFileName)
            if not os.path.isdir(stdPath):
                os.makedirs(stdPath)
            qimg = fn.ndarray_to_qimage(image, QtGui.QImage.Format.Format_RGBA8888)
            qimg.save(stdFileName)
            del qimg
        else:
            if stdImage is None:
                raise Exception("Test standard %s does not exist. Set "
                                "PYQTGRAPH_AUDIT=1 to add this image." % stdFileName)
            if os.getenv('CI') is not None:
                standardFile = os.path.join(os.getenv("SCREENSHOT_DIR", "screenshots"), standardFile)
                saveFailedTest(image, stdImage, standardFile)
            print(graphstate)
            raise


def assertImageMatch(im1, im2, minCorr=None, pxThreshold=50.,
                       pxCount=-1, maxPxDiff=None, avgPxDiff=None,
                       imgDiff=None):
    """Check that two images match.

    Images that differ in shape or dtype will fail unconditionally.
    Further tests for similarity depend on the arguments supplied.

    By default, images may have no pixels that gave a value difference greater
    than 50.

    Parameters
    ----------
    im1 : (h, w, 4) ndarray
        Test output image
    im2 : (h, w, 4) ndarray
        Test standard image
    minCorr : float or None
        Minimum allowed correlation coefficient between corresponding image
        values (see numpy.corrcoef)
    pxThreshold : float
        Minimum value difference at which two pixels are considered different
    pxCount : int or None
        Maximum number of pixels that may differ. Default is 0, on Windows some
        tests have a value of 2.
    maxPxDiff : float or None
        Maximum allowed difference between pixels
    avgPxDiff : float or None
        Average allowed difference between pixels
    imgDiff : float or None
        Maximum allowed summed difference between images

    """
    assert im1.ndim == 3
    assert im1.shape[2] == 4
    assert im1.dtype == im2.dtype

    if pxCount == -1:
        pxCount = 0
    
    diff = im1.astype(float) - im2.astype(float)
    if imgDiff is not None:
        assert np.abs(diff).sum() <= imgDiff

    pxdiff = diff.max(axis=2)  # largest value difference per pixel
    mask = np.abs(pxdiff) >= pxThreshold
    if pxCount is not None:
        assert mask.sum() <= pxCount

    maskedDiff = diff[mask]
    if maxPxDiff is not None and maskedDiff.size > 0:
        assert maskedDiff.max() <= maxPxDiff
    if avgPxDiff is not None and maskedDiff.size > 0:
        assert maskedDiff.mean() <= avgPxDiff

    if minCorr is not None:
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(im1.ravel(), im2.ravel())[0, 1]
        assert corr >= minCorr


def saveFailedTest(data, expect, filename):
    # concatenate data, expect, and diff into a single image
    ds = data.shape
    es = expect.shape

    shape = (max(ds[0], es[0]) + 4, ds[1] + es[1] + 8 + max(ds[1], es[1]), 4)
    img = np.empty(shape, dtype=np.ubyte)
    img[..., :3] = 100
    img[..., 3] = 255

    img[2:2+ds[0], 2:2+ds[1], :ds[2]] = data
    img[2:2+es[0], ds[1]+4:ds[1]+4+es[1], :es[2]] = expect

    diff = makeDiffImage(data, expect)
    img[2:2+diff.shape[0], -diff.shape[1]-2:-2] = diff

    png = makePng(data)  # change `img` to `data` to save just the failed image
    directory = os.path.dirname(filename)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(filename + ".png", "wb") as png_file:
        png_file.write(png)
    print("\nImage comparison failed. Test result: %s %s   Expected result: "
        "%s %s" % (data.shape, data.dtype, expect.shape, expect.dtype))


def makePng(img):
    """Given an array like (H, W, 4), return a PNG-encoded byte string.
    """
    io = QtCore.QBuffer()
    qim = fn.ndarray_to_qimage(img, QtGui.QImage.Format.Format_RGBX8888)
    qim.save(io, 'PNG')
    return bytes(io.data().data())


def makeDiffImage(im1, im2):
    """Return image array showing the differences between im1 and im2.

    Handles images of different shape. Alpha channels are not compared.
    """
    ds = im1.shape
    es = im2.shape

    diff = np.empty((max(ds[0], es[0]), max(ds[1], es[1]), 4), dtype=int)
    diff[..., :3] = 128
    diff[..., 3] = 255
    diff[:ds[0], :ds[1], :min(ds[2], 3)] += im1[..., :3]
    diff[:es[0], :es[1], :min(es[2], 3)] -= im2[..., :3]
    diff = np.clip(diff, 0, 255).astype(np.ubyte)
    return diff


class ImageTester(QtGui.QWidget):
    """Graphical interface for auditing image comparison tests.
    """
    def __init__(self):
        self.lastKey = None
        
        QtGui.QWidget.__init__(self)
        self.resize(1200, 800)
        #self.showFullScreen()
        
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        
        self.view = GraphicsLayoutWidget()
        self.layout.addWidget(self.view, 0, 0, 1, 2)

        self.label = QtGui.QLabel()
        self.layout.addWidget(self.label, 1, 0, 1, 2)
        self.label.setWordWrap(True)
        font = QtGui.QFont("monospace", 14, QtGui.QFont.Bold)
        self.label.setFont(font)

        self.passBtn = QtGui.QPushButton('Pass')
        self.failBtn = QtGui.QPushButton('Fail')
        self.layout.addWidget(self.passBtn, 2, 0)
        self.layout.addWidget(self.failBtn, 2, 1)
        self.passBtn.clicked.connect(self.passTest)
        self.failBtn.clicked.connect(self.failTest)

        self.views = (self.view.addViewBox(row=0, col=0),
                      self.view.addViewBox(row=0, col=1),
                      self.view.addViewBox(row=0, col=2))
        labelText = ['test output', 'standard', 'diff']
        for i, v in enumerate(self.views):
            v.setAspectLocked(1)
            v.invertY()
            v.image = ImageItem(axisOrder='row-major')
            v.image.setAutoDownsample(True)
            v.addItem(v.image)
            v.label = TextItem(labelText[i])
            v.setBackgroundColor(0.5)

        self.views[1].setXLink(self.views[0])
        self.views[1].setYLink(self.views[0])
        self.views[2].setXLink(self.views[0])
        self.views[2].setYLink(self.views[0])

    def test(self, im1, im2, message):
        """Ask the user to decide whether an image test passes or fails.
        
        This method displays the test image, reference image, and the difference
        between the two. It then blocks until the user selects the test output
        by clicking a pass/fail button or typing p/f. If the user fails the test,
        then an exception is raised.
        """
        self.show()
        if im2 is None:
            message += '\nImage1: %s %s   Image2: [no standard]' % (im1.shape, im1.dtype)
            im2 = np.zeros((1, 1, 3), dtype=np.ubyte)
        else:
            message += '\nImage1: %s %s   Image2: %s %s' % (im1.shape, im1.dtype, im2.shape, im2.dtype)
        self.label.setText(message)
        
        self.views[0].image.setImage(im1)
        self.views[1].image.setImage(im2)
        diff = makeDiffImage(im1, im2)

        self.views[2].image.setImage(diff)
        self.views[0].autoRange()

        while True:
            QtGui.QApplication.processEvents()
            lastKey = self.lastKey
            
            self.lastKey = None
            if lastKey in ('f', 'esc') or not self.isVisible():
                raise Exception("User rejected test result.")
            elif lastKey == 'p':
                break
            time.sleep(0.03)

        for v in self.views:
            v.image.setImage(np.zeros((1, 1, 3), dtype=np.ubyte))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.lastKey = 'esc'
        else:
            self.lastKey = str(event.text()).lower()

    def passTest(self):
        self.lastKey = 'p'

    def failTest(self):
        self.lastKey = 'f'


def getTestDataRepo():
    warnings.warn(
        "Test data data repo has been merged with the main repo"
        "use getTestDataDirectory() instead, this method will be removed"
        "in a future version of pyqtgraph",
        DeprecationWarning, stacklevel=2
    )
    return getTestDataDirectory()


def getTestDataDirectory():
    dataPath = Path(__file__).absolute().parent / "images"
    return dataPath.as_posix()


def scenegraphState(view, name):
    """Return information about the scenegraph for debugging test failures.
    """
    state = "====== Scenegraph state for %s ======\n" % name
    state += "view size: %dx%d\n" % (view.width(), view.height())
    state += "view transform:\n" + indent(transformStr(view.transform()), "  ")
    for item in view.scene().items():
        if item.parentItem() is None:
            state += itemState(item) + '\n'
    return state

    
def itemState(root):
    state = str(root) + '\n'
    from pyqtgraph import ViewBox
    state += 'bounding rect: ' + str(root.boundingRect()) + '\n'
    if isinstance(root, ViewBox):
        state += "view range: " + str(root.viewRange()) + '\n'
    state += "transform:\n" + indent(transformStr(root.transform()).strip(), "  ") + '\n'
    for item in root.childItems():
        state += indent(itemState(item).strip(), "    ") + '\n'
    return state

    
def transformStr(t):
    return ("[%0.2f %0.2f %0.2f]\n"*3) % (t.m11(), t.m12(), t.m13(), t.m21(), t.m22(), t.m23(), t.m31(), t.m32(), t.m33())


def indent(s, pfx):
    return '\n'.join(pfx+line for line in s.split('\n'))


class TransposedImageItem(ImageItem):
    # used for testing image axis order; we can test row-major and col-major using
    # the same test images
    def __init__(self, *args, **kwds):
        self.__transpose = kwds.pop('transpose', False)
        ImageItem.__init__(self, *args, **kwds)
    def setImage(self, image=None, **kwds):
        if image is not None and self.__transpose is True:
            image = np.swapaxes(image, 0, 1)
        return ImageItem.setImage(self, image, **kwds)
