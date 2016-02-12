# Image-based testing borrowed from vispy

"""
Procedure for unit-testing with images:

1. Run unit tests at least once; this initializes a git clone of
   pyqtgraph/test-data in ~/.pyqtgraph.

2. Run individual test scripts with the PYQTGRAPH_AUDIT environment variable set:

       $ PYQTGRAPH_AUDIT=1 python pyqtgraph/graphicsItems/tests/test_PlotItem.py

   Any failing tests will
   display the test results, standard image, and the differences between the
   two. If the test result is bad, then press (f)ail. If the test result is
   good, then press (p)ass and the new image will be saved to the test-data
   directory.

3. After adding or changing test images, create a new commit:

        $ cd ~/.pyqtgraph/test-data
        $ git add ...
        $ git commit -a

4. Look up the most recent tag name from the `testDataTag` variable in
   getTestDataRepo() below. Increment the tag name by 1 in the function
   and create a new tag in the test-data repository:

        $ git tag test-data-NNN
        $ git push --tags origin master

    This tag is used to ensure that each pyqtgraph commit is linked to a specific
    commit in the test-data repository. This makes it possible to push new
    commits to the test-data repository without interfering with existing
    tests, and also allows unit tests to continue working on older pyqtgraph
    versions.

    Finally, update the tag name in ``getTestDataRepo`` to the new name.

"""

import time
import os
import sys
import inspect
import base64
from subprocess import check_call, check_output, CalledProcessError
import numpy as np

#from ..ext.six.moves import http_client as httplib
#from ..ext.six.moves import urllib_parse as urllib
import httplib
import urllib
from ..Qt import QtGui, QtCore
from .. import functions as fn
from .. import GraphicsLayoutWidget
from .. import ImageItem, TextItem


# This tag marks the test-data commit that this version of vispy should
# be tested against. When adding or changing test images, create
# and push a new tag and update this variable.
testDataTag = 'test-data-2'


tester = None


def getTester():
    global tester
    if tester is None:
        tester = ImageTester()
    return tester


def assertImageApproved(image, standardFile, message=None, **kwargs):
    """Check that an image test result matches a pre-approved standard.

    If the result does not match, then the user can optionally invoke a GUI
    to compare the images and decide whether to fail the test or save the new
    image as the standard.

    This function will automatically clone the test-data repository into
    ~/.pyqtgraph/test-data. However, it is up to the user to ensure this repository
    is kept up to date and to commit/push new images after they are saved.

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
        w = image
        image = np.zeros((w.height(), w.width(), 4), dtype=np.ubyte)
        qimg = fn.makeQImage(image, alpha=True, copy=False, transpose=False)
        painter = QtGui.QPainter(qimg)
        w.render(painter)
        painter.end()

    if message is None:
        code = inspect.currentframe().f_back.f_code
        message = "%s::%s" % (code.co_filename, code.co_name)

    # Make sure we have a test data repo available, possibly invoking git
    dataPath = getTestDataRepo()

    # Read the standard image if it exists
    stdFileName = os.path.join(dataPath, standardFile + '.png')
    if not os.path.isfile(stdFileName):
        stdImage = None
    else:
        pxm = QtGui.QPixmap()
        pxm.load(stdFileName)
        stdImage = fn.imageToArray(pxm.toImage(), copy=True, transpose=False)

    # If the test image does not match, then we go to audit if requested.
    try:
        if image.shape != stdImage.shape:
            # Allow im1 to be an integer multiple larger than im2 to account
            # for high-resolution displays
            ims1 = np.array(image.shape).astype(float)
            ims2 = np.array(stdImage.shape).astype(float)
            sr = ims1 / ims2
            if (sr[0] != sr[1] or not np.allclose(sr, np.round(sr)) or
               sr[0] < 1):
                raise TypeError("Test result shape %s is not an integer factor"
                                " larger than standard image shape %s." %
                                (ims1, ims2))
            sr = np.round(sr).astype(int)
            image = downsample(image, sr[0], axis=(0, 1)).astype(image.dtype)

        assertImageMatch(image, stdImage, **kwargs)
    except Exception:
        if stdFileName in gitStatus(dataPath):
            print("\n\nWARNING: unit test failed against modified standard "
                  "image %s.\nTo revert this file, run `cd %s; git checkout "
                  "%s`\n" % (stdFileName, dataPath, standardFile))
        if os.getenv('PYQTGRAPH_AUDIT') == '1':
            sys.excepthook(*sys.exc_info())
            getTester().test(image, stdImage, message)
            stdPath = os.path.dirname(stdFileName)
            print('Saving new standard image to "%s"' % stdFileName)
            if not os.path.isdir(stdPath):
                os.makedirs(stdPath)
            img = fn.makeQImage(image, alpha=True, copy=False, transpose=False)
            img.save(stdFileName)
        else:
            if stdImage is None:
                raise Exception("Test standard %s does not exist. Set "
                                "PYQTGRAPH_AUDIT=1 to add this image." % stdFileName)
            else:
                if os.getenv('TRAVIS') is not None:
                    saveFailedTest(image, stdImage, standardFile)
                raise


def assertImageMatch(im1, im2, minCorr=0.9, pxThreshold=50.,
                       pxCount=None, maxPxDiff=None, avgPxDiff=None,
                       imgDiff=None):
    """Check that two images match.

    Images that differ in shape or dtype will fail unconditionally.
    Further tests for similarity depend on the arguments supplied.

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
        Maximum number of pixels that may differ
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
    """Upload failed test images to web server to allow CI test debugging.
    """
    commit, error = check_output(['git', 'rev-parse',  'HEAD'])
    name = filename.split('/')
    name.insert(-1, commit.strip())
    filename = '/'.join(name)
    host = 'data.pyqtgraph.org'

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

    png = _make_png(img)
    conn = httplib.HTTPConnection(host)
    req = urllib.urlencode({'name': filename,
                            'data': base64.b64encode(png)})
    conn.request('POST', '/upload.py', req)
    response = conn.getresponse().read()
    conn.close()
    print("\nImage comparison failed. Test result: %s %s   Expected result: "
          "%s %s" % (data.shape, data.dtype, expect.shape, expect.dtype))
    print("Uploaded to: \nhttp://%s/data/%s" % (host, filename))
    if not response.startswith(b'OK'):
        print("WARNING: Error uploading data to %s" % host)
        print(response)


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
        self.showFullScreen()
        
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

        self.views = (self.view.addViewBox(row=0, col=0),
                      self.view.addViewBox(row=0, col=1),
                      self.view.addViewBox(row=0, col=2))
        labelText = ['test output', 'standard', 'diff']
        for i, v in enumerate(self.views):
            v.setAspectLocked(1)
            v.invertY()
            v.image = ImageItem()
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
        
        self.views[0].image.setImage(im1.transpose(1, 0, 2))
        self.views[1].image.setImage(im2.transpose(1, 0, 2))
        diff = makeDiffImage(im1, im2).transpose(1, 0, 2)

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


def getTestDataRepo():
    """Return the path to a git repository with the required commit checked
    out.

    If the repository does not exist, then it is cloned from
    https://github.com/vispy/test-data. If the repository already exists
    then the required commit is checked out.
    """
    global testDataTag

    dataPath = os.path.expanduser('~/.pyqtgraph/test-data')
    gitPath = 'https://github.com/pyqtgraph/test-data'
    gitbase = gitCmdBase(dataPath)

    if os.path.isdir(dataPath):
        # Already have a test-data repository to work with.

        # Get the commit ID of testDataTag. Do a fetch if necessary.
        try:
            tagCommit = gitCommitId(dataPath, testDataTag)
        except NameError:
            cmd = gitbase + ['fetch', '--tags', 'origin']
            print(' '.join(cmd))
            check_call(cmd)
            try:
                tagCommit = gitCommitId(dataPath, testDataTag)
            except NameError:
                raise Exception("Could not find tag '%s' in test-data repo at"
                                " %s" % (testDataTag, dataPath))
        except Exception:
            if not os.path.exists(os.path.join(dataPath, '.git')):
                raise Exception("Directory '%s' does not appear to be a git "
                                "repository. Please remove this directory." %
                                dataPath)
            else:
                raise

        # If HEAD is not the correct commit, then do a checkout
        if gitCommitId(dataPath, 'HEAD') != tagCommit:
            print("Checking out test-data tag '%s'" % testDataTag)
            check_call(gitbase + ['checkout', testDataTag])

    else:
        print("Attempting to create git clone of test data repo in %s.." %
              dataPath)

        parentPath = os.path.split(dataPath)[0]
        if not os.path.isdir(parentPath):
            os.makedirs(parentPath)

        if os.getenv('TRAVIS') is not None:
            # Create a shallow clone of the test-data repository (to avoid
            # downloading more data than is necessary)
            os.makedirs(dataPath)
            cmds = [
                gitbase + ['init'],
                gitbase + ['remote', 'add', 'origin', gitPath],
                gitbase + ['fetch', '--tags', 'origin', testDataTag,
                           '--depth=1'],
                gitbase + ['checkout', '-b', 'master', 'FETCH_HEAD'],
            ]
        else:
            # Create a full clone
            cmds = [['git', 'clone', gitPath, dataPath]]

        for cmd in cmds:
            print(' '.join(cmd))
            rval = check_call(cmd)
            if rval == 0:
                continue
            raise RuntimeError("Test data path '%s' does not exist and could "
                               "not be created with git. Please create a git "
                               "clone of %s at this path." %
                               (dataPath, gitPath))

    return dataPath


def gitCmdBase(path):
    return ['git', '--git-dir=%s/.git' % path, '--work-tree=%s' % path]


def gitStatus(path):
    """Return a string listing all changes to the working tree in a git
    repository.
    """
    cmd = gitCmdBase(path) + ['status', '--porcelain']
    return check_output(cmd, stderr=None, universal_newlines=True)


def gitCommitId(path, ref):
    """Return the commit id of *ref* in the git repository at *path*.
    """
    cmd = gitCmdBase(path) + ['show', ref]
    try:
        output = check_output(cmd, stderr=None, universal_newlines=True)
    except CalledProcessError:
        print(cmd)
        raise NameError("Unknown git reference '%s'" % ref)
    commit = output.split('\n')[0]
    assert commit[:7] == 'commit '
    return commit[7:]


#import subprocess
#def run_subprocess(command, return_code=False, **kwargs):
    #"""Run command using subprocess.Popen

    #Run command and wait for command to complete. If the return code was zero
    #then return, otherwise raise CalledProcessError.
    #By default, this will also add stdout= and stderr=subproces.PIPE
    #to the call to Popen to suppress printing to the terminal.

    #Parameters
    #----------
    #command : list of str
        #Command to run as subprocess (see subprocess.Popen documentation).
    #return_code : bool
        #If True, the returncode will be returned, and no error checking
        #will be performed (so this function should always return without
        #error).
    #**kwargs : dict
        #Additional kwargs to pass to ``subprocess.Popen``.

    #Returns
    #-------
    #stdout : str
        #Stdout returned by the process.
    #stderr : str
        #Stderr returned by the process.
    #code : int
        #The command exit code. Only returned if ``return_code`` is True.
    #"""
    ## code adapted with permission from mne-python
    #use_kwargs = dict(stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    #use_kwargs.update(kwargs)

    #p = subprocess.Popen(command, **use_kwargs)
    #output = p.communicate()

    ## communicate() may return bytes, str, or None depending on the kwargs
    ## passed to Popen(). Convert all to unicode str:
    #output = ['' if s is None else s for s in output]
    #output = [s.decode('utf-8') if isinstance(s, bytes) else s for s in output]
    #output = tuple(output)

    #if not return_code and p.returncode:
        #print(output[0])
        #print(output[1])
        #err_fun = subprocess.CalledProcessError.__init__
        #if 'output' in inspect.getargspec(err_fun).args:
            #raise subprocess.CalledProcessError(p.returncode, command, output)
        #else:
            #raise subprocess.CalledProcessError(p.returncode, command)
    #if return_code:
        #output = output + (p.returncode,)
    #return output
