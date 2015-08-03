from __future__ import print_function, absolute_import, division
from pyqtgraph.util import six
from pyqtgraph.multiprocess.parallelizer import Tasker

def test_Tasker_creation():
    # This is a smoketest to make sure that Tasker does not blow up because
    # python2/3 incompatibilities
    tasker = Tasker(parallelizer=None, process=None, tasks=None,
                    kwds={'1': 2, '3': 4})
