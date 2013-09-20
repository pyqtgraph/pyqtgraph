# -*- coding: utf-8 -*-
"""
Test for pyqtgraph/function.py
"""

import test
import time
from pyqtgraph.widgets import ValueLabel
from pyqtgraph.python2_3 import asUnicode as u

class ValueLabelTests(test.TestCase):

    init_param=[{}, #1
               {'suffix':'W'}, #2
               {'suffix':'W', 'siPrefix':True}, #3
               {'suffix':'W', 'error':True}, #4
               {'suffix':'W', 'error':True, 'siPrefix':True}, #5
               {'suffix':'W', 'error':True, 'errorType':'max'}, #6
               {'suffix':'W', 'error':True, 'errorType':'stdDev'}, #7
               {'suffix':'W', 'error':True, 'errorType':'stdErr'}, #8
               ]

    value = [u('5.7e+03'), #1
             u('5.7e+03 W'), #2
             u('5.68 kW'), #3
             u('5.68e+03 ± 6e+01 W'), #4
             u('&nbsp;5.68&nbsp;&nbsp;±&nbsp;0.06&nbsp;kW'), #5
             u('5.68e+03 ± 6e+01 W'), #4
             u('5.7e+03 ± 0 W'), #4
             u('5.7e+03 ± 0 W'), #4
             ]

    time1 = [u('2e+02'), #1
             u('2e+02 W'), #2
             u('200 W'), #3
             u('2e+02 ± 3e+01 W'), #4
             u('&nbsp;&nbsp;200&nbsp;&nbsp;±&nbsp;&nbsp;30&nbsp;&nbsp;W'), #5
             u('2e+02 ± 6e+01 W'), #6
             u('2e+02 ± 8e+01 W'), #7
             u('2e+02 ± 5e+01 W'), #8
             ]

    time2 = [u('3.5e+02'), #1
             u('3.5e+02 W'), #2
             u('350 W'), #3
             u('3.5e+02 ± 2e+01 W'), #4
             u('&nbsp;&nbsp;350&nbsp;&nbsp;±&nbsp;&nbsp;20&nbsp;&nbsp;W'), #5
             u('3.5e+02 ± 6e+01 W'), #6
             u('4e+02 ± 2e+02 W'), #7
             u('3.5e+02 ± 7e+01 W'), #8
             ]

    time3 = [u('6.5e+02'), #1
             u('6.5e+02 W'), #2
             u('650 W'), #3
             u('6.5e+02 ± 2e+01 W'), #4
             u('&nbsp;&nbsp;650&nbsp;&nbsp;±&nbsp;&nbsp;20&nbsp;&nbsp;W'), #5
             u('6.5e+02 ± 4e+01 W'), #6
             u('6e+02 ± 2e+02 W'), #7
             u('6.5e+02 ± 7e+01 W'), #8
             ]

   
    def test_static(self):
        for i in xrange(len(self.init_param)):
            lw = ValueLabel.ValueLabel(**self.init_param[i])
            lw.setValue(5678.78,56)
            self.assertEqual(lw.generateText(),self.value[i])

    def  test_dynamic(self):
        for i in xrange(len(self.init_param)):
            lw = ValueLabel.ValueLabel(averageTime=0.01,**self.init_param[i])
            lw.setValue(100,56)
            lw.setValue(200,4)
            lw.setValue(300,37)
            self.assertEqual(lw.generateText(),self.time1[i])
            time.sleep(0.005)
            lw.setValue(400,32)
            lw.setValue(500,5)
            lw.setValue(600,12)
            self.assertEqual(lw.generateText(),self.time2[i])
            time.sleep(0.005)
            lw.setValue(700,43)
            lw.setValue(800,32)
            lw.setValue(900,1)
            self.assertEqual(lw.generateText(),self.time3[i])


if __name__ == '__main__':
    from pyqtgraph.Qt import QtGui
    app = QtGui.QApplication([])
    test.unittest.main()
