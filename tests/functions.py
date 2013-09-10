# -*- coding: utf-8 -*-
"""
Test for pyqtgraph/function.py
"""

import pyqtgraph.functions as fnc
from pyqtgraph.python2_3 import asUnicode as u
import test

class siScale(test.TestCase):
    expected_values=((0.0001,(1e6,u('µ'))),
                    (0.1,(1e3,u('m'))),
                    (3,(1,u(''))),
                    (5432,(1e-3,u('k'))))

    def test_function(self):
        for inp, out in self.expected_values:
            result=fnc.siScale(inp)
            self.assertEqual(result[1],out[1])
            self.assertAlmostEqual(result[0],out[0])

class siEval(test.TestCase):
    expected_values=((u("100 µV"),1e-4),
                    (u("100µV"),1e-4),
                    (u("100uV"),1e-4),
                    (u("100 uV"),1e-4),
                    (u("100μV"),1e-4),
                    (u("100 μV"),1e-4),
                    (u("430MPa"),430e6),
                    (u("-32 PVfds"),-32e15))

    def test_function(self):
        for inp, out in self.expected_values:
            result=fnc.siEval(inp)
            self.assertAlmostEqual(result,out)

class siFormat(test.TestCase):
    expected_values=((((0.0001,), {'suffix':'V'}),u'100 µV'),
                     (((1000,), {'suffix':'V','error':23, 'groupedError':True}),u("1.00 ±  0.02   kV")),
                     (((10,), {'suffix':'V','error':230, 'groupedError':True}),u(" 10  ±  200    V")),
                     (((432432,), {'suffix':'V','error':230,'precision':4 , 'groupedError':True}),u("432.4 ±  0.2    kV")),
                     (((432432,), {'suffix':'V','error':230,'precision':4}),u("432.4 kV ± 230 V")),
                     (((1000,), {'suffix':'V','error':23}),u("1 kV ± 23 V")),
                     (((1000,), {'suffix':'V','error':23, 'groupedError':True}),u("1.00 ±  0.02   kV")),
                     (((432432,), {'suffix':'V','error':230,'precision':5 , 'groupedError':True}),u("432.4  ±  0.2    kV")),
                     (((432432,), {'suffix':'V','error':23000,'precision':5 , 'groupedError':True}),u(" 430   ±   20    kV")),
                     (((432432,), {'suffix':'V','error':30,'precision':5 , 'groupedError':True}),u("432.43 ±  0.03   kV")),
                     (((432432,), {'suffix':'V','error':0.01,'precision':4 , 'groupedError':True}),u("432.4 ± 1e-05   kV")),
                     (((23,), {'suffix':'V','error':45334,'precision':4 , 'groupedError':True}),u(" 20   ± 50000   V")),
                     (((23,), {'suffix':'V','error':4533432424,'precision':4 , 'groupedError':True}),u(" 20   ± 5e+09   V")),
                    )
    
    def test_function(self):
        for inp, out in self.expected_values:
            result=fnc.siFormat(*inp[0],**inp[1])
            self.assertEqual(result,out)


if __name__ == '__main__':
    test.unittest.main()
