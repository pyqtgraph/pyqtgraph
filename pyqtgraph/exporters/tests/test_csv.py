"""
SVG export test
"""
from __future__ import division, print_function, absolute_import
import pyqtgraph as pg
import csv
import os
import tempfile

app = pg.mkQApp()


def approxeq(a, b):
    return (a-b) <= ((a + b) * 1e-6)


def test_CSVExporter():
    tempfilename = tempfile.NamedTemporaryFile(suffix='.csv').name
    print("using %s as a temporary file" % tempfilename)
    
    plt = pg.plot()
    y1 = [1,3,2,3,1,6,9,8,4,2]
    plt.plot(y=y1, name='myPlot')
    
    y2 = [3,4,6,1,2,4,2,3,5,3,5,1,3]
    x2 = pg.np.linspace(0, 1.0, len(y2))
    plt.plot(x=x2, y=y2)
    
    y3 = [1,5,2,3,4,6,1,2,4,2,3,5,3]
    x3 = pg.np.linspace(0, 1.0, len(y3)+1)
    plt.plot(x=x3, y=y3, stepMode=True)
    
    ex = pg.exporters.CSVExporter(plt.plotItem)
    ex.export(fileName=tempfilename)

    r = csv.reader(open(tempfilename, 'r'))
    lines = [line for line in r]
    header = lines.pop(0)
    assert header == ['myPlot_x', 'myPlot_y', 'x0001', 'y0001', 'x0002', 'y0002']
    
    i = 0
    for vals in lines:
        vals = list(map(str.strip, vals))
        assert (i >= len(y1) and vals[0] == '') or approxeq(float(vals[0]), i) 
        assert (i >= len(y1) and vals[1] == '') or approxeq(float(vals[1]), y1[i]) 
        
        assert (i >= len(x2) and vals[2] == '') or approxeq(float(vals[2]), x2[i])
        assert (i >= len(y2) and vals[3] == '') or approxeq(float(vals[3]), y2[i])
        
        assert (i >= len(x3) and vals[4] == '') or approxeq(float(vals[4]), x3[i])
        assert (i >= len(y3) and vals[5] == '') or approxeq(float(vals[5]), y3[i])
        i += 1

    os.unlink(tempfilename)

if __name__ == '__main__':
    test_CSVExporter()
