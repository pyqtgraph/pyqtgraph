import numpy as np
import pyqtgraph as pg
from pyqtgraph.tests import assertImageApproved


def test_PlotCurveItem():
    p = pg.GraphicsWindow()
    v = p.addViewBox()
    p.resize(200, 150)
    data = np.array([1,4,2,3,np.inf,5,7,6,-np.inf,8,10,9,np.nan,-1,-2,0])
    c = pg.PlotCurveItem(data)
    v.addItem(c)
    v.autoRange()
    
    assert np.allclose(v.viewRange(), [[-1.1457564053237301, 16.145756405323731], [-3.076811473165955, 11.076811473165955]])
    assertImageApproved(p, 'plotcurveitem/connectall', "Plot curve with all points connected.")
    
    c.setData(data, connect='pairs')
    assertImageApproved(p, 'plotcurveitem/connectpairs', "Plot curve with pairs connected.")
    
    c.setData(data, connect='finite')
    assertImageApproved(p, 'plotcurveitem/connectfinite', "Plot curve with finite points connected.")
    
    c.setData(data, connect=np.array([1,1,1,0,1,1,0,0,1,0,0,0,1,1,0,0]))
    assertImageApproved(p, 'plotcurveitem/connectarray', "Plot curve with connection array.")
    


if __name__ == '__main__':
    test_PlotCurveItem()
