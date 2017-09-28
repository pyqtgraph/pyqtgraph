from pyqtgraph.Qt import QtGui, QtCore
import numpy
import pyqtgraph as pg
from OpenGL.GL import *
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

class glGradientLegendItem(GLGraphicsItem):
    def __init__(self, **kwds):
        """All keyword arguments are passed to setData()"""
        GLGraphicsItem.__init__(self)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = None
        self.stops = None
        self.colors = None
        self.gradient = None
        self.setData(**kwds)

    def setData(self, **kwds):
        args = ['size', 'pos', 'gradient', 'labels']
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        self.antialias = False
        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])
                #self.vbo.pop(arg, None)
        if self.gradient is not None and hasattr(self.gradient,"getStops"):
            self.stops, self.colors = self.gradient.getStops("float")

            self.colors = self.colors/255.0

        self.update()

    def paint(self):
        if self.pos is None:
            return
        self.setupGLState()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0.0, self.view().width(),self.view().height(), 0.0, -1.0, 10.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_CULL_FACE)

        glClear(GL_DEPTH_BUFFER_BIT)

        glTranslate(self.pos[0],self.pos[1],0)
        glScale(self.size[0],self.size[1],0)
        glBegin(GL_QUAD_STRIP)
        for p, c in zip(self.stops,self.colors):
            glColor3f(*c)
            glVertex2d(0,1-p)
            glColor3f(*c)
            glVertex2d(1,1-p)
        glEnd()

        #scaling and translate doent work on rendertext
        glColor3f(1,1,1)
        for k in self.labels:
            x = 1.1*self.size[0]+self.pos[0]
            y = self.size[1]-self.labels[k]*self.size[1]+self.pos[1]+8
            self.view().renderText(x,y,k)



        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

if __name__ == '__main__':
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    import numpy
    import sys
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()

    def fn(x, y):
        return np.cos((x**2 + y**2)**0.5)
    n=50
    y = numpy.linspace(-10,10,50)
    x = numpy.linspace(-10,10,50)
    d = (x**2 + y[:,None]**2)**0.5
    z = 10*numpy.cos(d) / (d+1)
    c = z/z.max()

    cmap = pg.ColorMap((0,.25,.5,.75,1),((0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0)))
    c = cmap.map(c)

    for i in range(n):
        yi = y[i]
        di = d[i]
        zi = z[i]
        ci = c[i]
        pos = numpy.vstack([x,numpy.ones(len(x))*yi,zi]).transpose()
        plt = gl.GLLinePlotItem(pos=pos, color=ci, antialias=True)
        w.addItem(plt)

    gll = glGradientLegendItem(pos = (10,10), size=(50,300), gradient=cmap,labels= {".1":.1,".5":.5, ".7":.7, "1":1})
    w.addItem(gll)

    QtGui.QApplication.instance().exec_()

