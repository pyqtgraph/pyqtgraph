import numpy
import pyqtgraph as pg
from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem

class GLGradientLegendItem(GLGraphicsItem):
    """
    Displays legend colorbar on the screen.
    """
    def __init__(self, **kwds):
        """
        Arguments:
            pos: position of the colorbar on the screen, from the top left corner, in pixels
            size: size of the colorbar without the text, in pixels
            gradient: a pg.ColorMap used to color the colorbar
            labels: a dict of "text":value to display next to the colorbar.
                The value corresponds to a position in the gradient from 0 to 1.
            fontColor: sets the color of the texts
            #Todo:
                size as percentage
                legend title
        """
        GLGraphicsItem.__init__(self)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = (10,10)
        self.size = (10,100)
        self.fontColor = (1.0, 1.0, 1.0, 1.0)
        self.stops = None
        self.colors = None
        self.gradient = None
        self.setData(**kwds)

    def setData(self, **kwds):
        args = ['size', 'pos', 'gradient', 'labels', 'fontColor']
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))

        self.antialias = False

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

        ##todo: add title
        ##todo: take more gradient types
        if self.gradient is not None and hasattr(self.gradient, "getStops"):
            self.stops, self.colors = self.gradient.getStops("float")
            self.colors = self.colors/255.0

        self.update()

    def paint(self):
        if self.pos is None or self.stops is None:
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

        #draw the colorbar
        glTranslate(self.pos[0],self.pos[1],0)
        glScale(self.size[0],self.size[1],0)
        glBegin(GL_QUAD_STRIP)
        for p, c in zip(self.stops,self.colors):
            glColor3f(*c)
            glVertex2d(0,1-p)
            glColor3f(*c)
            glVertex2d(1,1-p)
        glEnd()

        #draw labels
        #scaling and translate doent work on rendertext
        glColor4f(*self.fontColor)
        for labelText,labelPosition in self.labels.items():
            ## todo: draw ticks, position text properly
            x = 1.1*self.size[0]+self.pos[0]
            y = self.size[1]- labelPosition *self.size[1]+self.pos[1]+8
            ##todo: fonts
            self.view().renderText(x,y,labelText)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)



