from math import cos, radians, sin, tan
import importlib
import warnings

from OpenGL import GL
import numpy as np

from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets, QT_LIB

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

class GLViewMixin:
    def __init__(self, *args, rotationMethod='euler', **kwargs):
        """
        Mixin class providing functionality for GLViewWidget

        ================ ==============================================================
        **Arguments:**
        rotationMethod   (str): Mechanism to drive the rotation method, options are
                         'euler' and 'quaternion'. Defaults to 'euler'.
        ================ ==============================================================
        """
        super().__init__(*args, **kwargs)

        if rotationMethod not in ["euler", "quaternion"]:
            raise ValueError("Rotation method should be either 'euler' or 'quaternion'")
        
        self.opts = {
            'rotationMethod': rotationMethod
        }
        self.reset()
        self.items = []
        
        self.noRepeatKeys = [QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_PageUp, QtCore.Qt.Key.Key_PageDown]
        self.keysPressed = {}
        self.keyTimer = QtCore.QTimer()
        self.keyTimer.timeout.connect(self.evalKeyState)

        self._modelViewStack = []
        self._projectionStack = []
        self.default_vao = QtOpenGL.QOpenGLVertexArrayObject(self)

    def deviceWidth(self):
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr)

    def deviceHeight(self):
        dpr = self.devicePixelRatioF()
        return int(self.height() * dpr)

    def reset(self):
        """
        Initialize the widget state or reset the current state to the original state.
        """
        self.opts['center'] = Vector(0,0,0)  ## will always appear at the center of the widget
        self.opts['distance'] = 10.0         ## distance of camera from center
        self.opts['fov'] = 60                ## horizontal field of view in degrees

        if self.opts['rotationMethod'] == 'quaternion':
            self.opts['rotation'] = QtGui.QQuaternion(1,0,0,0)  ## camera rotation (quaternion:wxyz)
        else:
            self.opts['elevation'] = 30      ## camera's angle of elevation in degrees
            self.opts['azimuth'] = 45        ## camera's azimuthal angle in degrees
                                             ## (rotation around z-axis 0 points along x-axis)

        self.setBackgroundColor(getConfigOption('background'))

    def addItem(self, item):
        self.items.append(item)

        if self.isValid():
            item.initialize()
                
        item._setView(self)
        self.update()
        
    def removeItem(self, item):
        """
        Remove the item from the scene.
        """
        self.items.remove(item)
        item._setView(None)
        self.update()

    def clear(self):
        """
        Remove all items from the scene.
        """
        for item in self.items:
            item._setView(None)
        self.items = []
        self.update()        
        
    def initializeGL(self):
        """
        Initialize items that were not initialized during addItem().
        """
        ctx = self.context()
        fmt = ctx.format()
        if ctx.isOpenGLES():
            warnings.warn(
                f"pyqtgraph.opengl is primarily tested against OpenGL Desktop"
                f" but OpenGL {fmt.version()} ES detected",
                RuntimeWarning,
                stacklevel=2
            )
        elif fmt.version() < (2, 1):
            verString = GL.glGetString(GL.GL_VERSION)
            raise RuntimeError(
                "pyqtgraph.opengl: Requires >= OpenGL 2.1; Found %s" % verString
            )

        # Core profile requires a non-default VAO
        if fmt.profile() == QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile:
            if not self.default_vao.isCreated():
                self.default_vao.create()
                self.default_vao.bind()

        for item in self.items:
            if not item.isInitialized():
                item.initialize()
        
    def setBackgroundColor(self, *args, **kwds):
        """
        Set the background color of the widget. Accepts the same arguments as
        :func:`~pyqtgraph.mkColor`.
        """
        self.opts['bgcolor'] = fn.mkColor(*args, **kwds).getRgbF()
        self.update()
        
    def getViewport(self):
        return (0, 0, self.width(), self.height())
        
    def setProjection(self, region, viewport):
        m = self.projectionMatrix(region, viewport)
        self._projectionStack.clear()
        self._projectionStack.append(m)

    def projectionMatrix(self, region, viewport):
        x0, y0, w, h = viewport
        dist = self.opts['distance']
        fov = self.opts['fov']
        nearClip = dist * 0.001
        farClip = dist * 1000.

        r = nearClip * tan(0.5 * radians(fov))
        t = r * h / w

        ## Note that X0 and width in these equations must be the values used in viewport
        left  = r * ((region[0]-x0) * (2.0/w) - 1)
        right = r * ((region[0]+region[2]-x0) * (2.0/w) - 1)
        bottom = t * ((region[1]-y0) * (2.0/h) - 1)
        top    = t * ((region[1]+region[3]-y0) * (2.0/h) - 1)

        tr = QtGui.QMatrix4x4()
        tr.frustum(left, right, bottom, top, nearClip, farClip)
        return tr
        
    def setModelview(self):
        m = self.viewMatrix()
        self._modelViewStack.clear()
        self._modelViewStack.append(m)
        
    def viewMatrix(self):
        tr = QtGui.QMatrix4x4()
        tr.translate( 0.0, 0.0, -self.opts['distance'])
        if self.opts['rotationMethod'] == 'quaternion':
            tr.rotate(self.opts['rotation'])
        else:
            # default rotation method
            tr.rotate(self.opts['elevation']-90, 1, 0, 0)
            tr.rotate(self.opts['azimuth']+90, 0, 0, -1)  
        center = self.opts['center']
        tr.translate(-center.x(), -center.y(), -center.z())
        return tr

    def currentModelView(self):
        return self._modelViewStack[-1]

    def currentProjection(self):
        return self._projectionStack[-1]

    def itemsAt(self, region=None):
        """
        Return a list of the items displayed in the region (x, y, w, h)
        relative to the widget.        
        """
        region = (region[0], self.height()-(region[1]+region[3]), region[2], region[3])
        viewport = self.getViewport()
        
        #buf = np.zeros(100000, dtype=np.uint)
        buf = GL.glSelectBuffer(100000)
        try:
            GL.glRenderMode(GL.GL_SELECT)
            GL.glInitNames()
            GL.glPushName(0)
            self._itemNames = {}
            self.paint(region=region, viewport=viewport, useItemNames=True)
            
        finally:
            hits = GL.glRenderMode(GL.GL_RENDER)
            
        items = [(h.near, h.names[0]) for h in hits]
        items.sort(key=lambda i: i[0])
        return [self._itemNames[i[1]] for i in items]
    
    def paintGL(self):
        # when called by Qt, glViewport has already been called
        # with device pixel ratio taken of
        region = self.getViewport()
        self.paint(region=region, viewport=region)

    def paint(self, *, region, viewport, useItemNames=False):
        """
        It is caller's responsibility to call glViewport prior to calling this method.
        region specifies the sub-region of viewport that should be rendered.
        """
        self.setProjection(region, viewport)
        self.setModelview()
        bgcolor = self.opts['bgcolor']
        GL.glClearColor(*bgcolor)
        GL.glClear( GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT )
        self.drawItemTree(useItemNames=useItemNames)
        
    def drawItemTree(self, item=None, useItemNames=False):
        if item is None:
            items = [x for x in self.items if x.parentItem() is None]
        else:
            items = item.childItems()
            items.append(item)
        items.sort(key=lambda a: a.depthValue())
        for i in items:
            if not i.visible():
                continue
            if i is item:
                try:
                    if useItemNames:
                        GL.glLoadName(i._id)
                        self._itemNames[i._id] = i

                    # The GLGraphicsItem(s) making use of QPainter end
                    # up indirectly unbinding the default VAO, so we
                    # rebind it before each GLGraphicsItem.
                    if self.default_vao.isCreated():
                        self.default_vao.bind()

                    i.paint()
                except:
                    from .. import debug
                    debug.printExc()
                    print("Error while drawing item %s." % str(item))
            else:
                self._modelViewStack.append(self.currentModelView() * i.transform())
                try:
                    self.drawItemTree(i, useItemNames=useItemNames)
                finally:
                    self._modelViewStack.pop()

    def setCameraPosition(self, pos=None, distance=None, elevation=None, azimuth=None, rotation=None):
        if rotation is not None:
            # Alternatively, we could define that rotation overrides elevation and azimuth
            if elevation is not None:
                raise ValueError("cannot set both rotation and elevation")
            if azimuth is not None:
                raise ValueError("cannot set both rotation and azimuth")

        if pos is not None:
            self.opts['center'] = pos
        if distance is not None:
            self.opts['distance'] = distance

        if self.opts['rotationMethod'] == "quaternion":
            # note that "quaternion" mode modifies only opts['rotation']
            if elevation is not None or azimuth is not None:
                eu = self.opts['rotation'].toEulerAngles()
                if azimuth is not None:
                    eu.setZ(-azimuth-90)
                if elevation is not None:
                    eu.setX(elevation-90)
                self.opts['rotation'] = QtGui.QQuaternion.fromEulerAngles(eu)
            if rotation is not None:
                self.opts['rotation'] = rotation
        else:
            # note that "euler" mode modifies only opts['elevation'] and opts['azimuth']
            if elevation is not None:
                self.opts['elevation'] = elevation
            if azimuth is not None:
                self.opts['azimuth'] = azimuth
            if rotation is not None:
                eu = rotation.toEulerAngles()
                self.opts['elevation'] = eu.x() + 90
                self.opts['azimuth'] = -eu.z() - 90

        self.update()
        
    def cameraPosition(self):
        """Return current position of camera based on center, dist, elevation, and azimuth"""
        center = self.opts['center']
        dist = self.opts['distance']
        if self.opts['rotationMethod'] == "quaternion":
            pos = Vector(center - self.opts['rotation'].rotatedVector(Vector(0,0,dist) ))
        else:
            # using 'euler' rotation method
            elev = radians(self.opts['elevation'])
            azim = radians(self.opts['azimuth'])
            pos = Vector(
                center.x() + dist * cos(elev) * cos(azim),
                center.y() + dist * cos(elev) * sin(azim),
                center.z() + dist * sin(elev)
            )
        return pos

    def setCameraParams(self, **kwds):
        valid_keys = {'center', 'rotation', 'distance', 'fov', 'elevation', 'azimuth'}
        if not valid_keys.issuperset(kwds):
            raise ValueError(f'valid keywords are {valid_keys}')

        self.setCameraPosition(pos=kwds.get('center'), distance=kwds.get('distance'),
                               elevation=kwds.get('elevation'), azimuth=kwds.get('azimuth'),
                               rotation=kwds.get('rotation'))
        if 'fov' in kwds:
            self.opts['fov'] = kwds['fov']

    def cameraParams(self):
        valid_keys = ['center', 'distance', 'fov']

        if self.opts['rotationMethod'] == 'quaternion':
            valid_keys.append('rotation')
        else:
            valid_keys.extend(['elevation', 'azimuth'])

        return { k : self.opts[k] for k in valid_keys }

    def orbit(self, azim, elev):
        """Orbits the camera around the center position. *azim* and *elev* are given in degrees."""
        if self.opts['rotationMethod'] == 'quaternion':
            q = QtGui.QQuaternion.fromEulerAngles(
                    elev, -azim, 0
                    ) # rx-ry-rz
            q *= self.opts['rotation']
            self.opts['rotation'] = q
        else: # default euler rotation method
            self.opts['azimuth'] += azim
            self.opts['elevation'] = fn.clip_scalar(self.opts['elevation'] + elev, -90., 90.)
        self.update()
        
    def pan(self, dx, dy, dz, relative='global'):
        """
        Moves the center (look-at) position while holding the camera in place. 
        
        ==============  =======================================================
        **Arguments:**
        *dx*            Distance to pan in x direction
        *dy*            Distance to pan in y direction
        *dz*            Distance to pan in z direction
        *relative*      String that determines the direction of dx,dy,dz. 
                        If "global", then the global coordinate system is used.
                        If "view", then the z axis is aligned with the view
                        direction, and x and y axes are in the plane of the
                        view: +x points right, +y points up. 
                        If "view-upright", then x is in the global xy plane and
                        points to the right side of the view, y is in the
                        global xy plane and orthogonal to x, and z points in
                        the global z direction.
        ==============  =======================================================
        
        Distances are scaled roughly such that a value of 1.0 moves
        by one pixel on screen.
        """
        if relative == 'global':
            self.opts['center'] += QtGui.QVector3D(dx, dy, dz)
        elif relative == 'view-upright':
            cPos = self.cameraPosition()
            cVec = self.opts['center'] - cPos
            dist = cVec.length()  ## distance from camera to center
            xDist = dist * 2. * tan(0.5 * radians(self.opts['fov']))  ## approx. width of view at distance of center point
            xScale = xDist / self.width()
            zVec = QtGui.QVector3D(0,0,1)
            xVec = QtGui.QVector3D.crossProduct(zVec, cVec).normalized()
            yVec = QtGui.QVector3D.crossProduct(xVec, zVec).normalized()
            self.opts['center'] = self.opts['center'] + xVec * xScale * dx + yVec * xScale * dy + zVec * xScale * dz
        elif relative == 'view':
            # pan in plane of camera

            if self.opts['rotationMethod'] == 'quaternion':
                # obtain basis vectors
                qc = self.opts['rotation'].conjugated()
                xv = qc.rotatedVector( Vector(1,0,0) )
                yv = qc.rotatedVector( Vector(0,1,0) )
                zv = qc.rotatedVector( Vector(0,0,1) )

                scale_factor = self.pixelSize( self.opts['center'] )

                # apply translation
                self.opts['center'] += scale_factor * (xv*-dx + yv*dy + zv*dz)
            else: # use default euler rotation method
                elev = radians(self.opts['elevation'])
                azim = radians(self.opts['azimuth'])
                fov = radians(self.opts['fov'])
                dist = (self.opts['center'] - self.cameraPosition()).length()
                fov_factor = tan(fov / 2) * 2
                scale_factor = dist * fov_factor / self.width()
                z = scale_factor * cos(elev) * dy
                x = scale_factor * (sin(azim) * dx - sin(elev) * cos(azim) * dy)
                y = scale_factor * (cos(azim) * dx + sin(elev) * sin(azim) * dy)
                self.opts['center'] += QtGui.QVector3D(x, -y, z)
        else:
            raise ValueError("relative argument must be global, view, or view-upright")
        
        self.update()
        
    def pixelSize(self, pos):
        """
        Return the approximate size of a screen pixel at the location pos
        Pos may be a Vector or an (N,3) array of locations
        """
        cam = self.cameraPosition()
        if isinstance(pos, np.ndarray):
            cam = np.array(cam).reshape((1,)*(pos.ndim-1)+(3,))
            dist = ((pos-cam)**2).sum(axis=-1)**0.5
        else:
            dist = (pos-cam).length()
        xDist = dist * 2. * tan(0.5 * radians(self.opts['fov']))
        return xDist / self.width()

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos
        
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        
    def mouseReleaseEvent(self, ev):
        pass
        
    def wheelEvent(self, ev):
        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            self.opts['fov'] *= 0.999**delta
        else:
            self.opts['distance'] *= 0.999**delta
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
      
    def keyReleaseEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except KeyError:
                self.keysPressed = {}
            self.evalKeyState()
        
    def evalKeyState(self):
        speed = 2.0
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key.Key_Right:
                    self.orbit(azim=-speed, elev=0)
                elif key == QtCore.Qt.Key.Key_Left:
                    self.orbit(azim=speed, elev=0)
                elif key == QtCore.Qt.Key.Key_Up:
                    self.orbit(azim=0, elev=-speed)
                elif key == QtCore.Qt.Key.Key_Down:
                    self.orbit(azim=0, elev=speed)
                elif key == QtCore.Qt.Key.Key_PageUp:
                    pass
                elif key == QtCore.Qt.Key.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

    def readQImage(self):
        """
        Read the current buffer pixels out as a QImage.
        """
        return self.grabFramebuffer()
        
    def renderToArray(self, size, format=GL.GL_BGRA, type=GL.GL_UNSIGNED_BYTE, textureSize=1024, padding=256):
        w,h = map(int, size)
        
        self.makeCurrent()

        texwidth = textureSize

        fbo = QtOpenGL.QOpenGLFramebufferObject(texwidth, texwidth,
                    QtOpenGL.QOpenGLFramebufferObject.Attachment.CombinedDepthStencil,
                    GL.GL_TEXTURE_2D)

        output = np.empty((h, w, 4), dtype=np.ubyte)
        data = np.empty((texwidth, texwidth, 4), dtype=np.ubyte)

        try:
            p2 = 2 * padding
            for x in range(-padding, w-padding, texwidth-p2):
                for y in range(-padding, h-padding, texwidth-p2):
                    x2 = min(x+texwidth, w+padding)
                    y2 = min(y+texwidth, h+padding)
                    w2 = x2-x
                    h2 = y2-y
                    
                    fbo.bind()
                    GL.glViewport(0, 0, w2, h2)
                    self.paint(region=(x, h-y-h2, w2, h2), viewport=(0, 0, w, h))  # only render sub-region
                    
                    fbo.bind()
                    GL.glReadPixels(0, 0, texwidth, texwidth, format, type, data)
                    data_yflip = data[::-1, ...]
                    output[y+padding:y2-padding, x+padding:x2-padding] = data_yflip[-(h2-padding):-padding, padding:w2-padding]
                    
        finally:
            fbo.release()

        return output


class GLViewWidget(GLViewMixin, QtWidgets.QOpenGLWidget):
    def __init__(self, *args, devicePixelRatio=None, **kwargs):
        """
        Basic widget for displaying 3D data
          - Rotation/scale controls
          - Axis/grid display
          - Export options

        ================ ==============================================================
        **Arguments:**
        parent           (QObject, optional): Parent QObject. Defaults to None.
        devicePixelRatio No longer in use. High-DPI displays should automatically
                         detect the correct resolution.
        rotationMethod   (str): Mechanism to drive the rotation method, options are
                         'euler' and 'quaternion'. Defaults to 'euler'.
        ================ ==============================================================
        """
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
