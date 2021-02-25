from ..Qt import QtWidgets
from OpenGL.GL import *

class GLTest(QtWidgets.QOpenGLWidget):
    def initializeGL(self):
        print("GL version:" + glGetString(GL_VERSION).decode("utf-8"))
        print("MAX_TEXTURE_SIZE: %d" % glGetIntegerv(GL_MAX_TEXTURE_SIZE))
        print("MAX_3D_TEXTURE_SIZE: %d" % glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE))
        print("Extensions: " + glGetString(GL_EXTENSIONS).decode("utf-8").replace(" ", "\n"))

app = QtWidgets.QApplication([])
GLTest().show()
