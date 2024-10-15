from ..Qt import QtGui
from ..Qt import OpenGLConstants as GLC
from ..Qt import OpenGLHelpers


def print_version(funcs):
    glGetString = funcs.glGetString
    print('VENDOR:', glGetString(GLC.GL_VENDOR))
    print('RENDERER:', glGetString(GLC.GL_RENDERER))
    print('VERSION:', glGetString(GLC.GL_VERSION))
    print('GLSL_VERSION:', glGetString(GLC.GL_SHADING_LANGUAGE_VERSION))


def print_extensions(ctx):
    extensions = sorted([ext.data().decode() for ext in ctx.extensions()])
    print("Extensions:")
    for ext in extensions:
        print(f"   {ext}")


app = QtGui.QGuiApplication([])
surf = QtGui.QOffscreenSurface()
surf.create()
ctx = QtGui.QOpenGLContext()
ctx.create()
ctx.makeCurrent(surf)

print("openGLModuleType:", QtGui.QOpenGLContext.openGLModuleType())
print('isOpenGLES:', ctx.isOpenGLES())
glfn = OpenGLHelpers.getFunctions(ctx)
print_version(glfn)
print_extensions(ctx)
