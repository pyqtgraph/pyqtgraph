import importlib

from ..Qt import QT_LIB, QtGui

GL_VENDOR = 7936
GL_RENDERER = 7937
GL_VERSION = 7938


def print_version(funcs):
    glGetString = funcs.glGetString
    print('VENDOR:', glGetString(GL_VENDOR))
    print('RENDERER:', glGetString(GL_RENDERER))
    print('VERSION:', glGetString(GL_VERSION))


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

if QT_LIB == 'PySide2':
    funcs = ctx.functions()
elif QT_LIB == 'PyQt5':
    profile = QtGui.QOpenGLVersionProfile()
    profile.setVersion(2, 0)
    funcs = ctx.versionFunctions(profile)
elif QT_LIB in ['PyQt6', 'PySide6']:
    QtOpenGL = importlib.import_module(f'{QT_LIB}.QtOpenGL')
    profile = QtOpenGL.QOpenGLVersionProfile()
    profile.setVersion(2, 0)
    funcs_factory = QtOpenGL.QOpenGLVersionFunctionsFactory()
    funcs = funcs_factory.get(profile, ctx)

print('isOpenGLES:', ctx.isOpenGLES())
print_version(funcs)
print_extensions(ctx)
