from pyqtgraph.Qt import QtCore

class Mixin:
    def __init__(self):
        print('Mixin.__init__')

class Object(QtCore.QObject):
    def __init__(self):
        print('Object.__init__')
        QtCore.QObject.__init__(self)

class Right(Object, Mixin):
    def __init__(self):
        Object.__init__(self)
        Mixin.__init__(self)

class Left(Mixin, Object):
    def __init__(self):
        Object.__init__(self)
        Mixin.__init__(self)

def test_right_mixin():
    x = Right()

def test_left_mixin():
    x = Left()

if __name__ == '__main__':
    print('Right')
    Right()
    print('Left')
    Left()
    print('Done')

