from pyqtgraph.Qt import QtCore

def test_qbytearray():
    qba = QtCore.QByteArray(b'abcdefghijk')
    mv = memoryview(qba)