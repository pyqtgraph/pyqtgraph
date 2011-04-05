# -*- coding: utf-8 -*-
from PyQt4 import QtCore
from ptime import time

class SignalProxy(QtCore.QObject):
    """Object which collects rapid-fire signals and condenses them
    into a single signal. Used, for example, to prevent a SpinBox
    from generating multiple signals when the mouse wheel is rolled
    over it."""
    
    def __init__(self, source, signal, delay=0.3):
        """Initialization arguments:
        source - Any QObject that will emit signal, or None if signal is new style
        signal - Output of QtCore.SIGNAL(...), or obj.signal for new style
        delay - Time (in seconds) to wait for signals to stop before emitting (default 0.3s)"""
        
        QtCore.QObject.__init__(self)
        if source is None:
            signal.connect(self.signalReceived)
            self.signal = QtCore.SIGNAL('signal')
        else:
            source.connect(source, signal, self.signalReceived)
            self.signal = signal
        self.delay = delay
        self.args = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.flush)
        self.block = False
        
    def setDelay(self, delay):
        self.delay = delay
        
    def signalReceived(self, *args):
        """Received signal. Cancel previous timer and store args to be forwarded later."""
        if self.block:
            return
        self.args = args
        self.timer.stop()
        self.timer.start((self.delay*1000)+1)
        
    def flush(self):
        """If there is a signal queued up, send it now."""
        if self.args is None or self.block:
            return False
        self.emit(self.signal, *self.args)
        self.args = None
        return True
        
    def disconnect(self):
        self.block = True
    

def proxyConnect(source, signal, slot, delay=0.3):
    """Connect a signal to a slot with delay. Returns the SignalProxy
    object that was created. Be sure to store this object so it is not
    garbage-collected immediately."""
    sp = SignalProxy(source, signal, delay)
    if source is None:
        sp.connect(sp, QtCore.SIGNAL('signal'), slot)
    else:
        sp.connect(sp, signal, slot)
    return sp
    
    
if __name__ == '__main__':
    from PyQt4 import QtGui
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    spin = QtGui.QSpinBox()
    win.setCentralWidget(spin)
    win.show()
    
    def fn(*args):
        print "Got signal:", args
    
    proxy = proxyConnect(spin, QtCore.SIGNAL('valueChanged(int)'), fn)
    
        