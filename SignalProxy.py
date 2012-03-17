# -*- coding: utf-8 -*-
from Qt import QtCore
from ptime import time
import ThreadsafeTimer

__all__ = ['SignalProxy']

class SignalProxy(QtCore.QObject):
    """Object which collects rapid-fire signals and condenses them
    into a single signal. Used, for example, to prevent a SpinBox
    from generating multiple signals when the mouse wheel is rolled
    over it.
    
    Emits sigDelayed after input signals have stopped for a certain period of time.
    """
    
    sigDelayed = QtCore.Signal(object)
    
    def __init__(self, signal, delay=0.3, slot=None):
        """Initialization arguments:
        signal - a bound Signal or pyqtSignal instance
        delay - Time (in seconds) to wait for signals to stop before emitting (default 0.3s)
        slot - Optional function to connect sigDelayed to.
        """
        
        QtCore.QObject.__init__(self)
        signal.connect(self.signalReceived)
        self.signal = signal
        self.delay = delay
        self.args = None
        self.timer = ThreadsafeTimer.ThreadsafeTimer()
        self.timer.timeout.connect(self.flush)
        self.block = False
        self.slot = slot
        if slot is not None:
            self.sigDelayed.connect(slot)
        
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
        #self.emit(self.signal, *self.args)
        self.sigDelayed.emit(self.args)
        self.args = None
        self.timer.stop()
        return True
        
    def disconnect(self):
        self.block = True
        try:
            self.signal.disconnect(self.signalReceived)
        except:
            pass
        try:
            self.sigDelayed.disconnect(self.slot)
        except:
            pass
   
   

#def proxyConnect(source, signal, slot, delay=0.3):
    #"""Connect a signal to a slot with delay. Returns the SignalProxy
    #object that was created. Be sure to store this object so it is not
    #garbage-collected immediately."""
    #sp = SignalProxy(source, signal, delay)
    #if source is None:
        #sp.connect(sp, QtCore.SIGNAL('signal'), slot)
    #else:
        #sp.connect(sp, signal, slot)
    #return sp
    
    
if __name__ == '__main__':
    from Qt import QtGui
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    spin = QtGui.QSpinBox()
    win.setCentralWidget(spin)
    win.show()
    
    def fn(*args):
        print "Raw signal:", args
    def fn2(*args):
        print "Delayed signal:", args
    
    
    spin.valueChanged.connect(fn)
    #proxy = proxyConnect(spin, QtCore.SIGNAL('valueChanged(int)'), fn)
    proxy = SignalProxy(spin.valueChanged, delay=0.5, slot=fn2)
        