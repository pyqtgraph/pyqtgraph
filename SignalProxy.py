# -*- coding: utf-8 -*-
from PyQt4 import QtCore
from ptime import time

def proxyConnect(source, signal, slot, delay=0.3):
    """Connect a signal to a slot with delay. Returns the SignalProxy
    object that was created. Be sure to store this object so it is not
    garbage-collected immediately."""
    sp = SignalProxy(source, signal, delay)
    sp.connect(sp, signal, slot)
    return sp

class SignalProxy(QtCore.QObject):
    """Object which collects rapid-fire signals and condenses them
    into a single signal. Used, for example, to prevent a SpinBox
    from generating multiple signals when the mouse wheel is rolled
    over it."""
    
    def __init__(self, source, signal, delay=0.3):
        QtCore.QObject.__init__(self)
        source.connect(source, signal, self.signal)
        self.delay = delay
        self.waitUntil = 0
        self.args = None
        self.timers = 0
        self.signal = signal
        self.block = False
        
    def setDelay(self, delay):
        self.delay = delay
        
    def flush(self):
        """If there is a signal queued up, send it now."""
        if self.args is None or self.block:
            return False
        if self.block:
            return
        self.emit(self.signal, *self.args)
        self.args = None
        return True
        
        
    def signal(self, *args):
        """Received signal, queue to be forwarded later."""
        if self.block:
            return
        self.waitUntil = time() + self.delay
        self.args = args
        self.timers += 1
        QtCore.QTimer.singleShot((self.delay*1000)+1, self.tryEmit)
        
    def tryEmit(self):
        """Emit signal if it has been long enougn since receiving the last signal."""
        if self.args is None or self.block:
            return False
        self.timers -= 1
        t = time()
        if t >= self.waitUntil:
            return self.flush()
        else:
            if self.timers == 0:
                self.timers += 1
                QtCore.QTimer.singleShot((self.waitUntil - t) * 1000, self.tryEmit)
        return True
                
        
    def disconnect(self):
        self.block = True
    