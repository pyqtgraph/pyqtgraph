import weakref
from time import perf_counter

from .functions import SignalBlock
from .Qt import QtCore
from .ThreadsafeTimer import ThreadsafeTimer

__all__ = ['SignalProxy']


class SignalProxy(QtCore.QObject):
    """Object which collects rapid-fire signals and condenses them
    into a single signal or a rate-limited stream of signals.
    Used, for example, to prevent a SpinBox from generating multiple
    signals when the mouse wheel is rolled over it.

    Emits sigDelayed after input signals have stopped for a certain period of
    time.
    """

    sigDelayed = QtCore.Signal(object)

    def __init__(self, signal, delay=0.3, rateLimit=0, slot=None, *, threadSafe=True):
        """Initialization arguments:
        signal - a bound Signal or pyqtSignal instance
        delay - Time (in seconds) to wait for signals to stop before emitting (default 0.3s)
        slot - Optional function to connect sigDelayed to.
        rateLimit - (signals/sec) if greater than 0, this allows signals to stream out at a
                    steady rate while they are being received.
        threadSafe - Specify if thread-safety is required. For backwards compatibility, it
                     defaults to True.
        """

        QtCore.QObject.__init__(self)
        self.delay = delay
        self.rateLimit = rateLimit
        self.args = None
        Timer = ThreadsafeTimer if threadSafe else QtCore.QTimer
        self.timer = Timer()
        self.timer.timeout.connect(self.flush)
        self.lastFlushTime = None
        self.signal = signal
        self.signal.connect(self.signalReceived)
        if slot is not None:
            self.blockSignal = False
            self.sigDelayed.connect(slot)
            self.slot = weakref.ref(slot)
        else:
            self.blockSignal = True
            self.slot = None

    def setDelay(self, delay):
        self.delay = delay

    def signalReceived(self, *args):
        """Received signal. Cancel previous timer and store args to be
        forwarded later."""
        if self.blockSignal:
            return
        self.args = args
        if self.rateLimit == 0:
            self.timer.stop()
            self.timer.start(int(self.delay * 1000) + 1)
        else:
            now = perf_counter()
            if self.lastFlushTime is None:
                leakTime = 0
            else:
                lastFlush = self.lastFlushTime
                leakTime = max(0, (lastFlush + (1.0 / self.rateLimit)) - now)

            self.timer.stop()
            self.timer.start(int(min(leakTime, self.delay) * 1000) + 1)

    def flush(self):
        """If there is a signal queued up, send it now."""
        if self.args is None or self.blockSignal:
            return False
        args, self.args = self.args, None
        self.timer.stop()
        self.lastFlushTime = perf_counter()
        self.sigDelayed.emit(args)
        return True

    def disconnect(self):
        self.blockSignal = True
        try:
            self.signal.disconnect(self.signalReceived)
        except:
            pass
        try:
            slot = self.slot()
            if slot is not None:
                self.sigDelayed.disconnect(slot)
        except:
            pass
        finally:
            self.slot = None

    def connectSlot(self, slot):
        """Connect the `SignalProxy` to an external slot"""
        assert self.slot is None, "Slot was already connected!"
        self.slot = weakref.ref(slot)
        self.sigDelayed.connect(slot)
        self.blockSignal = False

    def block(self):
        """Return a SignalBlocker that temporarily blocks input signals to
        this proxy.
        """
        return SignalBlock(self.signal, self.signalReceived)
