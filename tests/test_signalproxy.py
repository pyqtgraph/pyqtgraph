import pytest

from pyqtgraph.Qt import QtCore, mkQApp
from pyqtgraph import SignalProxy


class Sender(QtCore.QObject):
    signalSend = QtCore.Signal()

    def __init__(self, parent=None):
        super(Sender, self).__init__(parent)


class Receiver(QtCore.QObject):

    def __init__(self, parent=None):
        super(Receiver, self).__init__(parent)
        self.counter = 0

    def slotReceive(self):
        self.counter += 1


@pytest.fixture
def qapp():
    app = mkQApp()
    if app is None:
        app = mkQApp()
    yield app
    app.processEvents(QtCore.QEventLoop.AllEvents, 100)


def test_signal_proxy_slot(qapp):
    """Test the normal work mode of SignalProxy with `signal` and `slot`"""
    sender = Sender(parent=qapp)
    receiver = Receiver(parent=qapp)
    proxy = SignalProxy(sender.signalSend, delay=0.0, rateLimit=0.6,
                        slot=receiver.slotReceive)

    assert proxy.blockSignal is False
    assert proxy is not None
    assert sender is not None
    assert receiver is not None

    sender.signalSend.emit()
    proxy.flush()
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)

    assert receiver.counter > 0


def test_signal_proxy_disconnect_slot(qapp):
    """Test the disconnect of SignalProxy with `signal` and `slot`"""
    sender = Sender(parent=qapp)
    receiver = Receiver(parent=qapp)
    proxy = SignalProxy(sender.signalSend, delay=0.0, rateLimit=0.6,
                        slot=receiver.slotReceive)

    assert proxy.blockSignal is False
    assert proxy is not None
    assert sender is not None
    assert receiver is not None
    assert proxy.slot is not None

    proxy.disconnect()
    assert proxy.slot is None
    assert proxy.blockSignal is True

    sender.signalSend.emit()
    proxy.flush()
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)

    assert receiver.counter == 0


def test_signal_proxy_no_slot_start(qapp):
    """Test the connect mode of SignalProxy without slot at start`"""
    sender = Sender(parent=qapp)
    receiver = Receiver(parent=qapp)
    proxy = SignalProxy(sender.signalSend, delay=0.0, rateLimit=0.6)

    assert proxy.blockSignal is True
    assert proxy is not None
    assert sender is not None
    assert receiver is not None

    sender.signalSend.emit()
    proxy.flush()
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)
    assert receiver.counter == 0

    # Start a connect
    proxy.connectSlot(receiver.slotReceive)
    assert proxy.blockSignal is False
    sender.signalSend.emit()
    proxy.flush()
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)
    assert receiver.counter > 0

    # An additional connect should raise an AssertionError
    with pytest.raises(AssertionError):
        proxy.connectSlot(receiver.slotReceive)


def test_signal_proxy_slot_block(qapp):
    """Test the block mode of SignalProxy with `signal` and `slot`"""
    sender = Sender(parent=qapp)
    receiver = Receiver(parent=qapp)
    proxy = SignalProxy(sender.signalSend, delay=0.0, rateLimit=0.6,
                        slot=receiver.slotReceive)

    assert proxy.blockSignal is False
    assert proxy is not None
    assert sender is not None
    assert receiver is not None

    with proxy.block():
        sender.signalSend.emit()
        sender.signalSend.emit()
        sender.signalSend.emit()
        proxy.flush()
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)

        assert receiver.counter == 0

    sender.signalSend.emit()
    proxy.flush()
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)

    assert receiver.counter > 0
