# -*- coding: utf-8 -*-
from ..Qt import QtCore
import traceback

class Mutex(QtCore.QMutex):
    """
    Subclass of QMutex that provides useful debugging information during
    deadlocks--tracebacks are printed for both the code location that is
    attempting to lock the mutex as well as the location that has already
    acquired the lock.

    Also provides __enter__ and __exit__ methods for use in "with" statements.
    """
    def __init__(self, *args, **kargs):
        if kargs.get('recursive', False):
            args = (QtCore.QMutex.Recursive,)
        QtCore.QMutex.__init__(self, *args)
        self.l = QtCore.QMutex()  ## for serializing access to self.tb
        self.tb = []
        self.debug = True ## True to enable debugging functions

    def tryLock(self, timeout=None, id=None):
        if timeout is None:
            locked = QtCore.QMutex.tryLock(self)
        else:
            locked = QtCore.QMutex.tryLock(self, timeout)

        if self.debug and locked:
            self.l.lock()
            try:
                if id is None:
                    self.tb.append(''.join(traceback.format_stack()[:-1]))
                else:
                    self.tb.append("  " + str(id))
                #print 'trylock', self, len(self.tb)
            finally:
                self.l.unlock()
        return locked

    def lock(self, id=None):
        c = 0
        waitTime = 5000  # in ms
        while True:
            if self.tryLock(waitTime, id):
                break
            c += 1
            if self.debug:
                self.l.lock()
                try:
                    print("Waiting for mutex lock (%0.1f sec). Traceback follows:"
                          % (c*waitTime/1000.))
                    traceback.print_stack()
                    if len(self.tb) > 0:
                        print("Mutex is currently locked from:\n")
                        print(self.tb[-1])
                    else:
                        print("Mutex is currently locked from [???]")
                finally:
                    self.l.unlock()
        #print 'lock', self, len(self.tb)

    def unlock(self):
        QtCore.QMutex.unlock(self)
        if self.debug:
            self.l.lock()
            try:
                #print 'unlock', self, len(self.tb)
                if len(self.tb) > 0:
                    self.tb.pop()
                else:
                    raise Exception("Attempt to unlock mutex before it has been locked")
            finally:
                self.l.unlock()

    def depth(self):
        self.l.lock()
        n = len(self.tb)
        self.l.unlock()
        return n

    def traceback(self):
        self.l.lock()
        try:
            ret = self.tb[:]
        finally:
            self.l.unlock()
        return ret

    def __exit__(self, *args):
        self.unlock()

    def __enter__(self):
        self.lock()
        return self
