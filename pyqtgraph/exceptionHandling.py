"""This module installs a wrapper around sys.excepthook and threading.excepthook that allows multiple
new exception handlers to be registered. 

Optionally, the wrapper also stops exceptions from causing long-term storage 
of local stack frames. This has two major effects:
  - Unhandled exceptions will no longer cause memory leaks
    (If an exception occurs while a lot of data is present on the stack, 
    such as when loading large files, the data would ordinarily be kept
    until the next exception occurs. We would rather release this memory 
    as soon as possible.)
  - Some debuggers may have a hard time handling uncaught exceptions
"""
import sys
import threading
import time
import traceback
from types import SimpleNamespace

callbacks = []
old_callbacks = []
clear_tracebacks = False


def registerCallback(fn):
    """Register a callable to be invoked when there is an unhandled exception.
    The callback will be passed an object with attributes: [exc_type, exc_value, exc_traceback, thread]
    (see threading.excepthook).
    Multiple callbacks will be invoked in the order they were registered.
    """
    callbacks.append(fn)


def unregisterCallback(fn):
    """Unregister a previously registered callback.
    """
    callbacks.remove(fn)


def register(fn):
    """Deprecated; see registerCallback

    Register a callable to be invoked when there is an unhandled exception.
    The callback will be passed the output of sys.exc_info(): (exception type, exception, traceback)
    Multiple callbacks will be invoked in the order they were registered.
    """
    old_callbacks.append(fn)


def unregister(fn):
    """Deprecated; see unregisterCallback

    Unregister a previously registered callback.
    """
    old_callbacks.remove(fn)


def setTracebackClearing(clear=True):
    """
    Enable or disable traceback clearing.
    By default, clearing is disabled and Python will indefinitely store unhandled exception stack traces.
    This function is provided since Python's default behavior can cause unexpected retention of 
    large memory-consuming objects.
    """
    global clear_tracebacks
    clear_tracebacks = clear


class ExceptionHandler(object):
    def __init__(self):
        self.orig_sys_excepthook = sys.excepthook
        self.orig_threading_excepthook = threading.excepthook
        sys.excepthook = self.sys_excepthook
        threading.excepthook = self.threading_excepthook

    def remove(self):
        """Restore original exception hooks, deactivating this exception handler
        """
        sys.excepthook = self.orig_sys_excepthook
        threading.excepthook = self.orig_threading_excepthook

    def sys_excepthook(self, *args):
        # sys.excepthook signature is (exc_type, exc_value, exc_traceback)
        args = SimpleNamespace(exc_type=args[0], exc_value=args[1], exc_traceback=args[2], thread=None)
        return self._excepthook(args, use_thread_hook=False)

    def threading_excepthook(self, args):
        # threading.excepthook signature is (namedtuple(exc_type, exc_value, exc_traceback, thread))
        return self._excepthook(args, use_thread_hook=True)

    def _excepthook(self, args, use_thread_hook):
        ## Start by extending recursion depth just a bit. 
        ## If the error we are catching is due to recursion, we don't want to generate another one here.
        recursionLimit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(recursionLimit+100)

            ## call original exception handler first (prints exception)
            global callbacks, clear_tracebacks
            header = "===== %s =====" % str(time.strftime("%Y.%m.%d %H:%m:%S", time.localtime(time.time())))
            try:
                print(header)
            except Exception:
                sys.stderr.write("Warning: stdout is broken! Falling back to stderr.\n")
                sys.stdout = sys.stderr

            if use_thread_hook:
                ret = self.orig_threading_excepthook(args)
            else:
                ret = self.orig_sys_excepthook(args.exc_type, args.exc_value, args.exc_traceback)

            for cb in callbacks:
                try:
                    cb(args)
                except Exception:
                    print("   --------------------------------------------------------------")
                    print("      Error occurred during exception callback %s" % str(cb))
                    print("   --------------------------------------------------------------")
                    traceback.print_exception(*sys.exc_info())

            # deprecated callback style requiring 3 args
            for cb in old_callbacks:
                try:
                    cb(args.exc_type, args.exc_value, args.exc_traceback)
                except Exception:
                    print("   --------------------------------------------------------------")
                    print("      Error occurred during exception callback %s" % str(cb))
                    print("   --------------------------------------------------------------")
                    traceback.print_exception(*sys.exc_info())

            ## Clear long-term storage of last traceback to prevent memory-hogging.
            ## (If an exception occurs while a lot of data is present on the stack,
            ## such as when loading large files, the data would ordinarily be kept
            ## until the next exception occurs. We would rather release this memory
            ## as soon as possible.)
            if clear_tracebacks is True:
                sys.last_traceback = None

            return ret
        
        finally:
            sys.setrecursionlimit(recursionLimit)            

    def implements(self, interface=None):
        ## this just makes it easy for us to detect whether an ExceptionHook is already installed.
        if interface is None:
            return ['ExceptionHandler']
        else:
            return interface == 'ExceptionHandler'


## replace built-in excepthook only if this has not already been done
if not (hasattr(sys.excepthook, 'implements') and sys.excepthook.implements('ExceptionHandler')):
    handler = ExceptionHandler()
    original_excepthook = handler.orig_sys_excepthook
    original_threading_excepthook = handler.orig_threading_excepthook
