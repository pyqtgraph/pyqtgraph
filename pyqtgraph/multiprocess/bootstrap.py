"""For starting up remote processes"""
import importlib
import os
import pickle
import sys

if __name__ == '__main__':
    if hasattr(os, 'setpgrp'):
        os.setpgrp()  ## prevents signals (notably keyboard interrupt) being forwarded from parent to this process
    #name, port, authkey, ppid, targetStr, path, pyside = pickle.load(sys.stdin.buffer)
    opts = pickle.load(sys.stdin.buffer)
    #print "key:",  ' '.join([str(ord(x)) for x in authkey])
    path = opts.pop('path', None)
    if path is not None:
        if isinstance(path, str):
            # if string, just insert this into the path
            sys.path.insert(0, path)
        else:
            # if list, then replace the entire sys.path
            ## modify sys.path in place--no idea who already has a reference to the existing list.
            while len(sys.path) > 0:
                sys.path.pop()
            sys.path.extend(path)

    qt_lib = opts.pop('qt_lib', None)
    if qt_lib is not None:
        globals()[qt_lib] = importlib.import_module(qt_lib)
    
    targetStr = opts.pop('targetStr')
    try:
        target = pickle.loads(targetStr)  ## unpickling the target should import everything we need
    except:
        print("Current sys.path:", sys.path)
        raise
    target(**opts)  ## Send all other options to the target function
    sys.exit(0)
