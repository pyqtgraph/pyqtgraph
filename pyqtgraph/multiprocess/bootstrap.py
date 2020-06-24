"""For starting up remote processes"""
import sys, pickle, os

if __name__ == '__main__':
    if hasattr(os, 'setpgrp'):
        os.setpgrp()  ## prevents signals (notably keyboard interrupt) being forwarded from parent to this process
    if sys.version[0] == '3':
        #name, port, authkey, ppid, targetStr, path, pyside = pickle.load(sys.stdin.buffer)
        opts = pickle.load(sys.stdin.buffer)
    else:
        #name, port, authkey, ppid, targetStr, path, pyside = pickle.load(sys.stdin)
        opts = pickle.load(sys.stdin)
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

    pyqtapis = opts.pop('pyqtapis', None)
    if pyqtapis is not None:
        import sip
        for k,v in pyqtapis.items():
            sip.setapi(k, v)
        
    qt_lib = opts.pop('qt_lib', None)
    if qt_lib == 'PySide':
        import PySide
    elif qt_lib == 'PySide2':
        import PySide2
    elif qt_lib == 'PyQt5':
        import PyQt5        
    
    targetStr = opts.pop('targetStr')
    try:
        target = pickle.loads(targetStr)  ## unpickling the target should import everything we need
    except:
        print("Current sys.path:", sys.path)
        raise
    target(**opts)  ## Send all other options to the target function
    sys.exit(0)
