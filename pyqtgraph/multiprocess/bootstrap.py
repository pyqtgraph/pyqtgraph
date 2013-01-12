"""For starting up remote processes"""
import sys, pickle, os

if __name__ == '__main__':
    if hasattr(os, 'setpgrp'):
        os.setpgrp()  ## prevents signals (notably keyboard interrupt) being forwarded from parent to this process
    name, port, authkey, ppid, targetStr, path = pickle.load(sys.stdin)
    #print "key:",  ' '.join([str(ord(x)) for x in authkey])
    if path is not None:
        ## rewrite sys.path without assigning a new object--no idea who already has a reference to the existing list.
        while len(sys.path) > 0:
            sys.path.pop()
        sys.path.extend(path)
    #import pyqtgraph
    #import pyqtgraph.multiprocess.processes
    target = pickle.loads(targetStr)  ## unpickling the target should import everything we need
    target(name, port, authkey, ppid)
    sys.exit(0)
