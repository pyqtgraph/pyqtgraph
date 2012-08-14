"""For starting up remote processes"""
import sys, pickle

if __name__ == '__main__':
    name, port, authkey, targetStr, path = pickle.load(sys.stdin)
    if path is not None:
        ## rewrite sys.path without assigning a new object--no idea who already has a reference to the existing list.
        while len(sys.path) > 0:
            sys.path.pop()
        sys.path.extend(path)
    #import pyqtgraph
    #import pyqtgraph.multiprocess.processes
    target = pickle.loads(targetStr)  ## unpickling the target should import everything we need
    target(name, port, authkey)
    sys.exit(0)
