from remoteproxy import RemoteEventHandler, ExitError, NoResultError, LocalObjectProxy, ObjectProxy
import subprocess, atexit, os, sys, time, random, socket
import cPickle as pickle
import multiprocessing.connection

class Process(RemoteEventHandler):
    def __init__(self, name=None, target=None):
        if target is None:
            target = startEventLoop
        if name is None:
            name = str(self)
        
        ## random authentication key
        authkey = ''.join([chr(random.getrandbits(7)) for i in range(20)])
        
        ## Listen for connection from remote process (and find free port number)
        port = 10000
        while True:
            try:
                l = multiprocessing.connection.Listener(('localhost', int(port)), authkey=authkey)
                break
            except socket.error as ex:
                if ex.errno != 98:
                    raise
                port += 1
        
        ## start remote process, instruct it to run target function
        self.proc = subprocess.Popen((sys.executable, __file__, 'remote'), stdin=subprocess.PIPE)
        pickle.dump((name+'_child', port, authkey, target), self.proc.stdin)
        self.proc.stdin.close()
        
        ## open connection for remote process
        conn = l.accept()
        RemoteEventHandler.__init__(self, conn, name+'_parent', pid=self.proc.pid)
        
        atexit.register(self.join)
        
    def join(self, timeout=10):
        if self.proc.poll() is None:
            self.close()
            start = time.time()
            while self.proc.poll() is None:
                if timeout is not None and time.time() - start > timeout:
                    raise Exception('Timed out waiting for remote process to end.')
                time.sleep(0.05)
        
        
def startEventLoop(name, port, authkey):
    conn = multiprocessing.connection.Client(('localhost', int(port)), authkey=authkey)
    global HANDLER
    HANDLER = RemoteEventHandler(conn, name, os.getppid())
    while True:
        try:
            HANDLER.processRequests()  # exception raised when the loop should exit
            time.sleep(0.01)
        except ExitError:
            break


class ForkedProcess(RemoteEventHandler):
    """
    ForkedProcess is a substitute for Process that uses os.fork() to generate a new process.
    This is much faster than starting a completely new interpreter, but carries some caveats
    and limitations:
      - open file handles are shared with the parent process, which is potentially dangerous
      - it is not possible to have a QApplication in both parent and child process
        (unless both QApplications are created _after_ the call to fork())
      - generally not thread-safe. Also, threads are not copied by fork(); the new process 
        will have only one thread that starts wherever fork() was called in the parent process.
      - forked processes are unceremoniously terminated when join() is called; they are not 
        given any opportunity to clean up. (This prevents them calling any cleanup code that
        was only intended to be used by the parent process)
    """
    
    def __init__(self, name=None, target=0, preProxy=None):
        """
        When initializing, an optional target may be given. 
        If no target is specified, self.eventLoop will be used.
        If None is given, no target will be called (and it will be up 
        to the caller to properly shut down the forked process)
        
        preProxy may be a dict of values that will appear as ObjectProxy
        in the remote process (but do not need to be sent explicitly since 
        they are available immediately before the call to fork().
        Proxies will be availabe as self.proxies[name].
        """
        self.hasJoined = False
        if target == 0:
            target = self.eventLoop
        if name is None:
            name = str(self)
        
        conn, remoteConn = multiprocessing.Pipe()
        
        proxyIDs = {}
        if preProxy is not None:
            for k, v in preProxy.iteritems():
                proxyId = LocalObjectProxy.registerObject(v)
                proxyIDs[k] = proxyId
        
        pid = os.fork()
        if pid == 0:
            self.isParent = False
            conn.close()
            sys.stdin.close()  ## otherwise we screw with interactive prompts.
            RemoteEventHandler.__init__(self, remoteConn, name+'_child', pid=os.getppid())
            if target is not None:
                target()
                
            ppid = os.getppid()
            self.forkedProxies = {}
            for name, proxyId in proxyIDs.iteritems():
                self.forkedProxies[name] = ObjectProxy(ppid, proxyId=proxyId, typeStr=repr(preProxy[name]))
        else:
            self.isParent = True
            self.childPid = pid
            remoteConn.close()
            RemoteEventHandler.handlers = {}  ## don't want to inherit any of this from the parent.
            
            RemoteEventHandler.__init__(self, conn, name+'_parent', pid=pid)
            atexit.register(self.join)
        
        
    def eventLoop(self):
        while True:
            try:
                self.processRequests()  # exception raised when the loop should exit
                time.sleep(0.01)
            except ExitError:
                sys.exit(0)
            except:
                print "Error occurred in forked event loop:"
                sys.excepthook(*sys.exc_info())
        
    def join(self, timeout=10):
        if self.hasJoined:
            return
        #os.kill(pid, 9)  
        try:
            self.close(callSync='sync', timeout=timeout, noCleanup=True)  ## ask the child process to exit and require that it return a confirmation.
        except IOError:  ## probably remote process has already quit
            pass  
        self.hasJoined = True


##Special set of subclasses that implement a Qt event loop instead.
        
class RemoteQtEventHandler(RemoteEventHandler):
    def __init__(self, *args, **kwds):
        RemoteEventHandler.__init__(self, *args, **kwds)
        
    def startEventTimer(self):
        from pyqtgraph.Qt import QtGui, QtCore
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.processRequests)
        self.timer.start(10)
    
    def processRequests(self):
        try:
            RemoteEventHandler.processRequests(self)
        except ExitError:
            from pyqtgraph.Qt import QtGui, QtCore
            QtGui.QApplication.instance().quit()
            self.timer.stop()
            #raise

class QtProcess(Process):
    def __init__(self, name=None):
        Process.__init__(self, name, target=startQtEventLoop)
        self.startEventTimer()
        
    def startEventTimer(self):
        from pyqtgraph.Qt import QtGui, QtCore  ## avoid module-level import to keep bootstrap snappy.
        self.timer = QtCore.QTimer()
        app = QtGui.QApplication.instance()
        if app is None:
            raise Exception("Must create QApplication before starting QtProcess")
        self.timer.timeout.connect(self.processRequests)
        self.timer.start(10)
        
    def processRequests(self):
        try:
            Process.processRequests(self)
        except ExitError:
            self.timer.stop()
    
def startQtEventLoop(name, port, authkey):
    conn = multiprocessing.connection.Client(('localhost', int(port)), authkey=authkey)
    from pyqtgraph.Qt import QtGui, QtCore
    #from PyQt4 import QtGui, QtCore
    app = QtGui.QApplication.instance()
    #print app
    if app is None:
        app = QtGui.QApplication([])
        app.setQuitOnLastWindowClosed(False)  ## generally we want the event loop to stay open 
                                              ## until it is explicitly closed by the parent process.
    
    global HANDLER
    HANDLER = RemoteQtEventHandler(conn, name, os.getppid())
    HANDLER.startEventTimer()
    app.exec_()


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'remote':  ## module has been invoked as script in new python interpreter.
        name, port, authkey, target = pickle.load(sys.stdin)
        target(name, port, authkey)
        sys.exit(0)
