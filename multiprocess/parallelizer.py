import os, sys, time, multiprocessing
from processes import ForkedProcess
from remoteproxy import ExitError

class Parallelize:
    """
    Class for ultra-simple inline parallelization on multi-core CPUs
    
    Example::
    
        ## Here is the serial (single-process) task:
        
        tasks = [1, 2, 4, 8]
        results = []
        for task in tasks:
            result = processTask(task)
            results.append(result)
        print results
        
        
        ## Here is the parallelized version:
        
        tasks = [1, 2, 4, 8]
        results = []
        with Parallelize(tasks, workers=4, results=results) as tasker:
            for task in tasker:
                result = processTask(task)
                tasker.results.append(result)
        print results
        
        
    The only major caveat is that *result* in the example above must be picklable.
    """

    def __init__(self, tasks, workers=None, block=True, **kwds):
        """
        Args:
        tasks   - list of objects to be processed (Parallelize will determine how to distribute the tasks)
        workers - number of worker processes or None to use number of CPUs in the system
        kwds    - objects to be shared by proxy with child processes
        """
        
        self.block = block
        if workers is None:
            workers = multiprocessing.cpu_count()
        if not hasattr(os, 'fork'):
            workers = 1
        self.workers = workers
        self.tasks = list(tasks)
        self.kwds = kwds
        
    def __enter__(self):
        self.proc = None
        workers = self.workers
        if workers == 1: 
            return Tasker(None, self.tasks, self.kwds)
            
        self.childs = []
        
        ## break up tasks into one set per worker
        chunks = [[] for i in xrange(workers)]
        i = 0
        for i in range(len(self.tasks)):
            chunks[i%workers].append(self.tasks[i])
        
        ## fork and assign tasks to each worker
        for i in range(workers):
            proc = ForkedProcess(target=None, preProxy=self.kwds)
            if not proc.isParent:
                self.proc = proc
                return Tasker(proc, chunks[i], proc.forkedProxies)
            else:
                self.childs.append(proc)
        
        ## process events from workers until all have exited.
        activeChilds = self.childs[:]
        while len(activeChilds) > 0:
            for ch in activeChilds:
                rem = []
                try:
                    ch.processRequests()
                except ExitError:
                    rem.append(ch)
            for ch in rem:
                activeChilds.remove(ch)
            time.sleep(0.1)
        
        return []  ## no tasks for parent process.
        
    def __exit__(self, *exc_info):
        if exc_info[0] is not None:
            sys.excepthook(*exc_info)
        if self.proc is not None:
            os._exit(0)
    
    def wait(self):
        ## wait for all child processes to finish
        pass
    
class Tasker:
    def __init__(self, proc, tasks, kwds):
        self.proc = proc
        self.tasks = tasks
        for k, v in kwds.iteritems():
            setattr(self, k, v)
        
    def __iter__(self):
        ## we could fix this up such that tasks are retrieved from the parent process one at a time..
        for task in self.tasks:
            yield task
        if self.proc is not None:
            self.proc.close()
    
    
    
#class Parallelizer:
    #"""
    #Use::
    
        #p = Parallelizer()
        #with p(4) as i:
            #p.finish(do_work(i))
        #print p.results()
    
    #"""
    #def __init__(self):
        #pass

    #def __call__(self, n):
        #self.replies = []
        #self.conn = None  ## indicates this is the parent process
        #return Session(self, n)
            
    #def finish(self, data):
        #if self.conn is None:
            #self.replies.append((self.i, data))
        #else:
            ##print "send", self.i, data
            #self.conn.send((self.i, data))
            #os._exit(0)
            
    #def result(self):
        #print self.replies
        
#class Session:
    #def __init__(self, par, n):
        #self.par = par
        #self.n = n
        
    #def __enter__(self):
        #self.childs = []
        #for i in range(1, self.n):
            #c1, c2 = multiprocessing.Pipe()
            #pid = os.fork()
            #if pid == 0:  ## child
                #self.par.i = i
                #self.par.conn = c2
                #self.childs = None
                #c1.close()
                #return i
            #else:
                #self.childs.append(c1)
                #c2.close()
        #self.par.i = 0
        #return 0
            
        
        
    #def __exit__(self, *exc_info):
        #if exc_info[0] is not None:
            #sys.excepthook(*exc_info)
        #if self.childs is not None:
            #self.par.replies.extend([conn.recv() for conn in self.childs])
        #else:
            #self.par.finish(None)
        
