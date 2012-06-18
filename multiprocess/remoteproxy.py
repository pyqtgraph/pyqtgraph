import os, __builtin__, time, sys, traceback, weakref
import cPickle as pickle

class ExitError(Exception):
    pass

class NoResultError(Exception):
    pass

    
class RemoteEventHandler(object):
    
    handlers = {}   ## maps {process ID : handler}. This allows unpickler to determine which process
                    ## an object proxy belongs to
                         
    def __init__(self, connection, name, pid):
        self.conn = connection
        self.name = name
        self.results = {} ## reqId: (status, result); cache of request results received from the remote process
                          ## status is either 'result' or 'error'
                          ##   if 'error', then result will be (exception, formatted exceprion)
                          ##   where exception may be None if it could not be passed through the Connection.
                          
        self.proxies = {} ## maps {weakref(proxy): proxyId}; used to inform the remote process when a proxy has been deleted.
        
        ## attributes that affect the behavior of the proxy. 
        ## See ObjectProxy._setProxyOptions for description
        self.proxyOptions = {
            'callSync': 'sync',      ## 'sync', 'async', 'off'
            'timeout': 10,           ## float
            'returnType': 'auto',    ## 'proxy', 'value', 'auto'
            'autoProxy': False,      ## bool
            'deferGetattr': False,   ## True, False
            'noProxyTypes': [ type(None), str, int, float, tuple, list, dict, LocalObjectProxy, ObjectProxy ],
        }
        
        self.nextRequestId = 0
        self.exited = False
        
        RemoteEventHandler.handlers[pid] = self  ## register this handler as the one communicating with pid
    
    @classmethod
    def getHandler(cls, pid):
        return cls.handlers[pid]
    
    def getProxyOption(self, opt):
        return self.proxyOptions[opt]
        
    def setProxyOptions(self, **kwds):
        """
        Set the default behavior options for object proxies.
        See ObjectProxy._setProxyOptions for more info.
        """
        self.proxyOptions.update(kwds)
    
    def processRequests(self):
        """Process all pending requests from the pipe, return
        after no more events are immediately available. (non-blocking)"""
        if self.exited:
            raise ExitError()
        
        while self.conn.poll():
            try:
                self.handleRequest()
            except ExitError:
                self.exited = True
                raise
            except:
                print "Error in process %s" % self.name
                sys.excepthook(*sys.exc_info())
    
    def handleRequest(self):
        """Handle a single request from the remote process. 
        Blocks until a request is available."""
        result = None
        try:
            cmd, reqId, optStr = self.conn.recv() ## args, kwds are double-pickled to ensure this recv() call never fails
        except EOFError:
            ## remote process has shut down; end event loop
            raise ExitError()
        except IOError:
            raise ExitError()
        #print os.getpid(), "received request:", cmd, reqId
            
        
        try:
            if cmd == 'result' or cmd == 'error':
                resultId = reqId
                reqId = None  ## prevents attempt to return information from this request
                              ## (this is already a return from a previous request)
            
            opts = pickle.loads(optStr)
            #print os.getpid(), "received request:", cmd, reqId, opts
            returnType = opts.get('returnType', 'auto')
            
            if cmd == 'result':
                self.results[resultId] = ('result', opts['result'])
            elif cmd == 'error':
                self.results[resultId] = ('error', (opts['exception'], opts['excString']))
            elif cmd == 'getObjAttr':
                result = getattr(opts['obj'], opts['attr'])
            elif cmd == 'callObj':
                obj = opts['obj']
                fnargs = opts['args']
                fnkwds = opts['kwds']
                if len(fnkwds) == 0:  ## need to do this because some functions do not allow keyword arguments.
                    #print obj, fnargs
                    result = obj(*fnargs)
                else:
                    result = obj(*fnargs, **fnkwds)
            elif cmd == 'getObjValue':
                result = opts['obj']  ## has already been unpickled into its local value
                returnType = 'value'
            elif cmd == 'transfer':
                result = opts['obj']
                returnType = 'proxy'
            elif cmd == 'import':
                name = opts['module']
                fromlist = opts.get('fromlist', [])
                mod = __builtin__.__import__(name, fromlist=fromlist)
                
                if len(fromlist) == 0:
                    parts = name.lstrip('.').split('.')
                    result = mod
                    for part in parts[1:]:
                        result = getattr(result, part)
                else:
                    result = map(mod.__getattr__, fromlist)
                
            elif cmd == 'del':
                LocalObjectProxy.releaseProxyId(opts['proxyId'])
                #del self.proxiedObjects[opts['objId']]
                
            elif cmd == 'close':
                if reqId is not None:
                    result = True
                    returnType = 'value'
                    
            exc = None
        except:
            exc = sys.exc_info()

            
            
        if reqId is not None:
            if exc is None:
                #print "returnValue:", returnValue, result
                if returnType == 'auto':
                    result = self.autoProxy(result, self.proxyOptions['noProxyTypes'])
                elif returnType == 'proxy':
                    result = LocalObjectProxy(result)
                
                try:
                    self.replyResult(reqId, result)
                except:
                    sys.excepthook(*sys.exc_info())
                    self.replyError(reqId, *sys.exc_info())
            else:
                self.replyError(reqId, *exc)
                    
        elif exc is not None:
            sys.excepthook(*exc)
    
        if cmd == 'close':
            if opts.get('noCleanup', False) is True:
                os._exit(0)  ## exit immediately, do not pass GO, do not collect $200.
                             ## (more importantly, do not call any code that would
                             ## normally be invoked at exit)
            else:
                raise ExitError()
        
    
    
    def replyResult(self, reqId, result):
        self.send(request='result', reqId=reqId, callSync='off', opts=dict(result=result))
    
    def replyError(self, reqId, *exc):
        excStr = traceback.format_exception(*exc)
        try:
            self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=exc[1], excString=excStr))
        except:
            self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=None, excString=excStr))
    
    def send(self, request, opts=None, reqId=None, callSync='sync', timeout=10, returnType=None, **kwds):
        """Send a request or return packet to the remote process.
        Generally it is not necessary to call this method directly; it is for internal use.
        (The docstring has information that is nevertheless useful to the programmer
        as it describes the internal protocol used to communicate between processes)
        
        ==========  ====================================================================
        Arguments:  
        request     String describing the type of request being sent (see below)
        reqId       Integer uniquely linking a result back to the request that generated
                    it. (most requests leave this blank)
        callSync    'sync':  return the actual result of the request
                    'async': return a Request object which can be used to look up the 
                             result later
                    'off':   return no result
        timeout     Time in seconds to wait for a response when callSync=='sync'
        opts        Extra arguments sent to the remote process that determine the way
                    the request will be handled (see below)
        returnType  'proxy', 'value', or 'auto'
        ==========  ====================================================================
        
        Description of request strings and options allowed for each:
        
        =============  =============  ========================================================
        request        option         description
        -------------  -------------  --------------------------------------------------------
        getObjAttr                    Request the remote process return (proxy to) an
                                      attribute of an object.
                       obj            reference to object whose attribute should be 
                                      returned
                       attr           string name of attribute to return
                       returnValue    bool or 'auto' indicating whether to return a proxy or
                                      the actual value. 
                       
        callObj                       Request the remote process call a function or 
                                      method. If a request ID is given, then the call's
                                      return value will be sent back (or information
                                      about the error that occurred while running the
                                      function)
                       obj            the (reference to) object to call
                       args           tuple of arguments to pass to callable
                       kwds           dict of keyword arguments to pass to callable
                       returnValue    bool or 'auto' indicating whether to return a proxy or
                                      the actual value. 
                       
        getObjValue                   Request the remote process return the value of
                                      a proxied object (must be picklable)
                       obj            reference to object whose value should be returned
                       
        transfer                      Copy an object to the remote process and request
                                      it return a proxy for the new object.
                       obj            The object to transfer.
                       
        import                        Request the remote process import new symbols
                                      and return proxy(ies) to the imported objects
                       module         the string name of the module to import
                       fromlist       optional list of string names to import from module
                       
        del                           Inform the remote process that a proxy has been 
                                      released (thus the remote process may be able to 
                                      release the original object)
                       proxyId        id of proxy which is no longer referenced by 
                                      remote host
                                      
        close                         Instruct the remote process to stop its event loop
                                      and exit. Optionally, this request may return a 
                                      confirmation.
            
        result                        Inform the remote process that its request has 
                                      been processed                        
                       result         return value of a request
                       
        error                         Inform the remote process that its request failed
                       exception      the Exception that was raised (or None if the 
                                      exception could not be pickled)
                       excString      string-formatted version of the exception and 
                                      traceback
        =============  =====================================================================
        """
        #if len(kwds) > 0:
            #print "Warning: send() ignored args:", kwds
            
        if opts is None:
            opts = {}
        
        assert callSync in ['off', 'sync', 'async'], 'callSync must be one of "off", "sync", or "async"'
        if reqId is None:
            if callSync != 'off': ## requested return value; use the next available request ID
                reqId = self.nextRequestId
                self.nextRequestId += 1
        else:
            ## If requestId is provided, this _must_ be a response to a previously received request.
            assert request in ['result', 'error']
        
        if returnType is not None:
            opts['returnType'] = returnType
        #print "send", opts
        ## double-pickle args to ensure that at least status and request ID get through
        try:
            optStr = pickle.dumps(opts)
        except:
            print "Error pickling:", opts
            raise
        
        request = (request, reqId, optStr)
        self.conn.send(request)
        
        if callSync == 'off':
            return
        
        req = Request(self, reqId, description=str(request), timeout=timeout)
        if callSync == 'async':
            return req
            
        if callSync == 'sync':
            try:
                return req.result()
            except NoResultError:
                return req
        
    def close(self, callSync='off', noCleanup=False, **kwds):
        self.send(request='close', opts=dict(noCleanup=noCleanup), callSync=callSync, **kwds)
    
    def getResult(self, reqId):
        ## raises NoResultError if the result is not available yet
        #print self.results.keys(), os.getpid()
        if reqId not in self.results:
            #self.readPipe()
            try:
                self.processRequests()
            except ExitError:
                pass
        if reqId not in self.results:
            raise NoResultError()
        status, result = self.results.pop(reqId)
        if status == 'result': 
            return result
        elif status == 'error':
            #print ''.join(result)
            exc, excStr = result
            if exc is not None:
                print "===== Remote process raised exception on request: ====="
                print ''.join(excStr)
                print "===== Local Traceback to request follows: ====="
                raise exc
            else:
                print ''.join(excStr)
                raise Exception("Error getting result. See above for exception from remote process.")
                
        else:
            raise Exception("Internal error.")
    
    def _import(self, mod, **kwds):
        """
        Request the remote process import a module (or symbols from a module)
        and return the proxied results. Uses built-in __import__() function, but 
        adds a bit more processing:
        
            _import('module')  =>  returns module
            _import('module.submodule')  =>  returns submodule 
                                             (note this differs from behavior of __import__)
            _import('module', fromlist=[name1, name2, ...])  =>  returns [module.name1, module.name2, ...]
                                             (this also differs from behavior of __import__)
            
        """
        return self.send(request='import', callSync='sync', opts=dict(module=mod), **kwds)
        
    def getObjAttr(self, obj, attr, **kwds):
        return self.send(request='getObjAttr', opts=dict(obj=obj, attr=attr), **kwds)
        
    def getObjValue(self, obj, **kwds):
        return self.send(request='getObjValue', opts=dict(obj=obj), **kwds)
        
    def callObj(self, obj, args, kwds, **opts):
        opts = opts.copy()
        noProxyTypes = opts.pop('noProxyTypes', None)
        if noProxyTypes is None:
            noProxyTypes = self.proxyOptions['noProxyTypes']
        autoProxy = opts.pop('autoProxy', self.proxyOptions['autoProxy'])
            
        if autoProxy is True:
            args = tuple([self.autoProxy(v, noProxyTypes) for v in args])
            for k, v in kwds.iteritems():
                opts[k] = self.autoProxy(v, noProxyTypes)
        
        return self.send(request='callObj', opts=dict(obj=obj, args=args, kwds=kwds), **opts)

    def registerProxy(self, proxy):
        ref = weakref.ref(proxy, self.deleteProxy)
        self.proxies[ref] = proxy._proxyId
    
    def deleteProxy(self, ref):
        proxyId = self.proxies.pop(ref)
        try:
            self.send(request='del', opts=dict(proxyId=proxyId), callSync='off')
        except IOError:  ## if remote process has closed down, there is no need to send delete requests anymore
            pass

    def transfer(self, obj, **kwds):
        """
        Transfer an object to the remote host (the object must be picklable) and return 
        a proxy for the new remote object.
        """
        return self.send(request='transfer', opts=dict(obj=obj), **kwds)
        
    def autoProxy(self, obj, noProxyTypes):
        ## Return object wrapped in LocalObjectProxy _unless_ its type is in noProxyTypes.
        for typ in noProxyTypes:
            if isinstance(obj, typ):
                return obj
        return LocalObjectProxy(obj)
        
        
class Request:
    ## used internally for tracking asynchronous requests and returning results
    def __init__(self, process, reqId, description=None, timeout=10):
        self.proc = process
        self.description = description
        self.reqId = reqId
        self.gotResult = False
        self._result = None
        self.timeout = timeout
        
    def result(self, block=True, timeout=None):
        """Return the result for this request. 
        If block is True, wait until the result has arrived or *timeout* seconds passes.
        If the timeout is reached, raise an exception. (use timeout=None to disable)
        If block is False, raises an exception if the result has not arrived yet."""
        
        if self.gotResult:
            return self._result
            
        if timeout is None:
           timeout = self.timeout 
        
        if block:
            start = time.time()
            while not self.hasResult():
                time.sleep(0.005)
                if timeout >= 0 and time.time() - start > timeout:
                    print "Request timed out:", self.description
                    import traceback
                    traceback.print_stack()
                    raise NoResultError()
            return self._result
        else:
            self._result = self.proc.getResult(self.reqId)  ## raises NoResultError if result is not available yet
            self.gotResult = True
            return self._result
        
    def hasResult(self):
        """Returns True if the result for this request has arrived."""
        try:
            #print "check result", self.description
            self.result(block=False)
        except NoResultError:
            #print "  -> not yet"
            pass
        
        return self.gotResult

class LocalObjectProxy(object):
    """Used for wrapping local objects to ensure that they are send by proxy to a remote host."""
    nextProxyId = 0
    proxiedObjects = {}  ## maps {proxyId: object}
    
    
    @classmethod
    def registerObject(cls, obj):
        ## assign it a unique ID so we can keep a reference to the local object
        
        pid = cls.nextProxyId
        cls.nextProxyId += 1
        cls.proxiedObjects[pid] = obj
        #print "register:", cls.proxiedObjects
        return pid
    
    @classmethod
    def lookupProxyId(cls, pid):
        return cls.proxiedObjects[pid]
    
    @classmethod
    def releaseProxyId(cls, pid):
        del cls.proxiedObjects[pid]
        #print "release:", cls.proxiedObjects 
    
    def __init__(self, obj):
        self.processId = os.getpid()
        #self.objectId = id(obj)
        self.typeStr = repr(obj)
        #self.handler = handler
        self.obj = obj
        
    def __reduce__(self):
        ## a proxy is being pickled and sent to a remote process.
        ## every time this happens, a new proxy will be generated in the remote process,
        ## so we keep a new ID so we can track when each is released.
        pid = LocalObjectProxy.registerObject(self.obj)
        return (unpickleObjectProxy, (self.processId, pid, self.typeStr))
        
## alias
proxy = LocalObjectProxy

def unpickleObjectProxy(processId, proxyId, typeStr, attributes=None):
    if processId == os.getpid():
        obj = LocalObjectProxy.lookupProxyId(proxyId)
        if attributes is not None:
            for attr in attributes:
                obj = getattr(obj, attr)
        return obj
    else:
        return ObjectProxy(processId, proxyId=proxyId, typeStr=typeStr)
    
class ObjectProxy(object):
    """
    Proxy to an object stored by the remote process. Proxies are created
    by calling Process._import(), Process.transfer(), or by requesting/calling
    attributes on existing proxy objects.
    
    For the most part, this object can be used exactly as if it
    were a local object.
    """
    def __init__(self, processId, proxyId, typeStr='', parent=None):
        object.__init__(self)
        ## can't set attributes directly because setattr is overridden.
        self.__dict__['_processId'] = processId
        self.__dict__['_typeStr'] = typeStr
        self.__dict__['_proxyId'] = proxyId
        self.__dict__['_attributes'] = ()
        ## attributes that affect the behavior of the proxy. 
        ## in all cases, a value of None causes the proxy to ask
        ## its parent event handler to make the decision
        self.__dict__['_proxyOptions'] = {
            'callSync': None,      ## 'sync', 'async', None 
            'timeout': None,       ## float, None
            'returnType': None,    ## 'proxy', 'value', 'auto', None
            'deferGetattr': None,  ## True, False, None
            'noProxyTypes': None,  ## list of types to send by value instead of by proxy
        }
        
        self.__dict__['_handler'] = RemoteEventHandler.getHandler(processId)
        self.__dict__['_handler'].registerProxy(self)  ## handler will watch proxy; inform remote process when the proxy is deleted.
    
    def _setProxyOptions(self, **kwds):
        """
        Change the behavior of this proxy. For all options, a value of None
        will cause the proxy to instead use the default behavior defined
        by its parent Process.
        
        Options are:
        
        =============  =============================================================
        callSync       'sync', 'async', 'off', or None. 
                       If 'async', then calling methods will return a Request object
                       which can be used to inquire later about the result of the 
                       method call.
                       If 'sync', then calling a method
                       will block until the remote process has returned its result
                       or the timeout has elapsed (in this case, a Request object
                       is returned instead).
                       If 'off', then the remote process is instructed _not_ to 
                       reply and the method call will return None immediately.
        returnType     'auto', 'proxy', 'value', or None. 
                       If 'proxy', then the value returned when calling a method
                       will be a proxy to the object on the remote process.
                       If 'value', then attempt to pickle the returned object and
                       send it back.
                       If 'auto', then the decision is made by consulting the
                       'noProxyTypes' option.
        autoProxy      bool or None. If True, arguments to __call__ are 
                       automatically converted to proxy unless their type is 
                       listed in noProxyTypes (see below). If False, arguments
                       are left untouched. Use proxy(obj) to manually convert
                       arguments before sending. 
        timeout        float or None. Length of time to wait during synchronous 
                       requests before returning a Request object instead.
        deferGetattr   True, False, or None. 
                       If False, all attribute requests will be sent to the remote 
                       process immediately and will block until a response is
                       received (or timeout has elapsed).
                       If True, requesting an attribute from the proxy returns a
                       new proxy immediately. The remote process is _not_ contacted
                       to make this request. This is faster, but it is possible to 
                       request an attribute that does not exist on the proxied
                       object. In this case, AttributeError will not be raised
                       until an attempt is made to look up the attribute on the
                       remote process.
        noProxyTypes   List of object types that should _not_ be proxied when
                       sent to the remote process.
        =============  =============================================================
        """
        self._proxyOptions.update(kwds)
    
    def _getProxyOption(self, opt):
        val = self._proxyOptions[opt]
        if val is None:
            return self._handler.getProxyOption(opt)
        return val
    
    def _getProxyOptions(self):
        return {k: self._getProxyOption(k) for k in self._proxyOptions}
    
    def __reduce__(self):
        return (unpickleObjectProxy, (self._processId, self._proxyId, self._typeStr, self._attributes))
    
    def __repr__(self):
        #objRepr = self.__getattr__('__repr__')(callSync='value')
        return "<ObjectProxy for process %d, object 0x%x: %s >" % (self._processId, self._proxyId, self._typeStr)
        
        
    def __getattr__(self, attr):
        #if '_processId' not in self.__dict__:
            #raise Exception("ObjectProxy has no processId")
        #proc = Process._processes[self._processId]
        deferred = self._getProxyOption('deferGetattr')
        if deferred is True:
            return self._deferredAttr(attr)
        else:
            opts = self._getProxyOptions()
            return self._handler.getObjAttr(self, attr, **opts)
        
    def _deferredAttr(self, attr):
        return DeferredObjectProxy(self, attr)
        
    def __call__(self, *args, **kwds):
        """
        Attempts to call the proxied object from the remote process.
        Accepts extra keyword arguments:
        
            _callSync    'off', 'sync', or 'async'
            _returnType   'value', 'proxy', or 'auto'
        
        """
        #opts = {}
        #callSync = kwds.pop('_callSync', self.)
        #if callSync is not None:
            #opts['callSync'] = callSync
        #returnType = kwds.pop('_returnType', self._defaultReturnValue)
        #if returnType is not None:
            #opts['returnType'] = returnType
        opts = self._getProxyOptions()
        for k in opts:
            if '_'+k in kwds:
                opts[k] = kwds.pop('_'+k)
        #print "call", opts
        return self._handler.callObj(obj=self, args=args, kwds=kwds, **opts)
    
    def _getValue(self):
        ## this just gives us an easy way to change the behavior of the special methods
        #proc = Process._processes[self._processId]
        return self._handler.getObjValue(self)
        
    
    ## Explicitly proxy special methods. Is there a better way to do this??
    
    def _getSpecialAttr(self, attr):
        #return self.__getattr__(attr)
        return self._deferredAttr(attr)
    
    def __getitem__(self, *args):
        return self._getSpecialAttr('__getitem__')(*args)
    
    def __setitem__(self, *args):
        return self._getSpecialAttr('__setitem__')(*args)
        
    def __setattr__(self, *args):
        return self._getSpecialAttr('__setattr__')(*args)
        
    def __str__(self, *args):
        return self._getSpecialAttr('__str__')(*args, _returnType=True)
        
    def __len__(self, *args):
        return self._getSpecialAttr('__len__')(*args)
    
    def __add__(self, *args):
        return self._getSpecialAttr('__add__')(*args)
    
    def __sub__(self, *args):
        return self._getSpecialAttr('__sub__')(*args)
        
    def __div__(self, *args):
        return self._getSpecialAttr('__div__')(*args)
        
    def __mul__(self, *args):
        return self._getSpecialAttr('__mul__')(*args)
        
    def __pow__(self, *args):
        return self._getSpecialAttr('__pow__')(*args)
        
    def __rshift__(self, *args):
        return self._getSpecialAttr('__rshift__')(*args)
        
    def __lshift__(self, *args):
        return self._getSpecialAttr('__lshift__')(*args)
        
    def __floordiv__(self, *args):
        return self._getSpecialAttr('__pow__')(*args)
        
    def __eq__(self, *args):
        return self._getSpecialAttr('__eq__')(*args)
    
    def __ne__(self, *args):
        return self._getSpecialAttr('__ne__')(*args)
        
    def __lt__(self, *args):
        return self._getSpecialAttr('__lt__')(*args)
    
    def __gt__(self, *args):
        return self._getSpecialAttr('__gt__')(*args)
        
    def __le__(self, *args):
        return self._getSpecialAttr('__le__')(*args)
    
    def __ge__(self, *args):
        return self._getSpecialAttr('__ge__')(*args)
        
    def __and__(self, *args):
        return self._getSpecialAttr('__and__')(*args)
        
    def __or__(self, *args):
        return self._getSpecialAttr('__or__')(*args)
        
    def __xor__(self, *args):
        return self._getSpecialAttr('__or__')(*args)
        
    def __mod__(self, *args):
        return self._getSpecialAttr('__mod__')(*args)
        
    def __radd__(self, *args):
        return self._getSpecialAttr('__radd__')(*args)
    
    def __rsub__(self, *args):
        return self._getSpecialAttr('__rsub__')(*args)
        
    def __rdiv__(self, *args):
        return self._getSpecialAttr('__rdiv__')(*args)
        
    def __rmul__(self, *args):
        return self._getSpecialAttr('__rmul__')(*args)
        
    def __rpow__(self, *args):
        return self._getSpecialAttr('__rpow__')(*args)
        
    def __rrshift__(self, *args):
        return self._getSpecialAttr('__rrshift__')(*args)
        
    def __rlshift__(self, *args):
        return self._getSpecialAttr('__rlshift__')(*args)
        
    def __rfloordiv__(self, *args):
        return self._getSpecialAttr('__rpow__')(*args)
        
    def __rand__(self, *args):
        return self._getSpecialAttr('__rand__')(*args)
        
    def __ror__(self, *args):
        return self._getSpecialAttr('__ror__')(*args)
        
    def __rxor__(self, *args):
        return self._getSpecialAttr('__ror__')(*args)
        
    def __rmod__(self, *args):
        return self._getSpecialAttr('__rmod__')(*args)
        
class DeferredObjectProxy(ObjectProxy):
    def __init__(self, parentProxy, attribute):
        ## can't set attributes directly because setattr is overridden.
        for k in ['_processId', '_typeStr', '_proxyId', '_handler']:
            self.__dict__[k] = getattr(parentProxy, k)
        self.__dict__['_parent'] = parentProxy  ## make sure parent stays alive
        self.__dict__['_attributes'] = parentProxy._attributes + (attribute,)
        self.__dict__['_proxyOptions'] = parentProxy._proxyOptions.copy()
    
    def __repr__(self):
        return ObjectProxy.__repr__(self) + '.' + '.'.join(self._attributes)
    