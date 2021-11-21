import numpy as np

__all__ = ['PlotDataBuffer']

class PlotDataBuffer(object):
    """ Growing / rolling buffer with dynamic size adjustment """
    # buffers will be over-allocated by a number of elements and start filling with different offsets.
    # This avoids all the pre-allocated buffers rolling over simultaneously, at least for element-by-element addition.
    INITIAL_EXTENSION = 64 
    INITIAL_OFFSET = INITIAL_EXTENSION - 1 # start filling multiple buffers at different elements to avoid
    
    def __init__(self, *args, **kwargs): #  length=100, verbose=False):
        """ Initialize new growing / rolling buffer """
        self.diagnostics = kwargs.get('diagnostics', False)
        self.limit       = kwargs.get('limit', None) 
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            kwargs['data'] = args[0]
        data = kwargs.get('data', np.empty(0))
        self.length = len(data)
        self._buffer_length = self.length + self.INITIAL_EXTENSION
        self._buffer = np.empty( self._buffer_length )
        # start this buffer with an offset of LAG_VALUE:
        self._ptr0 = self.INITIAL_OFFSET               # points at first valid element
        self._ptrN = self.INITIAL_OFFSET + self.length # points at next free element
        self._buffer[self._ptr0 : self._ptrN] = data # copy in data
        self.INITIAL_OFFSET -= 1 # adjust so that the next buffer rolls at a different time.
        if self.INITIAL_OFFSET < 0: 
            self.INITIAL_OFFSET = self.INITIAL_EXTENSION-1
        
    def data(self):
        """ Returns data as a continuous view of internal buffer """
        return self._buffer[self._ptr0 : self._ptrN]
        
    def last(self):
        """ Returns last buffer element. Undefined if buffer is empty. """
        return self._buffer[ self._ptrN - 1 ]
        
    def add(self, data, limit=None):
        """ Adds a value or numpy array of values """
        if limit is not None: # update limit setting
            self.limit = min(1, limit)
        if np.isinstance(data, np.ndarray):
            add_length = len(data)
            add_array = True
        else:
            add_length = 1
            add_array = False
        new_len = self.length + add_length
        if new_len > self.limit:
            # roll mode
            new_ptrN = self._ptrN + add_length
            if new_ptrN < self._buffer_length:
                # fits existing buffer
                if add_array:
                    self._buffer[self._ptrN:new_ptrN] = data
                else:
                    self._buffer[self._ptrN] = data
                self._ptrN = new_ptrN
            else:
                # roll buffer
        
        self._buffer[ self._ptrN ] = value
        self._ptrN += 1
        
        
        self._ptr0 = max( self._ptr0, self._ptrN - self.length )
        if self._ptrN >= self._ext_length:
            if self.verbose: print('roll!')
            self._buffer[:self.length] = self._buffer[EXTENSION:] # shift left by EXTENSION elements
            self._ptr0 -= EXTENSION # move start and end pointers by the same steps
            self._ptrN -= EXTENSION 