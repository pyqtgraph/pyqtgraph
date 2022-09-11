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

        req_buffer_length = int( 1.3 * (self.length) ) + self.INITIAL_EXTENSION # over-allocate by 30%
        if self.limit is not None:
            max_buffer_length = int( 1.3 * (self.limit) ) + self.INITIAL_EXTENSION # over-allocate by 30%
            req_buffer_length > min(req_buffer_length, max_buffer_length) # do not exceed 30% over element limit
        self._buffer = np.empty( req_buffer_length )

        # start this buffer with an offset of LAG_VALUE:
        self._ptrStart = self.INITIAL_OFFSET               # points at first valid element
        self._ptrFree  = self.INITIAL_OFFSET + self.length # points at next free element
        self._buffer[self._ptrStart : self._ptrFree] = data # copy in data
        self.INITIAL_OFFSET -= 1 # adjust so that the next buffer rolls at a different time.
        if self.INITIAL_OFFSET < 0: 
            self.INITIAL_OFFSET = self.INITIAL_EXTENSION-1
        
    def data(self):
        """ Returns data as a continuous view of internal buffer """
        return self._buffer[self._ptrStart : self._ptrFree]
        
    def last(self):
        """ Returns last buffer element. Undefined if buffer is empty. """
        return self._buffer[ self._ptrFree - 1 ]
        
    def add(self, data, limit=None):
        """ Adds a value or numpy array of values """
        if limit is not None: # update limit setting
            self.limit = max(1, limit)
        if np.isinstance(data, np.ndarray):
            add_length = len(data)
            add_array = True
        else:
            add_length = 1
            add_array = False
        new_ptrFree = self._ptrFree + add_length # pointer to next available element after adding data
        if new_ptrFree >= len( self._buffer ): # buffer is full. 
            new_buffer = self._buffer
            # step 1: re-allocate only if needed:
            cur_buffer_length = len(self._buffer)
            req_buffer_length = int( 1.3 * (cur_buffer_length + add_length) ) + self.INITIAL_EXTENSION # over-allocate by 30%
            if limit is None: # grow without limit
                new_buffer = np.empty( req_buffer_length )
            else: # take care not to exceed buffer limit
                max_buffer_length = int( 1.3 * self.limit ) + self.INITIAL_EXTENSION
                if cur_buffer_length != max_buffer_length: # differs from targeted length
                    if req_buffer_length < max_buffer_length:
                        new_buffer = np.empty( req_buffer_length ) # grow by 30% over-allocation
                    else:
                        new_buffer = np.empty( max_buffer_length ) # maintain at 30% over element limit
            # step 2: relocate data into new buffer or new location in existing buffer
            if self.limit is not None:
                self.length = self.limit - add_length # Remaining length of original data
            if self.length > 0: # updated buffer still holds some original data
                new_buffer[:self.length] = self._buffer[ (self._ptrFree-self.length):self._ptrFree ] # copy latest elements according to new length
                self._ptrStart = 0
                self._ptrFree  = self.length
                new_ptrFree = self._ptrFree + add_length # pointer to next available element after adding data
        # step 3: add new data. Space is available, unless a negative self._ptrFree indicates that appended data alone exceeds limit
        if self._ptrFree >= 0:
            if add_array:
                self._buffer[ self._ptrFree : new_ptrFree ] = data
            else:
                self._buffer[ self._ptrFree ] = data
        else:
            data_start = -self._ptrFree # only array data can exceed the buffer length
            self._buffer[ 0 : new_ptrFree ] = data[ data_start: ]
        self._ptrFree = new_ptrFree
        self.length += add_length
