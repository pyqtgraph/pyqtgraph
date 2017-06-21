import pyqtgraph as pg
import numpy as np
import time
from . import relax


class ChainSim(pg.QtCore.QObject):
    
    stepped = pg.QtCore.Signal()
    relaxed = pg.QtCore.Signal()
    
    def __init__(self):
        pg.QtCore.QObject.__init__(self)
        
        self.damping = 0.1  # 0=full damping, 1=no damping
        self.relaxPerStep = 10
        self.maxTimeStep = 0.01
        
        self.pos = None      # (Npts, 2) float
        self.mass = None     # (Npts) float
        self.fixed = None    # (Npts) bool
        self.links = None    # (Nlinks, 2), uint
        self.lengths = None  # (Nlinks), float
        self.push = None     # (Nlinks), bool
        self.pull = None     # (Nlinks), bool
        
        self.initialized = False
        self.lasttime = None
        self.lastpos = None
        
    def init(self):
        if self.initialized:
            return
        
        #assert None not in [self.pos, self.mass, self.links, self.lengths]
        assert all([item is not None for item in [self.pos, self.mass, self.links, self.lengths]]) 
        
        if self.fixed is None:
            self.fixed = np.zeros(self.pos.shape[0], dtype=bool)
        if self.push is None:
            self.push = np.ones(self.links.shape[0], dtype=bool)
        if self.pull is None:
            self.pull = np.ones(self.links.shape[0], dtype=bool)
            
        
        # precompute relative masses across links
        l1 = self.links[:,0]
        l2 = self.links[:,1]
        m1 = self.mass[l1]
        m2 = self.mass[l2]
        self.mrel1 = (m1 / (m1+m2))[:,np.newaxis]
        self.mrel1[self.fixed[l1]] = 1  # fixed point constraint
        self.mrel1[self.fixed[l2]] = 0
        self.mrel2 = 1.0 - self.mrel1

        for i in range(10):
            self.relax(n=10)
        
        self.initialized = True
        
    def makeGraph(self):
        #g1 = pg.GraphItem(pos=self.pos, adj=self.links[self.rope], pen=0.2, symbol=None)
        brushes = np.where(self.fixed, pg.mkBrush(0,0,0,255), pg.mkBrush(50,50,200,255))
        g2 = pg.GraphItem(pos=self.pos, adj=self.links[self.push & self.pull], pen=0.5, brush=brushes, symbol='o', size=(self.mass**0.33), pxMode=False)
        p = pg.ItemGroup()
        #p.addItem(g1)
        p.addItem(g2)
        return p
    
    def update(self):
        # approximate physics with verlet integration
        
        now = pg.ptime.time()
        if self.lasttime is None:
            dt = 0
        else:
            dt = now - self.lasttime
        self.lasttime = now
        
        # limit amount of work to be done between frames
        if not relax.COMPILED:
            dt = self.maxTimeStep

        if self.lastpos is None:
            self.lastpos = self.pos

        # remember fixed positions
        fixedpos = self.pos[self.fixed]
        
        while dt > 0:
            dt1 = min(self.maxTimeStep, dt)
            dt -= dt1
            
            # compute motion since last timestep
            dx = self.pos - self.lastpos
            self.lastpos = self.pos
            
            # update positions for gravity and inertia
            acc = np.array([[0, -5]]) * dt1
            inertia = dx * (self.damping**(dt1/self.mass))[:,np.newaxis]  # with mass-dependent damping
            self.pos = self.pos + inertia + acc

            self.pos[self.fixed] = fixedpos  # fixed point constraint
            
            # correct for link constraints
            self.relax(self.relaxPerStep)
        self.stepped.emit()
        
        
    def relax(self, n=50):
        # speed up with C magic if possible
        relax.relax(self.pos, self.links, self.mrel1, self.mrel2, self.lengths, self.push, self.pull, n)
        self.relaxed.emit()
        
        

