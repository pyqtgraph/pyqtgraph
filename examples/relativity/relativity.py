import numpy as np
import collections
import sys, os
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
import pyqtgraph.configfile
from pyqtgraph.python2_3 import xrange


class RelativityGUI(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        
        self.animations = []
        self.animTimer = QtCore.QTimer()
        self.animTimer.timeout.connect(self.stepAnimation)
        self.animTime = 0
        self.animDt = .016
        self.lastAnimTime = 0
        
        self.setupGUI()
        
        self.objectGroup = ObjectGroupParam()
        
        self.params = Parameter.create(name='params', type='group', children=[
            dict(name='Load Preset..', type='list', values=[]),
            #dict(name='Unit System', type='list', values=['', 'MKS']),
            dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]),
            dict(name='Reference Frame', type='list', values=[]),
            dict(name='Animate', type='bool', value=True),
            dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]),
            dict(name='Recalculate Worldlines', type='action'),
            dict(name='Save', type='action'),
            dict(name='Load', type='action'),
            self.objectGroup,
            ])
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Recalculate Worldlines').sigActivated.connect(self.recalculate)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Load').sigActivated.connect(self.load)
        self.params.param('Load Preset..').sigValueChanged.connect(self.loadPreset)
        self.params.sigTreeStateChanged.connect(self.treeChanged)
        
        ## read list of preset configs
        presetDir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'presets')
        if os.path.exists(presetDir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(presetDir)]
            self.params.param('Load Preset..').setLimits(['']+presets)
        
        
        
        
    def setupGUI(self):
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)
        
        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)
        
        self.splitter2 = QtGui.QSplitter()
        self.splitter2.setOrientation(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.splitter2)
        
        self.worldlinePlots = pg.GraphicsLayoutWidget()
        self.splitter2.addWidget(self.worldlinePlots)
        
        self.animationPlots = pg.GraphicsLayoutWidget()
        self.splitter2.addWidget(self.animationPlots)
        
        self.splitter2.setSizes([int(self.height()*0.8), int(self.height()*0.2)])
        
        self.inertWorldlinePlot = self.worldlinePlots.addPlot()
        self.refWorldlinePlot = self.worldlinePlots.addPlot()
        
        self.inertAnimationPlot = self.animationPlots.addPlot()
        self.inertAnimationPlot.setAspectLocked(1)
        self.refAnimationPlot = self.animationPlots.addPlot()
        self.refAnimationPlot.setAspectLocked(1)
        
        self.inertAnimationPlot.setXLink(self.inertWorldlinePlot)
        self.refAnimationPlot.setXLink(self.refWorldlinePlot)

    def recalculate(self):
        ## build 2 sets of clocks
        clocks1 = collections.OrderedDict()
        clocks2 = collections.OrderedDict()
        for cl in self.params.param('Objects'):
            clocks1.update(cl.buildClocks())
            clocks2.update(cl.buildClocks())
        
        ## Inertial simulation
        dt = self.animDt * self.params['Animation Speed']
        sim1 = Simulation(clocks1, ref=None, duration=self.params['Duration'], dt=dt)
        sim1.run()
        sim1.plot(self.inertWorldlinePlot)
        self.inertWorldlinePlot.autoRange(padding=0.1)
        
        ## reference simulation
        ref = self.params['Reference Frame']
        dur = clocks1[ref].refData['pt'][-1] ## decide how long to run the reference simulation
        sim2 = Simulation(clocks2, ref=clocks2[ref], duration=dur, dt=dt)
        sim2.run()
        sim2.plot(self.refWorldlinePlot)
        self.refWorldlinePlot.autoRange(padding=0.1)
        
        
        ## create animations
        self.refAnimationPlot.clear()
        self.inertAnimationPlot.clear()
        self.animTime = 0
        
        self.animations = [Animation(sim1), Animation(sim2)]
        self.inertAnimationPlot.addItem(self.animations[0])
        self.refAnimationPlot.addItem(self.animations[1])
        
        ## create lines representing all that is visible to a particular reference
        #self.inertSpaceline = Spaceline(sim1, ref)
        #self.refSpaceline = Spaceline(sim2)
        self.inertWorldlinePlot.addItem(self.animations[0].items[ref].spaceline())
        self.refWorldlinePlot.addItem(self.animations[1].items[ref].spaceline())
        
        
        

    def setAnimation(self, a):
        if a:
            self.lastAnimTime = pg.ptime.time()
            self.animTimer.start(int(self.animDt*1000))
        else:
            self.animTimer.stop()
            
    def stepAnimation(self):
        now = pg.ptime.time()
        dt = (now-self.lastAnimTime) * self.params['Animation Speed']
        self.lastAnimTime = now
        self.animTime += dt
        if self.animTime > self.params['Duration']:
            self.animTime = 0
            for a in self.animations:
                a.restart()
            
        for a in self.animations:
            a.stepTo(self.animTime)
            
        
    def treeChanged(self, *args):
        clocks = []
        for c in self.params.param('Objects'):
            clocks.extend(c.clockNames())
        #for param, change, data in args[1]:
            #if change == 'childAdded':
        self.params.param('Reference Frame').setLimits(clocks)
        self.setAnimation(self.params['Animate'])
        
    def save(self):
        filename = pg.QtGui.QFileDialog.getSaveFileName(self, "Save State..", "untitled.cfg", "Config Files (*.cfg)")
        if isinstance(filename, tuple):
            filename = filename[0]  # Qt4/5 API difference
        if filename == '':
            return
        state = self.params.saveState()
        pg.configfile.writeConfigFile(state, str(filename)) 
        
    def load(self):
        filename = pg.QtGui.QFileDialog.getOpenFileName(self, "Save State..", "", "Config Files (*.cfg)")
        if isinstance(filename, tuple):
            filename = filename[0]  # Qt4/5 API difference
        if filename == '':
            return
        state = pg.configfile.readConfigFile(str(filename)) 
        self.loadState(state)
        
    def loadPreset(self, param, preset):
        if preset == '':
            return
        path = os.path.abspath(os.path.dirname(__file__))
        fn = os.path.join(path, 'presets', preset+".cfg")
        state = pg.configfile.readConfigFile(fn)
        self.loadState(state)
        
    def loadState(self, state):
        if 'Load Preset..' in state['children']:
            del state['children']['Load Preset..']['limits']
            del state['children']['Load Preset..']['value']
        self.params.param('Objects').clearChildren()
        self.params.restoreState(state, removeChildren=False)
        self.recalculate()
        
        
class ObjectGroupParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Objects", addText="Add New..", addList=['Clock', 'Grid'])
        
    def addNew(self, typ):
        if typ == 'Clock':
            self.addChild(ClockParam())
        elif typ == 'Grid':
            self.addChild(GridParam())

class ClockParam(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Clock", autoIncrementName=True, renamable=True, removable=True, children=[
            dict(name='Initial Position', type='float', value=0.0, step=0.1),
            #dict(name='V0', type='float', value=0.0, step=0.1),
            AccelerationGroup(),
            
            dict(name='Rest Mass', type='float', value=1.0, step=0.1, limits=[1e-9, None]),
            dict(name='Color', type='color', value=(100,100,150)),
            dict(name='Size', type='float', value=0.5),
            dict(name='Vertical Position', type='float', value=0.0, step=0.1),
            ])
        #defs.update(kwds)
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)
            
    def buildClocks(self):
        x0 = self['Initial Position']
        y0 = self['Vertical Position']
        color = self['Color']
        m = self['Rest Mass']
        size = self['Size']
        prog = self.param('Acceleration').generate()
        c = Clock(x0=x0, m0=m, y0=y0, color=color, prog=prog, size=size)
        return {self.name(): c}
        
    def clockNames(self):
        return [self.name()]

pTypes.registerParameterType('Clock', ClockParam)
    
class GridParam(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Grid", autoIncrementName=True, renamable=True, removable=True, children=[
            dict(name='Number of Clocks', type='int', value=5, limits=[1, None]),
            dict(name='Spacing', type='float', value=1.0, step=0.1),
            ClockParam(name='ClockTemplate'),
            ])
        #defs.update(kwds)
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)
            
    def buildClocks(self):
        clocks = {}
        template = self.param('ClockTemplate')
        spacing = self['Spacing']
        for i in range(self['Number of Clocks']):
            c = list(template.buildClocks().values())[0]
            c.x0 += i * spacing
            clocks[self.name() + '%02d' % i] = c
        return clocks
        
    def clockNames(self):
        return [self.name() + '%02d' % i for i in range(self['Number of Clocks'])]

pTypes.registerParameterType('Grid', GridParam)

class AccelerationGroup(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Acceleration", addText="Add Command..")
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)
        
    def addNew(self):
        nextTime = 0.0
        if self.hasChildren():
            nextTime = self.children()[-1]['Proper Time'] + 1
        self.addChild(Parameter.create(name='Command', autoIncrementName=True, type=None, renamable=True, removable=True, children=[
            dict(name='Proper Time', type='float', value=nextTime),
            dict(name='Acceleration', type='float', value=0.0, step=0.1),
            ]))
            
    def generate(self):
        prog = []
        for cmd in self:
            prog.append((cmd['Proper Time'], cmd['Acceleration']))
        return prog    
        
pTypes.registerParameterType('AccelerationGroup', AccelerationGroup)

            
class Clock(object):
    nClocks = 0
    
    def __init__(self, x0=0.0, y0=0.0, m0=1.0, v0=0.0, t0=0.0, color=None, prog=None, size=0.5):
        Clock.nClocks += 1
        self.pen = pg.mkPen(color)
        self.brush = pg.mkBrush(color)
        self.y0 = y0
        self.x0 = x0
        self.v0 = v0
        self.m0 = m0
        self.t0 = t0
        self.prog = prog
        self.size = size

    def init(self, nPts):
        ## Keep records of object from inertial frame as well as reference frame
        self.inertData = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float), ('f', float)])
        self.refData = np.empty(nPts, dtype=[('x', float), ('t', float), ('v', float), ('pt', float), ('m', float), ('f', float)])
        
        ## Inertial frame variables
        self.x = self.x0
        self.v = self.v0
        self.m = self.m0
        self.t = 0.0       ## reference clock always starts at 0
        self.pt = self.t0      ## proper time starts at t0
        
        ## reference frame variables
        self.refx = None
        self.refv = None
        self.refm = None
        self.reft = None
        
        self.recordFrame(0)
        
    def recordFrame(self, i):
        f = self.force()
        self.inertData[i] = (self.x, self.t, self.v, self.pt, self.m, f)
        self.refData[i] = (self.refx, self.reft, self.refv, self.pt, self.refm, f)
        
    def force(self, t=None):
        if len(self.prog) == 0:
            return 0.0
        if t is None:
            t = self.pt
        
        ret = 0.0
        for t1,f in self.prog:
            if t >= t1:
                ret = f
        return ret
        
    def acceleration(self, t=None):
        return self.force(t) / self.m0
        
    def accelLimits(self):
        ## return the proper time values which bound the current acceleration command
        if len(self.prog) == 0:
            return -np.inf, np.inf
        t = self.pt
        ind = -1
        for i, v in enumerate(self.prog):
            t1,f = v
            if t >= t1:
                ind = i
        
        if ind == -1:
            return -np.inf, self.prog[0][0]
        elif ind == len(self.prog)-1:
            return self.prog[-1][0], np.inf
        else:
            return self.prog[ind][0], self.prog[ind+1][0]
        
        
    def getCurve(self, ref=True):
        
        if ref is False:
            data = self.inertData
        else:
            data = self.refData[1:]
            
        x = data['x']
        y = data['t']
        
        curve = pg.PlotCurveItem(x=x, y=y, pen=self.pen)
            #x = self.data['x'] - ref.data['x']
            #y = self.data['t']
        
        step = 1.0
        #mod = self.data['pt'] % step
        #inds = np.argwhere(abs(mod[1:] - mod[:-1]) > step*0.9)
        inds = [0]
        pt = data['pt']
        for i in range(1,len(pt)):
            diff = pt[i] - pt[inds[-1]]
            if abs(diff) >= step:
                inds.append(i)
        inds = np.array(inds)
        
        #t = self.data['t'][inds]
        #x = self.data['x'][inds]   
        pts = []
        for i in inds:
            x = data['x'][i]
            y = data['t'][i]
            if i+1 < len(data):
                dpt = data['pt'][i+1]-data['pt'][i]
                dt = data['t'][i+1]-data['t'][i]
            else:
                dpt = 1
                
            if dpt > 0:
                c = pg.mkBrush((0,0,0))
            else:
                c = pg.mkBrush((200,200,200))
            pts.append({'pos': (x, y), 'brush': c})
            
        points = pg.ScatterPlotItem(pts, pen=self.pen, size=7)
        
        return curve, points


class Simulation:
    def __init__(self, clocks, ref, duration, dt):
        self.clocks = clocks
        self.ref = ref
        self.duration = duration
        self.dt = dt
    
    @staticmethod
    def hypTStep(dt, v0, x0, tau0, g):
        ## Hyperbolic step. 
        ## If an object has proper acceleration g and starts at position x0 with speed v0 and proper time tau0
        ## as seen from an inertial frame, then return the new v, x, tau after time dt has elapsed.
        if g == 0:
            return v0, x0 + v0*dt, tau0 + dt * (1. - v0**2)**0.5
        v02 = v0**2
        g2 = g**2
        
        tinit = v0 / (g * (1 - v02)**0.5)
        
        B = (1 + (g2 * (dt+tinit)**2))**0.5
        
        v1 = g * (dt+tinit) / B
        
        dtau = (np.arcsinh(g * (dt+tinit)) - np.arcsinh(g * tinit)) / g
        
        tau1 = tau0 + dtau
        
        x1 = x0 + (1.0 / g) * ( B - 1. / (1.-v02)**0.5 )
        
        return v1, x1, tau1


    @staticmethod
    def tStep(dt, v0, x0, tau0, g):
        ## Linear step.
        ## Probably not as accurate as hyperbolic step, but certainly much faster.
        gamma = (1. - v0**2)**-0.5
        dtau = dt / gamma
        return v0 + dtau * g, x0 + v0*dt, tau0 + dtau

    @staticmethod
    def tauStep(dtau, v0, x0, t0, g):
        ## linear step in proper time of clock.
        ## If an object has proper acceleration g and starts at position x0 with speed v0 at time t0
        ## as seen from an inertial frame, then return the new v, x, t after proper time dtau has elapsed.
        

        ## Compute how much t will change given a proper-time step of dtau
        gamma = (1. - v0**2)**-0.5
        if g == 0:
            dt = dtau * gamma
        else:
            v0g = v0 * gamma
            dt = (np.sinh(dtau * g + np.arcsinh(v0g)) - v0g) / g
        
        #return v0 + dtau * g, x0 + v0*dt, t0 + dt
        v1, x1, t1 = Simulation.hypTStep(dt, v0, x0, t0, g)
        return v1, x1, t0+dt
        
    @staticmethod
    def hypIntersect(x0r, t0r, vr, x0, t0, v0, g):
        ## given a reference clock (seen from inertial frame) has rx, rt, and rv,
        ## and another clock starts at x0, t0, and v0, with acceleration g,
        ## compute the intersection time of the object clock's hyperbolic path with 
        ## the reference plane.
        
        ## I'm sure we can simplify this...
        
        if g == 0:   ## no acceleration, path is linear (and hyperbola is undefined)
            #(-t0r + t0 v0 vr - vr x0 + vr x0r)/(-1 + v0 vr)
            
            t = (-t0r + t0 *v0 *vr - vr *x0 + vr *x0r)/(-1 + v0 *vr)
            return t
        
        gamma = (1.0-v0**2)**-0.5
        sel = (1 if g>0 else 0) + (1 if vr<0 else 0)
        sel = sel%2
        if sel == 0:
            #(1/(g^2 (-1 + vr^2)))(-g^2 t0r + g gamma vr + g^2 t0 vr^2 - 
            #g gamma v0 vr^2 - g^2 vr x0 + 
            #g^2 vr x0r + \[Sqrt](g^2 vr^2 (1 + gamma^2 (v0 - vr)^2 - vr^2 + 
            #2 g gamma (v0 - vr) (-t0 + t0r + vr (x0 - x0r)) + 
            #g^2 (t0 - t0r + vr (-x0 + x0r))^2)))
            
            t = (1./(g**2 *(-1. + vr**2)))*(-g**2 *t0r + g *gamma *vr + g**2 *t0 *vr**2 - g *gamma *v0 *vr**2 - g**2 *vr *x0 + g**2 *vr *x0r + np.sqrt(g**2 *vr**2 *(1. + gamma**2 *(v0 - vr)**2 - vr**2 + 2 *g *gamma *(v0 - vr)* (-t0 + t0r + vr *(x0 - x0r)) + g**2 *(t0 - t0r + vr* (-x0 + x0r))**2)))
            
        else:
            
            #-(1/(g^2 (-1 + vr^2)))(g^2 t0r - g gamma vr - g^2 t0 vr^2 + 
            #g gamma v0 vr^2 + g^2 vr x0 - 
            #g^2 vr x0r + \[Sqrt](g^2 vr^2 (1 + gamma^2 (v0 - vr)^2 - vr^2 + 
            #2 g gamma (v0 - vr) (-t0 + t0r + vr (x0 - x0r)) + 
            #g^2 (t0 - t0r + vr (-x0 + x0r))^2)))
        
            t = -(1./(g**2 *(-1. + vr**2)))*(g**2 *t0r - g *gamma* vr - g**2 *t0 *vr**2 + g *gamma *v0 *vr**2 + g**2* vr* x0 - g**2 *vr *x0r + np.sqrt(g**2* vr**2 *(1. + gamma**2 *(v0 - vr)**2 - vr**2 + 2 *g *gamma *(v0 - vr) *(-t0 + t0r + vr *(x0 - x0r)) + g**2 *(t0 - t0r + vr *(-x0 + x0r))**2)))
        return t
        
    def run(self):
        nPts = int(self.duration/self.dt)+1
        for cl in self.clocks.values():
            cl.init(nPts)
            
        if self.ref is None:
            self.runInertial(nPts)
        else:
            self.runReference(nPts)
        
    def runInertial(self, nPts):
        clocks = self.clocks
        dt = self.dt
        tVals = np.linspace(0, dt*(nPts-1), nPts)
        for cl in self.clocks.values():
            for i in xrange(1,nPts):
                nextT = tVals[i]
                while True:
                    tau1, tau2 = cl.accelLimits()
                    x = cl.x
                    v = cl.v
                    tau = cl.pt
                    g = cl.acceleration()
                    
                    v1, x1, tau1 = self.hypTStep(dt, v, x, tau, g)
                    if tau1 > tau2:
                        dtau = tau2-tau
                        cl.v, cl.x, cl.t = self.tauStep(dtau, v, x, cl.t, g)
                        cl.pt = tau2
                    else:
                        cl.v, cl.x, cl.pt = v1, x1, tau1
                        cl.t += dt
                        
                    if cl.t >= nextT:
                        cl.refx = cl.x
                        cl.refv = cl.v
                        cl.reft = cl.t
                        cl.recordFrame(i)
                        break
            
        
    def runReference(self, nPts):
        clocks = self.clocks
        ref = self.ref
        dt = self.dt
        dur = self.duration
        
        ## make sure reference clock is not present in the list of clocks--this will be handled separately.
        clocks = clocks.copy()
        for k,v in clocks.items():
            if v is ref:
                del clocks[k]
                break
        
        ref.refx = 0
        ref.refv = 0
        ref.refm = ref.m0
        
        ## These are the set of proper times (in the reference frame) that will be simulated
        ptVals = np.linspace(ref.pt, ref.pt + dt*(nPts-1), nPts)
        
        for i in xrange(1,nPts):
                
            ## step reference clock ahead one time step in its proper time
            nextPt = ptVals[i]  ## this is where (when) we want to end up
            while True:
                tau1, tau2 = ref.accelLimits()
                dtau = min(nextPt-ref.pt, tau2-ref.pt)  ## do not step past the next command boundary
                g = ref.acceleration()
                v, x, t = Simulation.tauStep(dtau, ref.v, ref.x, ref.t, g)
                ref.pt += dtau
                ref.v = v
                ref.x = x
                ref.t = t
                ref.reft = ref.pt
                if ref.pt >= nextPt:
                    break
                #else:
                    #print "Stepped to", tau2, "instead of", nextPt
            ref.recordFrame(i)
            
            ## determine plane visible to reference clock
            ## this plane goes through the point ref.x, ref.t and has slope = ref.v
            
            
            ## update all other clocks
            for cl in clocks.values():
                while True:
                    g = cl.acceleration()
                    tau1, tau2 = cl.accelLimits()
                    ##Given current position / speed of clock, determine where it will intersect reference plane
                    #t1 = (ref.v * (cl.x - cl.v * cl.t) + (ref.t - ref.v * ref.x)) / (1. - cl.v)
                    t1 = Simulation.hypIntersect(ref.x, ref.t, ref.v, cl.x, cl.t, cl.v, g)
                    dt1 = t1 - cl.t
                    
                    ## advance clock by correct time step
                    v, x, tau = Simulation.hypTStep(dt1, cl.v, cl.x, cl.pt, g)
                    
                    ## check to see whether we have gone past an acceleration command boundary.
                    ## if so, we must instead advance the clock to the boundary and start again
                    if tau < tau1:
                        dtau = tau1 - cl.pt
                        cl.v, cl.x, cl.t = Simulation.tauStep(dtau, cl.v, cl.x, cl.t, g)
                        cl.pt = tau1-0.000001  
                        continue
                    if tau > tau2:
                        dtau = tau2 - cl.pt
                        cl.v, cl.x, cl.t = Simulation.tauStep(dtau, cl.v, cl.x, cl.t, g)
                        cl.pt = tau2
                        continue
                    
                    ## Otherwise, record the new values and exit the loop
                    cl.v = v
                    cl.x = x
                    cl.pt = tau
                    cl.t = t1
                    cl.m = None
                    break
                
                ## transform position into reference frame
                x = cl.x - ref.x
                t = cl.t - ref.t
                gamma = (1.0 - ref.v**2) ** -0.5
                vg = -ref.v * gamma
                
                cl.refx = gamma * (x - ref.v * t)
                cl.reft = ref.pt  #  + gamma * (t - ref.v * x)   # this term belongs here, but it should always be equal to 0.
                cl.refv = (cl.v - ref.v) / (1.0 - cl.v * ref.v)
                cl.refm = None
                cl.recordFrame(i)
                
            t += dt
        
    def plot(self, plot):
        plot.clear()
        for cl in self.clocks.values():
            c, p = cl.getCurve()
            plot.addItem(c)
            plot.addItem(p)

class Animation(pg.ItemGroup):
    def __init__(self, sim):
        pg.ItemGroup.__init__(self)
        self.sim = sim
        self.clocks = sim.clocks
        
        self.items = {}
        for name, cl in self.clocks.items():
            item = ClockItem(cl)
            self.addItem(item)
            self.items[name] = item
        
    def restart(self):
        for cl in self.items.values():
            cl.reset()
        
    def stepTo(self, t):
        for i in self.items.values():
            i.stepTo(t)
        

class ClockItem(pg.ItemGroup):
    def __init__(self, clock):
        pg.ItemGroup.__init__(self)
        self.size = clock.size
        self.item = QtGui.QGraphicsEllipseItem(QtCore.QRectF(0, 0, self.size, self.size))
        tr = QtGui.QTransform.fromTranslate(-self.size*0.5, -self.size*0.5)
        self.item.setTransform(tr)
        self.item.setPen(pg.mkPen(100,100,100))
        self.item.setBrush(clock.brush)
        self.hand = QtGui.QGraphicsLineItem(0, 0, 0, self.size*0.5)
        self.hand.setPen(pg.mkPen('w'))
        self.hand.setZValue(10)
        self.flare = QtGui.QGraphicsPolygonItem(QtGui.QPolygonF([
            QtCore.QPointF(0, -self.size*0.25),
            QtCore.QPointF(0, self.size*0.25),
            QtCore.QPointF(self.size*1.5, 0),
            QtCore.QPointF(0, -self.size*0.25),
            ]))
        self.flare.setPen(pg.mkPen('y'))
        self.flare.setBrush(pg.mkBrush(255,150,0))
        self.flare.setZValue(-10)
        self.addItem(self.hand)
        self.addItem(self.item)
        self.addItem(self.flare)
 
        self.clock = clock
        self.i = 1
        
        self._spaceline = None
        
        
    def spaceline(self):
        if self._spaceline is None:
            self._spaceline = pg.InfiniteLine()
            self._spaceline.setPen(self.clock.pen)
        return self._spaceline
        
    def stepTo(self, t):
        data = self.clock.refData
        
        while self.i < len(data)-1 and data['t'][self.i] < t:
            self.i += 1
        while self.i > 1 and data['t'][self.i-1] >= t:
            self.i -= 1
        
        self.setPos(data['x'][self.i], self.clock.y0)
        
        t = data['pt'][self.i]
        self.hand.setRotation(-0.25 * t * 360.)
        
        v = data['v'][self.i]
        gam = (1.0 - v**2)**0.5
        self.setTransform(QtGui.QTransform.fromScale(gam, 1.0))
        
        f = data['f'][self.i]
        tr = QtGui.QTransform()
        if f < 0:
            tr.translate(self.size*0.4, 0)
        else:
            tr.translate(-self.size*0.4, 0)
        
        tr.scale(-f * (0.5+np.random.random()*0.1), 1.0)
        self.flare.setTransform(tr)
        
        if self._spaceline is not None:
            self._spaceline.setPos(pg.Point(data['x'][self.i], data['t'][self.i]))
            self._spaceline.setAngle(data['v'][self.i] * 45.)
        
        
    def reset(self):
        self.i = 1
        

#class Spaceline(pg.InfiniteLine):
    #def __init__(self, sim, frame):
        #self.sim = sim
        #self.frame = frame
        #pg.InfiniteLine.__init__(self)
        #self.setPen(sim.clocks[frame].pen)
        
    #def stepTo(self, t):
        #self.setAngle(0)
        
        #pass

if __name__ == '__main__':
    pg.mkQApp()
    #import pyqtgraph.console
    #cw = pyqtgraph.console.ConsoleWidget()
    #cw.show()
    #cw.catchNextException()
    win = RelativityGUI()
    win.setWindowTitle("Relativity!")
    win.show()
    win.resize(1100,700)
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    
    
    #win.params.param('Objects').restoreState(state, removeChildren=False)

