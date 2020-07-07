# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore
from .Exporter import Exporter
from .. import PlotItem
from .. import functions as fn

__all__ = ['MatplotlibExporter']

"""
It is helpful when using the matplotlib Exporter if your
.matplotlib/matplotlibrc file is configured appropriately.
The following are suggested for getting usable PDF output that
can be edited in Illustrator, etc.

backend      : Qt4Agg
text.usetex : True  # Assumes you have a findable LaTeX installation
interactive : False
font.family : sans-serif
font.sans-serif : 'Arial'  # (make first in list)
mathtext.default : sf
figure.facecolor : white  # personal preference
# next setting allows pdf font to be readable in Adobe Illustrator
pdf.fonttype : 42   # set fonts to TrueType (otherwise it will be 3
                    # and the text will be vectorized.
text.dvipnghack : True  # primarily to clean up font appearance on Mac

The advantage is that there is less to do to get an exported file cleaned and ready for
publication. Fonts are not vectorized (outlined), and window colors are white.

"""
    
class MatplotlibExporter(Exporter):
    Name = "Matplotlib Window"
    windows = []

    def __init__(self, item):
        Exporter.__init__(self, item)
        
    def parameters(self):
        return None

    def cleanAxes(self, axl):
        if type(axl) is not list:
            axl = [axl]
        for ax in axl:
            if ax is None:
                continue
            for loc, spine in ax.spines.items():
                if loc in ['left', 'bottom']:
                    pass
                elif loc in ['right', 'top']:
                    spine.set_color('none')
                    # do not draw the spine
                else:
                    raise ValueError('Unknown spine location: %s' % loc)
                # turn off ticks when there is no spine
                ax.xaxis.set_ticks_position('bottom')
    
    def export(self, fileName=None):
        if not isinstance(self.item, PlotItem):
            raise Exception("MatplotlibExporter currently only works with PlotItem")
    
        mpw = MatplotlibWindow()
        MatplotlibExporter.windows.append(mpw)

        fig = mpw.getFigure()

        xax = self.item.getAxis('bottom')
        yax = self.item.getAxis('left')
        
        # get labels from the graphic item
        xlabel = xax.label.toPlainText()
        ylabel = yax.label.toPlainText()
        title = self.item.titleLabel.text

        # if axes use autoSIPrefix, scale the data so mpl doesn't add its own
        # scale factor label
        xscale = yscale = 1.0
        if xax.autoSIPrefix:
            xscale = xax.autoSIPrefixScale
        if yax.autoSIPrefix:
            yscale = yax.autoSIPrefixScale

        ax = fig.add_subplot(111, title=title)
        ax.clear()
        self.cleanAxes(ax)
        for item in self.item.curves:
            x, y = item.getData()
            x = x * xscale
            y = y * yscale

            opts = item.opts
            pen = fn.mkPen(opts['pen'])
            if pen.style() == QtCore.Qt.NoPen:
                linestyle = ''
            else:
                linestyle = '-'
            color = tuple([c/255. for c in fn.colorTuple(pen.color())])
            symbol = opts['symbol']
            if symbol == 't':
                symbol = '^'
            symbolPen = fn.mkPen(opts['symbolPen'])
            symbolBrush = fn.mkBrush(opts['symbolBrush'])
            markeredgecolor = tuple([c/255. for c in fn.colorTuple(symbolPen.color())])
            markerfacecolor = tuple([c/255. for c in fn.colorTuple(symbolBrush.color())])
            markersize = opts['symbolSize']
            
            if opts['fillLevel'] is not None and opts['fillBrush'] is not None:
                fillBrush = fn.mkBrush(opts['fillBrush'])
                fillcolor = tuple([c/255. for c in fn.colorTuple(fillBrush.color())])
                ax.fill_between(x=x, y1=y, y2=opts['fillLevel'], facecolor=fillcolor)
            
            pl = ax.plot(x, y, marker=symbol, color=color, linewidth=pen.width(), 
                    linestyle=linestyle, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor,
                    markersize=markersize)

            xr, yr = self.item.viewRange()
            ax.set_xbound(xr[0]*xscale, xr[1]*xscale)
            ax.set_ybound(yr[0]*yscale, yr[1]*yscale)

        ax.set_xlabel(xlabel)  # place the labels.
        ax.set_ylabel(ylabel)
        mpw.draw()
                
MatplotlibExporter.register()        
        

class MatplotlibWindow(QtGui.QMainWindow):
    def __init__(self):
        from ..widgets import MatplotlibWidget
        QtGui.QMainWindow.__init__(self)
        self.mpl = MatplotlibWidget.MatplotlibWidget()
        self.setCentralWidget(self.mpl)
        self.show()
        
    def __getattr__(self, attr):
        return getattr(self.mpl, attr)
        
    def closeEvent(self, ev):
        MatplotlibExporter.windows.remove(self)
        self.deleteLater()
