#Exporters = []
#from pyqtgraph import importModules
#import os
#d = os.path.split(__file__)[0]
#for mod in importModules('', globals(), locals(), excludes=['Exporter']).values():
    #if hasattr(mod, '__all__'):
        #names = mod.__all__
    #else:
        #names = [n for n in dir(mod) if n[0] != '_']
    #for k in names:
        #if hasattr(mod, k):
            #Exporters.append(getattr(mod, k))

from .Exporter import Exporter
from .ImageExporter import *
from .SVGExporter import *
from .Matplotlib import *
from .CSVExporter import *
from .PrintExporter import *


def listExporters():
    return Exporter.Exporters[:]

