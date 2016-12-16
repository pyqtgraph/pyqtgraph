from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.functions import siScale

import numpy as np

__all__ = ['SliderWidget']

class SliderWidget(QtGui.QWidget):
    '''
    shows a horizontal/vertical slider with a label showing its value
    '''
    sigValueChanged = QtCore.Signal(object)  ## value

    def __init__(self, horizontal=True, parent=None):
        '''
        horizontal -> True/False
        '''
        QtGui.QWidget.__init__(self, parent)
        self.mn, self.mx = None, None
        self.precission = 0
        self.step = 100
        self.valueLen=2
        self._suffix = None
 
        self.label = QtGui.QLabel()
        self.label.setFont(QtGui.QFont('Courier'))
        self.slider = QtGui.QSlider(QtCore.Qt.Orientation(
                        1 if horizontal else 0), self)#1...horizontal
        self.slider.setTickPosition(
                        QtGui.QSlider.TicksAbove if horizontal else QtGui.QSlider.TicksLeft)
        #self.slider.setRange (0, 100)
        self.slider.sliderMoved.connect(self._updateLabel)        
        self._updateLabel(self.slider.value())
        
        layout = QtGui.QHBoxLayout() if horizontal else QtGui.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
  
    def value(self):
        return self._value

    def setValue(self, val):
        if val == None:
            val = self.mn
        if self.mn != None:
            val = (val-self.mn) / (self.mx-self.mn) 
            val *= 99.0
            val = int(round(val))
        self.slider.setValue(val)
        self._updateLabel(val)

    def setRange(self, mn, mx):
        '''
        mn, mx -> arbitrary values that are not equal
        '''
        if mn == mx:
            raise ValueError('limits must be different values')
        self.mn = float(min(mn,mx))
        self.mx = float(max(mn,mx))
        self._calcPrecission()
        self._updateLabel(self.slider.value())

    def setSuffix(self, suffix):
        self._suffix = suffix
        self._updateLabel(self.slider.value())

    def _calcPrecission(self):
        #number of floating points:
        self.precission = int(round( np.log10( (self.step / (self.mx-self.mn) )) ) )
        if self.precission < 0: 
            self.precission = 0
        #length of the number in the label:
        self.valueLen = max(len(str(int(self.mn))), len(str(int(self.mx))) ) + self.precission

    def setOpts(self, bounds=None):
        if bounds != None:
            self.setRange(*bounds)

    def _updateLabel(self, val):
        if self.mn != None:
            val /= 99.0 #val->0...1
            val = val *(self.mx-self.mn) + self.mn  
        self._value = val = round(val, self.precission)
        self.sigValueChanged.emit(val)

        if self._suffix:
            scale,pre = siScale(val)
            txt = '%s %s%s' %(val*scale,pre,self._suffix)
        else:
            #to have a fixed width of the label: format the value to a given length:
            txt = format(val, '%s.%sf' %(self.valueLen,self.precission))            
        self.label.setText(txt)



if __name__ == '__main__':
    import sys
    app = QtGui.QApplication([])
    s = SliderWidget()
    s.setRange(2,7)
    s.setValue(4.6)
    s.setSuffix('m')
    s.show()
    sys.exit(app.exec_())
    