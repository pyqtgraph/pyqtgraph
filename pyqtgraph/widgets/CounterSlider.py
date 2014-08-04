# -*- coding: utf-8 -*-
"""
New widget with a QSlider and a Counter (QPushButton and QLineEdit) used for 
example to navigate into a 3D array.

This widget is heavily inspired from the QwtSlider and the QwtCounter widgets.

See the CounterSlider example in the examples folder for one example of use.

@author : Vincent Le Saux (vincent.le_saux@ensta-bretagne.fr) 

TODO : make sure the labels of the slider axis item are never trimmed
TODO : add properly the QPushButton icons (they are in the pixmaps directory
       but I do not know how to add them from this directory. Any help is 
       welcomed!
"""

from ..Qt import QtGui, QtCore
from .GraphicsView import GraphicsView
from ..graphicsItems.GraphicsWidget import GraphicsWidget
from ..graphicsItems.AxisItem import AxisItem


class CounterSlider(QtGui.QWidget):
    
    """
    **Bases:** :class:`QWidget <QtGui.QWidget>`

    =============================== ===================================================
    **Signals:**
    sigValueChange        int, emitted when the "value" of the widget has changed 
                          This signal is emitted when the slider is moved, the
                          text in the lineedit has changed or when one button
                          has been pressed 
    =============================== ===================================================
    """    

    sigValueChanged = QtCore.Signal(object)

    def __init__(self, parent=None, value=500, mini=1, maxi=1000,
                 inc=[1, 10, 100], pen=None, enableSlider=True,
                 enableCounter=True):
        """
        =============== =======================================================
        **Arguments:**
        
        value           int, the initial value of the widget
        mini            int, the minimum value of the widget
        maxi            int, the maximum value of the widget
        inc             [int, int, int], the increment of each button 
        pen             QPen, the pen used to draw the axis
        enableSlider    boolean, set to True if the slider is visible. Set to
                        False otherwise
        enableCounter   boolean, set to True if the counter (lineedit and
                        sets of buttons) is visible. Set to False otherwise
        =============== =======================================================
        """
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Counter Widget')
        self.inc = inc
        self.val = value
        self.min = mini
        self.max = maxi
        self.nButtons = 3
        self.enableSlider = enableSlider
        self.enableCounter = enableCounter
        self.buttonsUp = []
        self.buttonsDown = []
        if pen is None:
            pen='k'
        signalMapper = QtCore.QSignalMapper(self)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(self.min, self.max)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.axis = AxisWidget(pen=pen)
        self.axis.item.axis.setRange(self.min, self.max)

        vLayout = QtGui.QVBoxLayout()
        vLayout.setSpacing(0)
        vLayout.setMargin(0)
        vLayout.addWidget(self.slider)
        vLayout.addWidget(self.axis)

        hLayout = QtGui.QHBoxLayout()
        hLayout.setSpacing(0)
        hLayout.setMargin(0)
        for n in range(self.nButtons):
            n = self.nButtons-n
            button = QtGui.QPushButton()
            icon = QtGui.QIcon("down"+str(n)+".svg")
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(20, 24))
            button.setFixedWidth(20)
            button.setFixedHeight(24)
            button.setDisabled(True)
            self.buttonsDown.append(button)
            hLayout.addWidget(button)
            button.clicked.connect(signalMapper.map)
            signalMapper.setMapping(button, n) 

        self.value = QtGui.QLineEdit()
        self.value.setReadOnly(False)
        self.value.setValidator(QtGui.QIntValidator())
        self.value.setMinimumWidth(50)
        self.value.setMaximumWidth(50)
        self.value.setText(str(1))
        hLayout.addWidget(self.value)

        for n in range(self.nButtons):
            button = QtGui.QPushButton()
            icon = QtGui.QIcon("up"+str(n)+".svg")
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(20, 24))
            button.setFixedWidth(20)
            button.setFixedHeight(24)
            self.buttonsUp.append(button)
            hLayout.addWidget(button)
            button.clicked.connect(signalMapper.map)
            inc = 11-self.nButtons
            n += self.nButtons+inc
            signalMapper.setMapping(button, n)

        hhLayout = QtGui.QHBoxLayout()
        hhLayout.addLayout(vLayout)
        hhLayout.addLayout(hLayout) 

        mainLayout = QtGui.QVBoxLayout()
        mainLayout.setSpacing(4)
        mainLayout.setMargin(4)        
        mainLayout.addLayout(hhLayout)
        mainLayout.addStretch(1)
                
        self.setLayout(mainLayout)
        self.showWidgets()        
        self.setMaximumHeight(self.minimumSizeHint().height())
        self.update()
        
        self.slider.sliderMoved.connect(self.sliderChanged)
        self.value.editingFinished.connect(self.textChanged)
        signalMapper.mapped.connect(self.btnClicked)

    def sliderChanged(self, value):
        self.setValue(value)
        self.checkValue()
        self.updateButtons()
        self.value.setText(str(self.getValue()))
        self.sigValueChanged.emit(self.getValue())

    def btnClicked(self, index):
        """ Called when on Button is cliked
            Arguments : index (int)
            Values : 1,  2,  3  for the back buttons
                     11, 12, 13 for the up buttons  """
        if (index == 1) or (index == 2) or (index==3):
            inc = -self.inc[index-1]
        else:
            inc = self.inc[index-11]
        self.incValue(inc)
        self.update()
        self.value.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sigValueChanged.emit(self.getValue())

    def textChanged(self):
        value = int(self.value.text())
        self.setValue(value)
        self.update()
        self.sigValueChanged.emit(self.getValue())

    def checkValue(self):
        value = self.getValue()
        if value <= self.min:
            self.value.setFocusPolicy(QtCore.Qt.NoFocus)
            self.setValue(self.min)
        elif value > self.max:
            self.setValue(self.max)

    def updateButtons(self):
        value = self.getValue()
        for n in range(self.nButtons):
            button = self.buttonsUp[n]
            button.setEnabled(value < self.max)
            button = self.buttonsDown[n]
            button.setEnabled(value > self.min)

    def setValue(self, value):
        self.val = value

    def getValue(self):
        return self.val

    def incValue(self, inc):
        self.val += inc

    def setRange(self, mini, maxi):
        self.min = mini
        self.max = maxi
        self.slider.setRange(mini, maxi)
        self.axis.item.axis.setRange(mini, maxi)
        self.update()

    def setIncrement1(self, inc):
        """
        Define the increment related to the first button
            inc is a float
        """
        self.inc[0] = inc

    def setIncrement2(self, inc):
        """
        Define the increment related to the second button
            inc is a float
        """
        self.inc[1] = inc

    def setIncrement3(self, inc):
        """
        Define the increment related to the third button
            inc is a float
        """
        self.inc[2] = inc

    def setIncrement(self, inc):
        """
        Define the increments related to the three buttons
            inc is a list of float
        """
        self.inc = inc

    def update(self):
        self.checkValue()
        self.updateButtons()
        self.value.setText(str(self.getValue()))
        self.slider.setValue(self.getValue())

    def showWidgets(self):
        """ Show or hide the widgets (coming feature) """
        if not self.enableSlider:
            self.slider.hide()
            self.axis.hide()
        if not self.enableCounter:
            self.value.hide()
            for button in self.buttonsDown:
                button.hide()
            for button in self.buttonsUp:
                button.hide()


class AxisWidget(GraphicsView):
    def __init__(self, parent=None, *args, **kargs):
        GraphicsView.__init__(self, parent, background=None)
        self.item = AxisItemWidget(*args, **kargs)
        self.setCentralItem(self.item)
        self.setCacheMode(self.CacheNone)
        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.TextAntialiasing)
        self.setFrameStyle(QtGui.QFrame.NoFrame | QtGui.QFrame.Plain)
        self.setMinimumHeight(self.item.height())


class AxisItemWidget(GraphicsWidget):
    def __init__(self, pen=(0, 0, 0), *args, **kargs):
        GraphicsWidget.__init__(self)
        self.layout = QtGui.QGraphicsLinearLayout()
        self.layout.setContentsMargins(8,0,9,0)  
        self.axis = AxisItem('bottom', maxTickLength=10, 
                             showValues=True, pen=pen)
        self.layout.addItem(self.axis)
        self.setLayout(self.layout)
        
