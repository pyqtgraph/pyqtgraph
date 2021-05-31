import initExample ## Add path to library (just for examples; you do not need this)


from functools import wraps

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
app = pg.mkQApp()

class LAST_RESULT:
  """Just for testing purposes"""
  value = None

def printResult(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    LAST_RESULT.value = func(*args, **kwargs)
    QtWidgets.QMessageBox.information(QtWidgets.QApplication.desktop(),
                                      'Function Run!', f'Func result: {LAST_RESULT.value}')
  return wrapper

host = Parameter(name='Interactive Parameter Use', type='group')

@host.interact_deco()
@printResult
def easySample(a=5, b=6):
    return a + b

@host.interact_deco()
@printResult
def stringParams(a='5', b='6'):
    return a + b

@host.interact_deco(a=10)
@printResult
def requiredParam(a, b=10):
    return a + b

@host.interact_deco(ignores=['a'])
@printResult
def ignoredAParam(a=10, b=20):
    return a*b

@host.interact_deco(runOpts=Parameter.RUN_BTN)
@printResult
def runOnButton(a=10, b=20):
    return a + b

x = 5
@host.interact_deco(deferred={'x': lambda: x})
@printResult
def accessVarInDifferentScope(x, y=10):
    return x + y

oldFmt = Parameter.RUN_TITLE_FMT
Parameter.RUN_TITLE_FMT = lambda name: name.upper()
@host.interact_deco()
@printResult
def capslocknames(a=5):
    return a
Parameter.RUN_TITLE_FMT = oldFmt

@host.interact_deco(runOpts=(Parameter.RUN_CHANGED, Parameter.RUN_BTN),
                    a={'type': 'list', 'limits': [5, 10, 20]}
)
@printResult
def runOnBtnOrChange_listOpts(a=5):
    return a


@host.interact_deco(childrenOnly=True)
@printResult
def onlyTheArgumentsAppear(thisIsAFunctionArg=True):
    return thisIsAFunctionArg

tree = ParameterTree()
tree.setParameters(host)

tree.show()
app.exec()