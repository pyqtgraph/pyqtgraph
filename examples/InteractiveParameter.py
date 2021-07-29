import initExample ## Add path to library (just for examples; you do not need this)


from functools import wraps

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter as GP

app = pg.mkQApp()

class LAST_RESULT:
  """Just for testing purposes"""
  value = None

def printResult(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    LAST_RESULT.value = func(*args, **kwargs)
    QtWidgets.QMessageBox.information(QtWidgets.QApplication.activeWindow(),
                                      'Function Run!', f'Func result: {LAST_RESULT.value}')
  return wrapper

host = Parameter.create(name='Interactive Parameter Use', type='group')

@host.interactDecorator()
@printResult
def easySample(a=5, b=6):
    return a + b

@host.interactDecorator()
@printResult
def stringParams(a='5', b='6'):
    return a + b

@host.interactDecorator(a=10)
@printResult
def requiredParam(a, b=10):
    return a + b

@host.interactDecorator(ignores=['a'])
@printResult
def ignoredAParam(a=10, b=20):
    return a*b

@host.interactDecorator(runOpts=GP.RUN_BUTTON)
@printResult
def runOnButton(a=10, b=20):
    return a + b

x = 5
@host.interactDecorator(deferred={'x': lambda: x})
@printResult
def accessVarInDifferentScope(x, y=10):
    return x + y

with GP.interactiveOptsContext(runTitleFormat=str.upper):
    @host.interactDecorator()
    @printResult
    def capslocknames(a=5):
        return a

@host.interactDecorator(runOpts=(GP.RUN_CHANGED, GP.RUN_BUTTON),
                         a={'type': 'list', 'limits': [5, 10, 20]}
                         )
@printResult
def runOnBtnOrChange_listOpts(a=5):
    return a


@host.interactDecorator(nest=False)
@printResult
def onlyTheArgumentsAppear(thisIsAFunctionArg=True):
    return thisIsAFunctionArg

tree = ParameterTree()
tree.setParameters(host)

tree.show()
if __name__ == '__main__':
    pg.exec()