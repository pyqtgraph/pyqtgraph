from functools import wraps

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
    Parameter,
    ParameterTree,
    RunOptions,
    InteractiveFunction,
    Interactor,
)

app = pg.mkQApp()


class LAST_RESULT:
    """Just for testing purposes"""

    value = None


def printResult(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        LAST_RESULT.value = func(*args, **kwargs)
        QtWidgets.QMessageBox.information(
            QtWidgets.QApplication.activeWindow(),
            "Function Run!",
            f"Func result: {LAST_RESULT.value}",
        )

    return wrapper


host = Parameter.create(name="Interactive Parameter Use", type="group")
interactor = Interactor(parent=host, runOptions=RunOptions.ON_CHANGED)


@interactor.decorate()
@printResult
def easySample(a=5, b=6):
    return a + b


@interactor.decorate()
@printResult
def stringParams(a="5", b="6"):
    return a + b


@interactor.decorate(a=10)
@printResult
def requiredParam(a, b=10):
    return a + b


@interactor.decorate(ignores=["a"])
@printResult
def ignoredAParam(a=10, b=20):
    return a * b


@interactor.decorate(runOptions=RunOptions.ON_ACTION)
@printResult
def runOnButton(a=10, b=20):
    return a + b


x = 5


@printResult
def accessVarInDifferentScope(x, y=10):
    return x + y


func_interactive = InteractiveFunction(
    accessVarInDifferentScope, closures={"x": lambda: x}
)
# Value is redeclared, but still bound
x = 10
interactor(func_interactive)


with interactor.optsContext(titleFormat=str.upper):

    @interactor.decorate()
    @printResult
    def capslocknames(a=5):
        return a


@interactor.decorate(
    runOptions=(RunOptions.ON_CHANGED, RunOptions.ON_ACTION),
    a={"type": "list", "limits": [5, 10, 20]},
)
@printResult
def runOnBtnOrChange_listOpts(a=5):
    return a


@interactor.decorate(nest=False)
@printResult
def onlyTheArgumentsAppear(thisIsAFunctionArg=True):
    return thisIsAFunctionArg


tree = ParameterTree()
tree.setParameters(host)

tree.show()
if __name__ == "__main__":
    pg.exec()
