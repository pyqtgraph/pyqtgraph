Interactive Parameters
======================

As indicated by the documentation for Parameters and Parameter Trees, a
Parameter is commonly used to expose a value to the user without
burdening developers with GUI representations. The ``interact`` method
and friends extend such support to arbitrary Python functions. [1]_
Before reading further, make sure to read existing Parameter
documentation to become familiar with common extra options, creation
techniques, and so on.

Basic Use
---------

Consider a function ``a`` whose arguments should be tweakable by the
user to immediately update some result. *Without* using ``interact``,
you might do something like this:

.. code:: python

    from pyqtgraph.Qt import QtWidgets
    import pyqtgraph as pg
    from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes as ptypes

    def a(x=5, y=6):
       QtWidgets.QMessageBox.information(None, 'Hello World', f'X is {x}, Y is {y}')

    # -----------
    # discussion is from here:
    params = Parameter.create(name='"a" parameters', type='group', children=[
       dict(name='x', type='int', value=5),
       dict(name='y', type='int', value=6)
    ])

    def onChange(_param, _value):
       a(**params)

    for child in params.children():
       child.sigValueChanged.connect(onChange)
    # to here
    # -----------

    app = pg.mkQApp()
    tree = ParameterTree()
    tree.setParameters(params)
    tree.show()
    pg.exec()

Notice in the ``-----`` comment block, lots of boilerplate and value
duplication takes place. If an argument name changes, or the default
value changes, the parameter definition must be independently updated as
well. In (very common) cases like these, use ``interact`` instead (the
code below is functionally equivalent to above):

.. code:: python

    from pyqtgraph.Qt import QtWidgets
    import pyqtgraph as pg
    from pyqtgraph.parametertree import Parameter, ParameterTree, interact

    def a(x=5, y=6):
       QtWidgets.QMessageBox.information(None, 'Hello World', f'X is {x}, Y is {y}')

    # One line of code, no name/value duplication
    params = interact(a)

    app = pg.mkQApp()
    tree = ParameterTree()
    tree.setParameters(params)
    tree.show()
    pg.exec()

There are several caveats, but this is one of the most common scenarios
for function interaction.

``runOptions``
^^^^^^^^^^^^^^

Often, an ``interact``-ed function shouldn't run until multiple
parameter values are changed. Or, the function should be run every time
a value is *changing*, not just changed. In these cases, modify the
``runOptions`` parameter.

.. code:: python

    from pyqtgraph.parametertree import interact, RunOptions

    # Will add a button named "Run". When clicked, the function will run
    params = interact(a, runOptions=RunOptions.ON_ACTION)
    # Will run on any `sigValueChanging` signal
    params = interact(a, runOptions=RunOptions.ON_CHANGING)
    # Runs on `sigValueChanged` or when "Run" is pressed
    params = interact(a, runOptions=[RunOptions.ON_CHANGED, RunOptions.ON_ACTION])
    # Any combination of RUN_* options can be used

The default run behavior can also be modified. If several functions are
being interacted at once, and all should be with a "Run" button, simply
use the provided context manager:

.. code:: python

    from pyqtgraph.parametertree import interact
    # `runOptions` can be set to any combination of options as demonstrated above, too
    with interact.optsContext(runOptions=RunOptions.ON_ACTION):
        # All will have `runOptions` set to ON_ACTION
        p1 = interact(aFunc)
        p2 = interact(bFunc)
        p3 = interact(cFunc)
    # After the context, `runOptions` is back to the previous default

If the default for all interaction should be changed, you can directly
call ``interactDefaults.setOpts`` (but be warned - anyone who imports your
module will have it modified for them, too. So use the context manager
whenever possible). Thus, it is *highly* advised to make your own ``Interactor``
object in these cases. The previous options set will be returned for easy
resetting afterward:

.. code:: python

    from pyqtgraph.parametertree import Interactor
    myInteractor = Interactor()
    oldOpts = myInteractor.setOpts(runOptions=RunOptions.ON_ACTION)
    # Can also directly create interactor with these opts:
    # myInteractor = Interactor(runOptions=RunOptions.ON_ACTION)

    # ... do some things...
    # Unset option
    myInteractor.setOpts(**oldOpts)

``ignores``
^^^^^^^^^^^

When interacting with a function where some arguments should appear as
parameters and others should be hidden, use ``ignores``:

.. code:: python

    from pyqtgraph.parametertree import interact

    def a(x=5, y=6):
        print(x, y)

    # Only 'x' will show up in the parameter
    params = interact(a, ignores=['y'])

``closures``
^^^^^^^^^^^^

Sometimes, values that should be passed to the ``interact``-ed function
should come from a different scope (or "closure"), i.e. a variable definition that
should be propagated from somewhere else. In these cases, wrap that
argument in a function and pass it into ``closures`` like so. Note that
an ``InteractiveFunction`` object is needed as descibed in a later section.

.. code:: python

    from skimage import morphology as morph
    import numpy as np
    from pyqtgraph.parametertree import interact, InteractiveFunction, ParameterTree
    import pyqtgraph as pg


    def dilateImage(image, radius=3):
        image = morph.dilation(image, morph.disk(radius))
        view.setImage(image)

    app = pg.mkQApp()
    view = pg.ImageView()
    # Simulate a grayscale image
    image = np.random.randint(0, 256, size=(512, 512))
    dilate_interact = InteractiveFunction(dilateImage, closures={'image': lambda: image})
    params = interact(dilate_interact)
    # As the 'image' variable changes, the new value will be used during parameter interaction
    view.show()
    tree = ParameterTree()
    tree.setParameters(params)
    tree.show()
    image = 255 - image # Even though 'image' is reassigned, it will be used by the parameter
    pg.exec()

``parent``
^^^^^^^^^^

Often, one parameter tree is used to represent several different
interactive functions. When this is the case, specify the existing
parameter as the ``parent``. In all but simple cases, it is usually
easier to leverage the `decorator version <#the-decorator-version>`__.

.. code:: python

    from pyqtgraph.parametertree import Parameter
    def aFunc(x=5, y=6):
        QtWidgets.QMessageBox.information(None, 'Hello World', f'X is {x}, Y is {y}')
    def bFunc(first=5, second=6):
        QtWidgets.QMessageBox.information(None, 'Hello World', f'first is {first}, second is {second}')
    def cFunc(uno=5, dos=6):
        QtWidgets.QMessageBox.information(None, 'Hello World', f'uno is {uno}, dos is {dos}')

    params = Parameter.create(name='Parameters', type='group')
    # All interactions are in the same parent
    interact(aFunc, parent=params)
    interact(bFunc, parent=params)
    interact(cFunc, parent=params)

``nest``
^^^^^^^^

In all examples so far, ``interact`` makes a ``GroupParameter`` which
houses another ``GroupParameter`` inside. The inner group contains the
parameter definitions for the function arguments. If these arguments
should be directly inside the parent, use ``nest=False``:

.. code:: python

    def a(x=5, y=6):
        return x + y

    # 'x' and 'y' will be returned in a list, not nested inside another GroupParameter
    # If `parent=...` was specified in the `interact` call, `x` and `y` will be inserted
    # directly as children of `parent`
    params = interact(a, nest=False)

``runActionTemplate``
^^^^^^^^^^^^^^^^^^^^^
When the ``runOptions`` argument is set to (or contains) ``RunOptions.ON_ACTION``, a
button will be added next to the parameter group which can be clicked to run the
function with the current parameter values. The button's options can be customized
through passing a dictionary to ``runActionTemplate``. The dictionary can contain
any key accepted as an ``action`` parameter option. For instance, to run a function
either by pressing the button or a shortcut, you can interact like so:

.. code:: python

    def a(x=5, y=6):
        return x + y

    # The button will be labeled "Run" and will run the function when clicked or when
    # the shortcut "Ctrl+R" is pressed
    params = interact(a, runActionTemplate={'shortcut': 'Ctrl+R'})

    # Alternatively, add an icon to the button
    params = interact(a, runActionTemplate={'icon': 'run.png'})

    # Why not both?
    params = interact(a, runActionTemplate={'icon': 'run.png', 'shortcut': 'Ctrl+R'})

``existOk``
^^^^^^^^^^^

When ``nest=False``, there can be overlap when several function
arguments share the same name. In these cases, the result is an error
unless ``existOk=True`` (the default).

.. code:: python

    def a(x=5, y=6):
        return x + y
    def b(x=5, another=6):
        return x + another
    params = interact(a, nest=False)

    # Will raise an error, since 'x' was already in the parameter from interacting with 'a'
    interact(b, nest=False, parent=params, existOk=False)

``overrides``
^^^^^^^^^^^^^

In all examples so far, additional parameter arguments such as
``limits`` were ignored. Return to the `closures <#>`__ example and
observe what happens when ``radius`` is < 0:

::

    ValueError: All-zero footprint is not supported.

To prevent such cases, ``overrides`` can contain additional parameter
specifications (or default values) that will update the created
parameter:

.. code:: python

    # Cannot go lower than 0
    # These are bound to the 'radius' parameter
    params = interact(dilate_interact, radius={'limits': [1, None]})

Now, the user is unable to set the spinbox to a value < 1.

Similar options can be provided when the parameter type doesn’t match
the default value (``list`` is a common case):

.. code:: python

    def chooseOne(which='a'):
        print(which)

    params = interact(chooseOne, which={'type': 'list', 'limits': list('abc')})

Any value accepted in ``Parameter.create`` can be used in the override
for a parameter.

Also note that overrides can consist of raw values, in the case where
just the value should be adjusted or when there is no default:

.. code:: python

    def printAString(string):
        print(string)

    params = interact(printAString, string='anything')

Functions with ``**kwargs``
"""""""""""""""""""""""""""

Functions who allow ``**kwargs`` can accept additional specified overrides even if they don't
match argument names:

.. code:: python

    def a(**canBeNamedAnything):
        print(canBeNamedAnything)
    # 'one' and 'two' will be int parameters that appear
    params = interact(a, one=1, two=2)

If additional overrides are provided when the function *doesn't* accept keywords in this manner,
they are ignored.

The Decorator Version
---------------------

To simplify the process of interacting with multiple functions using the
same parameter, a decorator is provided:

.. code:: python

    from pyqtgraph.parametertree import Interactor, interact
    params = Parameter.create(name='Parameters', type='group')
    interactor = Interactor(parent=params) # Same parent for all `interact` calls

    info = QtWidgets.QMessageBox.information

    @interactor.decorate()
    def aFunc(x=5, y=6):
        info(None, 'Hello World', f'X is {x}, Y is {y}')

    @interactor.decorate()
    def bFunc(first=5, second=6):
        info(None, 'Hello World', f'first is {first}, second is {second}')

    @interactor.decorate()
    def cFunc(uno=5, dos=6):
        info(None, 'Hello World', f'uno is {uno}, dos is {dos}')

    # Alternatively, the default interactor can be used if you don't need to
    # make your own `Interactor` instance.
    @interact.decorate(parent=params)
    def anotherFunc(one="one"):
        print(one)

    # All interactions are in the same parent

Any value accepted by ``interact`` can be passed to the decorator.

Title Formatting
----------------

If functions should have formatted titles, specify this in the
``titleFormat`` parameter:

.. code:: python

    def my_snake_case_function(a=5):
        print(a)

    def titleFormat(name):
        return name.replace('_', ' ').title()

    # The title in the parameter tree will be "My Snake Case Function"
    params = interact(my_snake_case_function, titleFormat=titleFormat)

Using ``InteractiveFunction``
-----------------------------
In all versions of ``interact`` described so far, it is not possible to temporarily
stop an interacted function from triggering on parameter changes. Normally, one can
``disconnect`` the hooked-up signals, but since the actually connected functions are
out of scope, this is not possible when using ``interact``. Additionally, it is not
possible to change overrides or ``closures`` arguments after the fact. Finally, it
is not possible to easily call an interacted function with parameter arguments/defaults
through normal `interact` use. If any of these needs arise, use an
``InteractiveFunction`` instead during registration. This provides ``disconnect()``
and ``reconnect()`` methods, and object accessors to ``closures`` arguments.

.. code:: python

    from pyqtgraph.parametertree import InteractiveFunction, interact, Parameter, RunOptions

    def myfunc(a=5):
        print(a)

    useFunc = InteractiveFunction(myfunc)
    param = interact(useFunc, runOptions=RunOptions.ON_CHANGED)
    param['a'] = 6
    # Will print 6
    useFunc.disconnect()
    param['a'] = 5
    # Won't print anything
    useFunc.reconnect()
    param['a'] = 10
    # Will print 10

Note that in cases like these, where simple wrapping of a function must take place, you
can use ``InteractiveFunction`` like a decorator:

.. code:: python

    from pyqtgraph.parametertree import InteractiveFunction, interact, Parameter, RunOptions

    @InteractiveFunction
    def myfunc(a=5):
        print(a)

    # myfunc is now an InteractiveFunction that can be used as above
    # Also, calling `myfunc` will preserve parameter arguments
    param = interact(myfunc, RunOptions.ON_ACTION)
    param['a'] = 6

    myfunc()
    # will print '6' since this is the parameter value

.. [1]
    Functions defined in C or whose definitions cannot be parsed by
    ``inspect.signature`` cannot be used here. However, in these cases a dummy function
    can be wrapped and passed instead. Note that all values are passed
    as keywords, so if positional arguments are expected it will not work.
