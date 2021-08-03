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

``runOpts``
-----------

Often, an ``interact``-ed function shouldn’t run until multiple
parameter values are changed. Or, the function should be run every time
a value is *changing*, not just changed. In these cases, modify the
``runOpts`` parameter.

.. code:: python

   from pyqtgraph.parametertree import interact, RunOpts

   # Will add a button named "Run". When clicked, the function will run
   params = interact(a, runOpts=RunOpts.ON_BUTTON)
   # Will run on any `sigValueChanging` signal
   params = interact(a, runOpts=RunOpts.ON_CHANGING)
   # Runs on `sigValueChanged` or when "Run" is pressed
   params = interact(a, runOpts=[RunOpts.ON_CHANGED, RunOpts.ON_BUTTON])
   # Any combination of RUN_* options can be used

The default run behavior can also be modified. If several functions are
being interacted at once, and all should be with a “Run” button, simply
use the provided context manager:

.. code:: python

   # `runOpts` can be set to any combination of options as demonstrated above, too
   with RunOpts.optsContext(defaultRunOpts=RunOpts.ON_BUTTON):
       # All will have `runOpts` set to ON_BUTTON
       p1 = interact(aFunc)
       p2 = interact(bFunc)
       p3 = interact(cFunc)
   # After the context, `runOpts` is back to the previous default

If the default for all interaction should be changed, you can directly
modify ``defaultRunOpts`` (but be warned – anyone who imports your
module will have it modified for them, too. So use the context manager
whenever possible)

.. code:: python

   RunOpts.defaultRunOpts = RunOpts.ON_BUTTON

``ignores``
-----------

When interacting with a function where some arguments should appear as
parameters and others should be hidden, use ``ignores``:

.. code:: python

   from pyqtgraph.parametertree import interact

   def a(x=5, y=6):
       print(x, y)

   # Only 'x' will show up in the parameter
   params = interact(a, ignores=['y'])

``deferred``
------------

Sometimes, values that should be passed to the ``interact``-ed function
should come from a different scope, i.e. a variable definition that
should be propagated from somewhere else. In these cases, wrap that
argument in a function and pass it into ``deferred`` like so:

.. code:: python

   from skimage import morphology as morph
   import numpy as np
   from pyqtgraph.parametertree import interact
   import pyqtgraph as pg


   def dilateImage(image, radius=3):
       image = morph.dilation(image, morph.disk(radius))
       view.setImage(image)

   app = pg.mkQApp()
   view = pg.ImageView()
   # Simulate a grayscale image
   image = np.random.randint(0, 256, size=(512, 512))
   params = interact(dilateImage, deferred={'image': lambda: image})
   # As the 'image' variable changes, the new value will be used during parameter interaction
   view.show()
   pg.exec()

``parent``
----------

Often, one parameter tree is used to represent several different
interactive functions. When this is the case, specify the existing
parameter as the ``parent``. In all but simple cases, it is usually
easier to leverage the `decorator
version <#The%20Decorator%20Version>`__

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

``runFunc``
-----------

Often, override or decorator functions will use a definition only
accepting kwargs and pass them to a different function. When this is the
case, pass the raw, undecorated version to ``interact`` and pass the
actual function to run here. I.e. use ``runFunc`` in the following
scenario:

.. code:: python

   def a(x=5, y=6):
       return x + y

   def aWithLog(**kwargs):
       print('Running A')
       return a(**kwargs)

   params = interact(a, runFunc=aWithLog)

``nest``
--------

In all examples so far, ``interact`` makes a ``GroupParameter`` which
houses another ``GroupParameter`` inside. The inner group contains the
parameter definitions for the function arguments. If these arguments
should be directly inside the parent, use ``nest=False``:

.. code:: python

   def a(x=5, y=6):
       return x + y

   # 'x' and 'y' will be direct descendants of 'params', not nested inside another GroupParameter
   params = interact(a, nest=False)

``existOk``
-----------

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
-------------

In all examples so far, additional parameter arguments such as
``limits`` were ignored. Return to the `deferred <#>`__ example and
observe what happens when ``radius`` is < 0:

::

   ValueError: All-zero footprint is not supported.

To prevent such cases, ``overrides`` can contain additional parameter
specifications (or default values) that will update the created
parameter:

.. code:: python

   # Cannot go lower than 0
   # These are bound to the 'radius' parameter
   params = interact(dilateImage, deferred={'image': lambda: image}, radius={'limits': [0, None]})

Now, the user is unable to set the spinbox to a value < 0.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
=====================

To simplify the process of interacting with multiple functions using the
same parameter, a decorator is provided:

.. code:: python

   params = Parameter.create(name='Parameters', type='group')

   @params.interactDecorator()
   def aFunc(x=5, y=6):
       QtWidgets.QMessageBox.information(None, 'Hello World', f'X is {x}, Y is {y}')

   @params.interactDecorator()
   def bFunc(first=5, second=6):
       QtWidgets.QMessageBox.information(None, 'Hello World', f'first is {first}, second is {second}')

   @params.interactDecorator()
   def cFunc(uno=5, dos=6):
       QtWidgets.QMessageBox.information(None, 'Hello World', f'uno is {uno}, dos is {dos}')

   # All interactions are in the same parent

Any value accepted by ``interact`` can be passed to the decorator.

Title Formatting
----------------

If functions should have formatted titles, specify this in the
``runTitleFormat`` parameter:

.. code:: python

   def my_snake_case_function(a=5):
       print(a)

   def titleFormat(name):
       return name.replace('_', ' ').title()

   with RunOpts.optsContext(runTitleFormat=titleFormat):
       # The title in the parameter tree will be "My Snake Case Function"
       params = interact(my_snake_case_function)

Extra Options in the Docstring
==============================

With ``docstring_parser``
-------------------------

If the ``docstring_parser`` python package is available on your system,
you can add additional parameter options directly to your argument
documentation, provided your docstrings are well-formed. Returning to
the ``overrides`` example about dilating an image, instead of specifying
a ``limits`` override in the call to ``interact``, you can also do the
following:

.. code:: python

   def dilateImage(image, radius=3):
       """
       Dilates an image.
       :param radius: the dilation radius
       limits=[0, None]
       """
       image = morph.dilation(image, morph.disk(radius))
       view.setImage(image)

   # Also valid
   def dilateImage(image, radius=3):
       """
       Dilates an image.
       
       Parameters
       ----------
       radius: int
           The radius
           limits = [0, None]
       """
       
   # You get the idea

``limits`` will be added to the parameter just as if it was an
``override``.

See the ``docstring_parser`` package information for a list of supported
documentation standards.

Also note that ``docstring_parser`` will add any non-\ ``ini`` formatted
strings as a tooltip text, which is a helpful method of exposing
function documentation to the user.

Without ``docstring_parser``
----------------------------

If ``docstring_parser`` is not available on your system, or your
documentation does not conform to a supported style, you can also
manually denote parameter options simply by including appropriate
headers (``[arg.options]``, where ``arg`` is the argument name):

.. code:: python

   def dilateImage(image, radius=3):
       """
       Dilates an image.
       
       [radius.options]
       limits = [0, None]
       """
       image = morph.dilation(image, morph.disk(radius))
       view.setImage(image)
       
   # Also valid
   def dilateImage(image, radius=3):
       """
       Dilates an image.
       
       :param radius: My radius
       [radius.options]
       limits = [0, None]
       """

   # Also valid
   def dilateImage(image, radius=3):
       """
       Dilates an image.
       
       Parameters
       ----------
       radius: int
           The radius
           [radius.options]
           limits = [0, None]
       """
       
   # You get the idea

Docstring Limitations / Considerations
--------------------------------------

* ``ast.literal_eval`` is used to convert option values, so they cannot refer to anything other than builtin objects.
  If you want other defaults like ``np.linspace(-np.pi, np.pi)``, you must specify this as an ``override``. The details
  for this are in the corresponding section above.

* Since ``ini`` parsing is used behind the scenes, standard rules apply
  (no duplicate section headers, etc.).

* If any ``[*.options]`` section headers are present in the documentation,
  the non-\ ``docstring_parser`` evaluation will be used regardless of whether ``docstring_parser`` is available.

* If ``docstring_parser`` fails to parse the argument list, no output from the docstring will be
  forwarded to `interact`. Therefore, make sure that the function documentation is well-formed and
  that parsing works properly before rolling this out. Or, specify section headers manually.

.. [1]
   Functions defined in C or whose definitions cannot be parsed by
   ``inspect.signature`` cannot be used here. However, in these cases a dummy function
   can be wrapped while the C function is passed to the ``runFunc`` argument.
