# Contributing to PyQtGraph

Contributions to pyqtgraph are welcome! 

Please use the following guidelines when preparing changes:

## Submitting Code Changes

* The preferred method for submitting changes is by github pull request against the "develop" branch.
* Pull requests should include only a focused and related set of changes. Mixed features and unrelated changes may be rejected.
* For major changes, it is recommended to discuss your plans on the mailing list or in a github issue before putting in too much effort.
  * Along these lines, please note that `pyqtgraph.opengl` will be deprecated soon and replaced with VisPy.

## Documentation

* Writing proper documentation and unit tests is highly encouraged. PyQtGraph uses nose / pytest style testing, so tests should usually be included in a tests/ directory adjacent to the relevant code. 
* Documentation is generated with sphinx; please check that docstring changes compile correctly

## Style guidelines

* PyQtGraph prefers PEP8 for most style issues, but this is not enforced rigorously as long as the code is clean and readable.
* Use `python setup.py style` to see whether your code follows the mandatory style guidelines checked by flake8.
* Exception 1: All variable names should use camelCase rather than underscore_separation. This is done for consistency with Qt
* Exception 2: Function docstrings use ReStructuredText tables for describing arguments:

  ```text
  ============== ========================================================
  **Arguments:**
  argName1       (type) Description of argument
  argName2       (type) Description of argument. Longer descriptions must
                  be wrapped within the column guidelines defined by the
                  "====" header and footer.
  ============== ========================================================
  ```

  QObject subclasses that implement new signals should also describe 
  these in a similar table.
  
## Testing Setting up a test environment

### Dependencies

* tox
* tox-conda
* pytest
* pytest-cov
* pytest-xdist
* pytest-faulthandler
* Optional: pytest-xvfb

### Tox

As PyQtGraph supports a wide array of Qt-bindings, and python versions, we make use of `tox` to test against most of the configurations in our test matrix.  As some of the qt-bindings are only installable via `conda`, `conda` needs to be in your `PATH`, and we utilize the `tox-conda` plugin.

* Tests for a module should ideally cover all code in that module, i.e., statement coverage should be at 100%.
* To measure the test coverage, un `pytest --cov -n 4` to run the test suite with coverage on 4 cores.

### Continous Integration

For our Continuous Integration, we utilize Azure Pipelines.  On each OS, we test the following 6 configurations

* Python2.7 with PyQt4
* Python2.7 with PySide
* Python3.6 with PyQt5-5.9
* Python3.6 with PySide2-5.9
* Python3.7 with PyQt5-5.12
* Python3.7 with PySide2-5.12

More information on coverage and test failures can be found on the respective tabs of the [build results page](https://dev.azure.com/pyqtgraph/pyqtgraph/_build?definitionId=1)
