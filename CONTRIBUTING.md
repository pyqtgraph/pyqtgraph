# Contributing to PyQtGraph

Contributions to pyqtgraph are welcome!

Please use the following guidelines when preparing changes:

## Submitting Code Changes

* The preferred method for submitting changes is by github pull request against the "master" branch.
* Pull requests should include only a focused and related set of changes. Mixed features and unrelated changes may be rejected.
* For major changes, it is recommended to discuss your plans on the mailing list or in a github issue before putting in too much effort.
* The following deprecations are being considered by the maintainers
  * `pyqtgraph.opengl` may be deprecated and replaced with `VisPy` functionality
  * After v0.11, pyqtgraph will adopt [NEP-29](https://numpy.org/neps/nep-0029-deprecation_policy.html) which will effectively mean that python2 support will be deprecated
  * Qt4 will be deprecated shortly, as well as Qt5<5.9 (and potentially <5.12)

## Documentation

* Writing proper documentation and unit tests is highly encouraged. PyQtGraph uses pytest style testing, so tests should usually be included in a tests/ directory adjacent to the relevant code.
* Documentation is generated with sphinx; please check that docstring changes compile correctly

## Style guidelines

### Rules

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

### Pre-Commit

PyQtGraph developers are highly encouraged to (but not required) to use [`pre-commit`](https://pre-commit.com/).  `pre-commit` does a number of checks when attempting to commit the code to ensure it conforms to various standards, such as `flake8`, utf-8 encoding pragma, line-ending fixers, and so on.  If any of the checks fail, the commit will be rejected, and you will have the opportunity to make the necessary fixes before adding and committing a file again.  This ensures that every commit made conforms to (most) of the styling standards that the library enforces; and you will most likely pass the code style checks by the CI.

To make use of `pre-commit`, have it available in your `$PATH` and run `pre-commit install` from the root directory of PyQtGraph.

## Testing Setting up a test environment

### Dependencies

* tox
* tox-conda
* pytest
* pytest-cov
* pytest-xdist
* Optional: pytest-xvfb

If you have `pytest<5` (used in python2), you may also want to install `pytest-faulthandler==1.6` plugin to output extra debugging information in case of test failures. This isn't necessary with `pytest>=5`

### Tox

As PyQtGraph supports a wide array of Qt-bindings, and python versions, we make use of `tox` to test against most of the configurations in our test matrix.  As some of the qt-bindings are only installable via `conda`, `conda` needs to be in your `PATH`, and we utilize the `tox-conda` plugin.

* Tests for a module should ideally cover all code in that module, i.e., statement coverage should be at 100%.
* To measure the test coverage, un `pytest --cov -n 4` to run the test suite with coverage on 4 cores.

### Continous Integration

For our Continuous Integration, we utilize Azure Pipelines.  Tested configurations are visible on [README](README.md).  More information on coverage and test failures can be found on the respective tabs of the [build results page](https://dev.azure.com/pyqtgraph/pyqtgraph/_build?definitionId=1)
