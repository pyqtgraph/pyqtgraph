# Contributing to PyQtGraph

Contributions to pyqtgraph are welcome! Be kind and respectful! See
[our Code of Conduct](CODE_OF_CONDUCT.md) for details.

Please use the following guidelines when preparing changes:

## Development Environment Creation

First thing to do is fork the repository, and clone your own fork to your local
computer.

```bash
git clone https://github.com/<username>/pyqtgraph.git
cd pyqtgraph
```

While there is nothing preventing users from using `conda` environments, as a general
principle, we recommend using the `venv` module for creating an otherwise empty virtual
environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# on windows this would be .venv/Scripts/activate
python -m pip install numpy scipy pyqt6 -e .
```

PyQtGraph supports PySide2, PyQt5, PySide6 and PyQt6 bindings, but if a contributor is
to test against only one set of bindings, we suggest that it be PyQt6.  PyQt6 is the
only bindings that enforces the use of fully scoped enums. For example, apart from PyQt6, `QGraphicsItem.ItemIgnoresTransformations` would be recognized, but that would error in PyQt6. The cross-binding compatible reference to that enum would be
`QGraphicsItem.GraphicsItemFlags.ItemIgnoresTransformations`.

Before making changes to the code-base, create a different branch with a name that
should be unique (this makes it easier for maintainers to examine the proposed changes
locally).

```bash
git switch -c my-new-feature
```

The target of the pull request should be the `master` branch in the pyqtgraph repo.
Pull requests should include only a focused and related set of changes. Mixed features and unrelated changes may be rejected.

For major changes, it is recommended to discuss your plans on the mailing list or in a
github issue/discussion before putting in too much effort.

PyQtGraph has adopted [NEP-29](https://numpy.org/neps/nep-0029-deprecation_policy.html)
which governs the timeline for phasing out support for numpy and python versions.

## Documentation

* Writing proper documentation is highly encouraged.
* Documentation is generated with sphinx, and usage of
  [numpy-docstyle](https://numpydoc.readthedocs.io/en/latest/format.html) is encouraged
  (many places in the library do not use this docstring style at present, it's a gradual
  process to migrate).
* The docs built for this PR can be previewed by clicking on the "Details" link for the
  read-the-docs entry in the checks section of the PR conversation page.

To build the documentation locally, users will need to install
[graphviz](https://graphviz.org/) in such a way that the `dot` executable is in the
users's `PATH`.

```bash
pip install -r doc/requirements.txt 
```

```bash
cd doc
make html
```

To view the result, from the `./doc` directory, open `./build/html/index.html` in a web
browser.

To validate the documentation against numpydoc, a fork of numpydoc needs to be installed

```bash
pip install git+https://github.com/j9ac9k/numpydoc.git@see-more-accepts-sphinx-crosslinks
```

To verify your documentation changes conform to numpydoc, run:

```bash
numpydoc lint path/to/file.py
```

Every reasonable effort should be made to address the issues the linter outputs.

## Style guidelines

### Formatting ~~Rules~~ Suggestions

* PyQtGraph prefers PEP8 for most style issues, but this is not enforced rigorously as
  long as the code is clean and readable.
* Variable and Function/Methods that are intended to be part of the public API should be
  camelCase.
* "Private" methods/variables should have a leading underscore (`_`) before the name.

### Pre-Commit

PyQtGraph developers are highly encouraged to (but not required) to use
[`pre-commit`](https://pre-commit.com/).  `pre-commit` does a number of checks when
attempting to commit the code to being committed, such as ensuring no large files are
accidentally added, address mixed-line-endings types and so on.  Check the
[pre-commit documentation](https://pre-commit.com) on how to setup.

## Testing

### Basic Setup

* pytest
* Optional: pytest-qt    (used to check PyQt warnings)
* Optional: pytest-xdist (used to run tests in parallel)
* Optional: pytest-xvfb  (used on linux with headless displays)

To run the test suite, after installing the above dependencies run

```bash
pytest tests
```

In addition, the examples can be tested as well.  

```bash
pytest pyqtgraph/examples
```

The examples will each run for 1 second, and automatically close. If no unhandled
exception is created, it's considered a success.

### Tox

As PyQtGraph supports a wide array of Qt-bindings, and python versions, we make use of
`tox` to test against as many supported configurations as feasible.  With tox installed,
simply run `tox` and it will run through all the configurations.  This should be done if
there is uncertainty regarding changes working on specific combinations of PyQt bindings
and/or python versions.

### Continuous Integration

For our Continuous Integration, pyqtgraph tests across all OSs for a variety of
combinations of supported Qt bindings, python and numpy versions. For submitted changes
to be merged, it is expected that the CI passes.

### Benchmarks

( *Still under development* ) To ensure this library is performant, we use
[Air Speed Velocity (asv)](https://asv.readthedocs.io/en/stable/) to run benchmarks. For
developing on core functions and classes, be aware of any impact your changes have on
their speed. To configure and run asv:

```bash
pip install asv
python setup.py asv_config
asv run
```

( TODO publish results )
