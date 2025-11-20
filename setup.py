from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()
setupOpts = dict(
    name='pyqtgraph',
    description='Scientific Graphics and GUI Library for Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license =  'MIT',
    url='http://www.pyqtgraph.org',
    project_urls={
        'Documentation': 'https://pyqtgraph.readthedocs.io',
        'Source': 'https://github.com/pyqtgraph/pyqtgraph',
    },
    author='Luke Campagnola',
    author_email='luke.campagnola@gmail.com',
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: User Interfaces",
    ],
)


import os
import re
import sys

from setuptools import find_namespace_packages, setup
from setuptools.command import install

path = os.path.split(__file__)[0]
sys.path.append(path)
import tools.setupHelpers as helpers

## Decide what version string to use in the build
version, forcedVersion, gitVersion, initVersion = helpers.getVersionStrings(pkg='pyqtgraph')



class Install(install.install):
    """
    * Check for previously-installed version before installing
    * Set version string in __init__ after building. This helps to ensure that we
      know when an installation came from a non-release code base.
    """
    def run(self):
        global path, version, initVersion, forcedVersion, installVersion

        name = self.config_vars['dist_name']
        path = os.path.join(self.install_libbase, 'pyqtgraph')
        if os.path.exists(path):
            raise Exception("It appears another version of %s is already "
                            "installed at %s; remove this before installing." 
                            % (name, path))
        print("Installing to %s" % path)
        rval = install.install.run(self)

        
        # If the version in __init__ is different from the automatically-generated
        # version string, then we will update __init__ in the install directory
        if initVersion == version:
            return rval
        
        try:
            initfile = os.path.join(path, '__init__.py')
            with open(initfile, "r") as file_:
                data = file_.read()
            with open(initfile, "w") as file_:
                file_.write(re.sub(r"__version__ = .*", "__version__ = '%s'" % version, data))
            installVersion = version
        except:
            sys.stderr.write("Warning: Error occurred while setting version string in build path. "
                             "Installation will use the original version string "
                             "%s instead.\n" % (initVersion)
                             )
            if forcedVersion:
                raise
            installVersion = initVersion
            sys.excepthook(*sys.exc_info())
    
        return rval

setup(
    version=version,
    cmdclass={
        'asv_config': helpers.ASVConfigCommand,
    },
    packages=find_namespace_packages(include=['pyqtgraph', 'pyqtgraph.*']),
    python_requires=">=3.12",
    package_dir={"pyqtgraph": "pyqtgraph"},
    package_data={
        'pyqtgraph.examples': ['optics/*.gz', 'relativity/presets/*.cfg'],
        "pyqtgraph.icons": ["**/*.svg", "**/*.png"],
        "pyqtgraph": [
            "colors/maps/*.csv",
            "colors/maps/*.txt",
            "colors/maps/*.hex",
        ],
    },
    install_requires = [
        'numpy>=2.0.0',
        'colorama'
    ],
    **setupOpts
)
