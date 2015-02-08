DESCRIPTION = """\
PyQtGraph is a pure-python graphics and GUI library built on PyQt4/PySide and
numpy. 

It is intended for use in mathematics / scientific / engineering applications.
Despite being written entirely in python, the library is very fast due to its
heavy leverage of numpy for number crunching, Qt's GraphicsView framework for
2D display, and OpenGL for 3D display.
"""

setupOpts = dict(
    name='pyqtgraph',
    description='Scientific Graphics and GUI Library for Python',
    long_description=DESCRIPTION,
    license='MIT',
    url='http://www.pyqtgraph.org',
    author='Luke Campagnola',
    author_email='luke.campagnola@gmail.com',
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: User Interfaces",
        ],
)


import distutils.dir_util
from distutils.command import build
import os, sys, re
try:
    import setuptools
    from setuptools import setup
    from setuptools.command import install
except ImportError:
    from distutils.core import setup
    from distutils.command import install


path = os.path.split(__file__)[0]
sys.path.insert(0, os.path.join(path, 'tools'))
import setupHelpers as helpers

## generate list of all sub-packages
allPackages = (helpers.listAllPackages(pkgroot='pyqtgraph') + 
               ['pyqtgraph.'+x for x in helpers.listAllPackages(pkgroot='examples')])

## Decide what version string to use in the build
version, forcedVersion, gitVersion, initVersion = helpers.getVersionStrings(pkg='pyqtgraph')



class Build(build.build):
    """
    * Clear build path before building
    * Set version string in __init__ after building
    """
    def run(self):
        global path, version, initVersion, forcedVersion
        global buildVersion

        ## Make sure build directory is clean
        buildPath = os.path.join(path, self.build_lib)
        if os.path.isdir(buildPath):
            distutils.dir_util.remove_tree(buildPath)
    
        ret = build.build.run(self)
        
        # If the version in __init__ is different from the automatically-generated
        # version string, then we will update __init__ in the build directory
        if initVersion == version:
            return ret
        
        try:
            initfile = os.path.join(buildPath, 'pyqtgraph', '__init__.py')
            data = open(initfile, 'r').read()
            open(initfile, 'w').write(re.sub(r"__version__ = .*", "__version__ = '%s'" % version, data))
            buildVersion = version
        except:
            if forcedVersion:
                raise
            buildVersion = initVersion
            sys.stderr.write("Warning: Error occurred while setting version string in build path. "
                             "Installation will use the original version string "
                             "%s instead.\n" % (initVersion)
                             )
            sys.excepthook(*sys.exc_info())
        return ret
        

class Install(install.install):
    """
    * Check for previously-installed version before installing
    """
    def run(self):
        name = self.config_vars['dist_name']
        path = self.install_libbase
        if os.path.exists(path) and name in os.listdir(path):
            raise Exception("It appears another version of %s is already "
                            "installed at %s; remove this before installing." 
                            % (name, path))
        print("Installing to %s" % path)
        return install.install.run(self)

        
setup(
    version=version,
    cmdclass={'build': Build, 
              'install': Install,
              'deb': helpers.DebCommand, 
              'test': helpers.TestCommand,
              'debug': helpers.DebugCommand,
              'mergetest': helpers.MergeTestCommand,
              'style': helpers.StyleCommand},
    packages=allPackages,
    package_dir={'pyqtgraph.examples': 'examples'},  ## install examples along with the rest of the source
    package_data={'pyqtgraph.examples': ['optics/*.gz', 'relativity/presets/*.cfg']},
    install_requires = [
        'numpy',
        ],
    **setupOpts
)

