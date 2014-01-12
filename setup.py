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


from distutils.core import setup
import distutils.dir_util
import os, sys, re

path = os.path.split(__file__)[0]
sys.path.insert(0, os.path.join(path, 'tools'))
import setupHelpers as helpers

## generate list of all sub-packages
allPackages = helpers.listAllPackages(pkgroot='pyqtgraph') + ['pyqtgraph.examples']

## Decide what version string to use in the build
version, forcedVersion, gitVersion, initVersion = helpers.getVersionStrings(pkg='pyqtgraph')


import distutils.command.build

class Build(distutils.command.build.build):
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
    
        ret = distutils.command.build.build.run(self)
        
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
        
from distutils.core import Command
import shutil, subprocess

class DebCommand(Command):
    description = "build .deb package"
    user_options = []
    def initialize_options(self):
        self.cwd = None
    def finalize_options(self):
        self.cwd = os.getcwd()
    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root: %s' % self.cwd
        global version
        pkgName = "python-pyqtgraph-" + version
        debDir = "deb_build"
        if os.path.isdir(debDir):
            raise Exception('DEB build dir already exists: "%s"' % debDir)
        sdist = "dist/pyqtgraph-%s.tar.gz" % version
        if not os.path.isfile(sdist):
            raise Exception("No source distribution; run `setup.py sdist` first.")
        
        # copy sdist to build directory and extract
        os.mkdir(debDir)
        renamedSdist = 'python-pyqtgraph_%s.orig.tar.gz' % version
        shutil.copy(sdist, os.path.join(debDir, renamedSdist))
        if os.system("cd %s; tar -xzf %s" % (debDir, renamedSdist)) != 0:
            raise Exception("Error extracting source distribution.")
        buildDir = '%s/pyqtgraph-%s' % (debDir, version)
        
        # copy debian control structure
        shutil.copytree('tools/debian', buildDir+'/debian')
        
        # Write changelog
        #chlog = subprocess.check_output([sys.executable, 'tools/generateChangelog.py', 'CHANGELOG'])
        #open('%s/pyqtgraph-%s/debian/changelog', 'w').write(chlog)
        if os.system('python tools/generateChangelog.py CHANGELOG %s > %s/debian/changelog' % (version, buildDir)) != 0:
            raise Exception("Error writing debian/changelog")
        
        # build package
        if os.system('cd %s; debuild -us -uc' % buildDir) != 0:
            raise Exception("Error during debuild.")

class TestCommand(Command):
    description = ""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        global cmd
        cmd = self
        
setup(
    version=version,
    cmdclass={'build': Build, 'deb': DebCommand, 'test': TestCommand},
    packages=allPackages,
    package_dir={'pyqtgraph.examples': 'examples'},  ## install examples along with the rest of the source
    #package_data={'pyqtgraph': ['graphicsItems/PlotItem/*.png']},
    install_requires = [
        'numpy',
        'scipy',
        ],
    **setupOpts
)

