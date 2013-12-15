from distutils.core import setup
import distutils.dir_util
import os, re
from subprocess import check_output

## generate list of all sub-packages
path = os.path.abspath(os.path.dirname(__file__))
n = len(path.split(os.path.sep))
subdirs = [i[0].split(os.path.sep)[n:] for i in os.walk(os.path.join(path, 'pyqtgraph')) if '__init__.py' in i[2]]
all_packages = ['.'.join(p) for p in subdirs] + ['pyqtgraph.examples']


## Make sure build directory is clean before installing
buildPath = os.path.join(path, 'build')
if os.path.isdir(buildPath):
    distutils.dir_util.remove_tree(buildPath)


## Determine current version string
init = open(os.path.join(path, 'pyqtgraph', '__init__.py')).read()
m = re.search(r'__version__ = (\S+)\n', init)
if m is None:
    raise Exception("Cannot determine version number!")
version = m.group(1).strip('\'\"')
initVersion = version

# If this is a git checkout, append the current commit
if os.path.isdir(os.path.join(path, '.git')):
    def gitCommit(name):
        commit = check_output(['git', 'show', name], universal_newlines=True).split('\n')[0]
        assert commit[:7] == 'commit '
        return commit[7:]
    
    # Find last tag matching "pyqtgraph-.*"
    tagNames = check_output(['git', 'tag'], universal_newlines=True).strip().split('\n')
    while True:
        if len(tagNames) == 0:
            raise Exception("Could not determine last tagged version.")
        lastTagName = tagNames.pop()
        if re.match(r'pyqtgraph-.*', lastTagName):
            break
        
    # is this commit an unchanged checkout of the last tagged version? 
    lastTag = gitCommit(lastTagName)
    head = gitCommit('HEAD')
    if head != lastTag:
        branch = re.search(r'\* (.*)', check_output(['git', 'branch'])).group(1)
        version = version + "-%s-%s" % (branch, head[:10])
    
    # any uncommitted modifications?
    modified = False
    status = check_output(['git', 'status', '-s'], universal_newlines=True).strip().split('\n')
    for line in status:
        if line[:2] != '??':
            modified = True
            break        
                
    if modified:
        version = version + '+'

print("PyQtGraph version: " + version)

import distutils.command.build

class Build(distutils.command.build.build):
    def run(self):
        ret = distutils.command.build.build.run(self)
        
        # If the version in __init__ is different from the automatically-generated
        # version string, then we will update __init__ in the build directory
        global path, version, initVersion
        if initVersion == version:
            return ret
        
        initfile = os.path.join(path, self.build_lib, 'pyqtgraph', '__init__.py')
        data = open(initfile, 'r').read()
        open(initfile, 'w').write(re.sub(r"__version__ = .*", "__version__ = '%s'" % version, data))
        return ret
        

setup(name='pyqtgraph',
    version=version,
    cmdclass={'build': Build},
    description='Scientific Graphics and GUI Library for Python',
    long_description="""\
PyQtGraph is a pure-python graphics and GUI library built on PyQt4/PySide and
numpy. 

It is intended for use in mathematics / scientific / engineering applications.
Despite being written entirely in python, the library is very fast due to its
heavy leverage of numpy for number crunching, Qt's GraphicsView framework for
2D display, and OpenGL for 3D display.
""",
    license='MIT',
    url='http://www.pyqtgraph.org',
    author='Luke Campagnola',
    author_email='luke.campagnola@gmail.com',
    packages=all_packages,
    package_dir={'pyqtgraph.examples': 'examples'},  ## install examples along with the rest of the source
    #package_data={'pyqtgraph': ['graphicsItems/PlotItem/*.png']},
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
    install_requires = [
        'numpy',
        'scipy',
        ],
)

