import os
import sys
from pathlib import Path
from setuptools import find_namespace_packages, setup

root_path = Path(__file__).parent
sys.path.append(os.fsdecode(root_path))
import tools.setupHelpers as helpers

version = helpers.getInitVersion(root_path / "pyqtgraph")
long_description = (root_path / "README.md").read_text()

setup(
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
)
