import json
import os
import re
import subprocess
import sys
from contextlib import suppress
from setuptools import Command
from typing import Any, Dict

def getInitVersion(pkgroot):
    """Return the version string defined in __init__.py"""
    path = os.getcwd()
    initfile = os.path.join(path, pkgroot, '__init__.py')
    init = open(initfile).read()
    m = re.search(r'__version__ = (\S+)\n', init)
    if m is None or len(m.groups()) != 1:
        raise Exception("Cannot determine __version__ from init file: "
                        + "'%s'!" % initfile)
    version = m.group(1).strip('\'\"')
    return version

def gitCommit(name):
    """Return the commit ID for the given name."""
    commit = subprocess.check_output(
        ['git', 'show', name],
        universal_newlines=True).split('\n')[0]
    assert commit[:7] == 'commit '
    return commit[7:]

def getGitVersion(tagPrefix):
    """Return a version string with information about this git checkout.
    If the checkout is an unmodified, tagged commit, then return the tag
    version

    If this is not a tagged commit, return the output of
    ``git describe --tags``

    If this checkout has been modified, append "+" to the version.
    """
    path = os.getcwd()
    if not os.path.isdir(os.path.join(path, '.git')):
        return None

    try:
        v = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--dirty", '--match="%s*"' % tagPrefix],
                stderr=subprocess.DEVNULL)
            .strip()
            .decode("utf-8")
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    # chop off prefix
    assert v.startswith(tagPrefix)
    v = v[len(tagPrefix):]

    # split up version parts
    parts = v.split('-')

    # has working tree been modified?
    modified = False
    if parts[-1] == 'dirty':
        modified = True
        parts = parts[:-1]

    # have commits been added on top of last tagged version?
    # (git describe adds -NNN-gXXXXXXX if this is the case)
    local = None
    if (len(parts) > 2 and
        re.match(r'\d+', parts[-2]) and
        re.match(r'g[0-9a-f]{7}', parts[-1])):
        local = parts[-1]
        parts = parts[:-2]

    gitVersion = '-'.join(parts)
    if local is not None:
        gitVersion += '+' + local
    if modified:
        gitVersion += 'm'

    return gitVersion

def getGitBranch():
    m = re.search(
        r'\* (.*)',
        subprocess.check_output(['git', 'branch'],
        universal_newlines=True))
    if m is None:
        return ''
    else:
        return m.group(1)

def getVersionStrings(pkg):
    """
    Returns 4 version strings:

      * the version string to use for this build,
      * version string requested with --force-version (or None)
      * version string that describes the current git checkout (or None).
      * version string in the pkg/__init__.py,

    The first return value is (forceVersion or gitVersion or initVersion).
    """

    ## Determine current version string from __init__.py
    initVersion = getInitVersion(pkgroot=pkg)

    # If this is a git checkout
    # try to generate a more descriptive version string
    try:
        gitVersion = getGitVersion(tagPrefix=pkg+'-')
    except:
        gitVersion = None
        sys.stderr.write("This appears to be a git checkout, but an error "
                        "occurred while attempting to determine a version "
                        "string for the current commit.\n")
        sys.excepthook(*sys.exc_info())

    # See whether a --force-version flag was given
    forcedVersion = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--force-version'):
            if arg == '--force-version':
                forcedVersion = sys.argv[i+1]
                sys.argv.pop(i)
                sys.argv.pop(i)
            elif arg.startswith('--force-version='):
                forcedVersion = sys.argv[i].replace('--force-version=', '')
                sys.argv.pop(i)


    ## Finally decide on a version string to use:
    if forcedVersion is not None:
        version = forcedVersion
    else:
        version = initVersion
        # if git says this is a modified branch, add local version information
        if gitVersion is not None:
            _, _, local = gitVersion.partition('+')
            if local != '':
                version = version + '+' + local
                sys.stderr.write("Detected git commit; "
                                 + "will use version string: '%s'\n" % version)

    return version, forcedVersion, gitVersion, initVersion


DEFAULT_ASV: Dict[str, Any] = {
    "version": 1,
    "project": "pyqtgraph",
    "project_url": "https://pyqtgraph.org/",
    "repo": ".",
    "branches": ["master"],
    "environment_type": "virtualenv",
    "show_commit_url": "https://github.com/pyqtgraph/pyqtgraph/commit/",
    "pythons": ["3.12", "3.13", "3.14",],
    "matrix": {
        "env_nobuild": {
            "PYQTGRAPH_QT_LIB": ["PySide6", "PyQt5", "PyQt6"]
        },
        "req": {
            "pyqt6": [""],
            "pyqt5": [""],
            "PySide6-Essentials": [""],
            "numba": [""],  # always have numba, parametrize not using it...,
        }
    },
    "benchmark_dir": "benchmarks",
    "env_dir": ".asv/env",
    "results_dir": ".asv/results",
    "html_dir": ".asv/html",
    "build_cache_size": 5,
    "build_command": [
        "python -m pip install build",
        "python -m build --wheel -o {build_cache_dir} {build_dir}"
    ],
    "install_command": [
        "in-dir={env_dir} python -mpip install --no-deps {wheel_file}",
        "in-dir={env_dir} python -mpip install colorama"
    ]
}


class ASVConfigCommand(Command):
    description = "Setup the ASV benchmarking config for this system"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        config = DEFAULT_ASV
        with suppress(FileNotFoundError, subprocess.CalledProcessError):
            cuda_check = subprocess.check_output(["nvcc", "--version"])
            if (match := re.search(r"release (\d{1,2}\.\d)", cuda_check.decode("utf-8"))):
                ver = match.groups()[0]  # e.g. 11.0
                major, _, _ = ver.partition(".")
                config["matrix"]["req"][f"cupy-cuda{major}x"] = [""]

        with open("asv.conf.json", "w") as conf_file:
            conf_file.write(json.dumps(config, indent=2))
