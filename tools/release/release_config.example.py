"""
Example configuration file required by build-pg-release and upload-pg-release.

Copy this file to release_config.py and edit.
"""

# Where to find the repository from which the release files will be built.
# This repository must have a branch called release-x.y.z
sourcedir = '/home/user/pyqtgraph'

# Where to generate build files--source packages, deb packages, .exe installers, etc.
builddir = '/home/user/pyqtgraph-build'

# Where to archive build files (optional)
archivedir = builddir + '/archive'

# A windows machine (typically a VM) running an SSH server for automatically building .exe installers
winhost = 'luke@192.168.56.101'
