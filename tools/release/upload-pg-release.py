#!/usr/bin/python

from release_config import *
import os

usage = """
upload-pg-release.py x.y.z

    * Uploads source dist to pypi
    * Uploads packages & docs to website
    * Pushes new master branch to github

""" % (source, winhost)



pypi_err = """
Missing ~/.pypirc file. Should look like:
-----------------------------------------

[distutils]
index-servers =
    pypi

[pypi]
username:your_username
password:your_password

"""

if not os.path.isfile(os.path.expanduser('~/.pypirc')):
    print pypi_err
    sys.exit(-1)

### Upload everything to server
shell("""
    # Uploading documentation..
    cd pyqtgraph
    rsync -rv doc/build/* slice:/www/code/pyqtgraph/pyqtgraph/documentation/build/

    # Uploading source dist to website
    rsync -v dist/pyqtgraph-{ver}.tar.gz slice:/www/code/pyqtgraph/downloads/
    cp dist/pyqtgraph-{ver}.tar.gz ../archive

    # Upload deb to website
    rsync -v dist/pyqtgraph-{ver}-deb/python-pyqtgraph_{ver}-1_all.deb slice:/www/code/pyqtgraph/downloads/
    
    # Update APT repository..
    ssh slice "cd /www/debian; ln -sf /www/code/pyqtgraph/downloads/*.deb dev/; dpkg-scanpackages dev /dev/null | gzip -9c > dev/Packages.gz"
    cp -a dist/pyqtgraph-{ver}-deb ../archive/

    # Uploading windows executables..
    rsync -v dist/*.exe slice:/www/code/pyqtgraph/downloads/
    cp dist/*.exe ../archive/

    # Push to github
    git push --tags https://github.com/pyqtgraph/pyqtgraph master:master

    # Upload to pypi..
    python setup.py sdist upload
    
    
""".format(**vars))

print """

======== Upload complete. =========

Next steps to publish:
    - update website
    - mailing list announcement
    - new conda recipe (http://conda.pydata.org/docs/build.html)
    - contact deb maintainer (gianfranco costamagna)
    - other package maintainers?

""".format(**vars)
