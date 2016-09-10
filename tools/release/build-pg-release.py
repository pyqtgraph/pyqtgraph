#!/usr/bin/python
from common import *


usage = """
Usage: build_pg_release.py x.y.z

    * Will attempt to clone branch release-x.y.z from %s.
    * Will attempt to contact windows host at %s (suggest running bitvise ssh server).
""" % (sourcedir, winhost)


if len(sys.argv) != 2:
    print usage
    sys.exit(-1)
version = sys.argv[1]
if re.match(r'\d+\.\d+.*', version) is None:
    print 'Invalid version number "%s".' % version
    sys.exit(-1)


# Clone source repository and tag the release branch
shell('''
    # Clone and merge release branch
    cd {bld}
    rm -rf pyqtgraph
    git clone --depth 1 -b master http://github.com/pyqtgraph/pyqtgraph
    cd {bld}/pyqtgraph
    git checkout -b release-{ver}
    git pull {src} release-{ver}
    git checkout master
    git merge --no-ff --no-commit release-{ver}
    
    # Write new version number into the source
    sed -i "s/__version__ = .*/__version__ = '{ver}'/" pyqtgraph/__init__.py
    #sed -i "s/    version=.*,/    version='{ver}',/" setup.py  # now automated
    sed -i "s/version = .*/version = '{ver}'/" doc/source/conf.py
    sed -i "s/release = .*/release = '{ver}'/" doc/source/conf.py
    
    # make sure changelog mentions unreleased changes
    grep "pyqtgraph-{ver}.*unreleased.*" CHANGELOG    
    sed -i "s/pyqtgraph-{ver}.*unreleased.*/pyqtgraph-{ver}/" CHANGELOG

    # Commit and tag new release
    git commit -a -m "PyQtGraph release {ver}"
    git tag pyqtgraph-{ver}

    # Build HTML documentation
    cd doc
        make clean
        make html
    cd ..
    find ./ -name "*.pyc" -delete

    # package source distribution
    python setup.py sdist
    
    # test pip install source distribution
    rm -rf release-{ver}-virtenv
    virtualenv --system-site-packages release-{ver}-virtenv
    . release-{ver}-virtenv/bin/activate
    echo "PATH: $PATH"
    echo "ENV: $VIRTUAL_ENV" 
    pip install --no-index dist/pyqtgraph-{ver}.tar.gz
    deactivate

    # build deb packages
    #python setup.py --command-packages=stdeb.command bdist_deb
    python setup.py --command-packages=stdeb.command sdist_dsc
    cd deb_dist/pyqtgraph-{ver}
    sed -i "s/^Depends:.*/Depends: python (>= 2.6), python-qt4 | python-pyside, python-numpy/" debian/control    
    dpkg-buildpackage
    cd ../../
    mv deb_dist dist/pyqtgraph-{ver}-deb
'''.format(**vars))


# build windows installers
if winhost is not None:
    shell("# Build windows executables")
    ssh(winhost, '''
        rmdir /s/q pyqtgraph-build
        git clone {self}:{bld}/pyqtgraph pyqtgraph-build
        cd pyqtgraph-build
        python setup.py build --plat-name=win32 bdist_wininst
        python setup.py build --plat-name=win-amd64 bdist_wininst
        exit
    '''.format(**vars))

    shell('''
        scp {win}:pyqtgraph-build/dist/*.exe {bld}/pyqtgraph/dist/
    '''.format(**vars))


print """

======== Build complete. =========

* Dist files in {bld}/pyqtgraph/dist
""".format(**vars)


if winhost is not None:
    print """    * Dist files on windows host at {win}:pyqtgraph-build/dist
    """.format(**vars)



