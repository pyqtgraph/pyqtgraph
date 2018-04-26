import tempfile, os, sys, shutil
import pyqtgraph as pg
import pyqtgraph.reload


pgpath = os.path.join(os.path.dirname(pg.__file__), '..')

# make temporary directory to write module code
path = None

def setup_module():
    # make temporary directory to write module code
    global path
    path = tempfile.mkdtemp()
    sys.path.insert(0, path)

def teardown_module():
    global path
    shutil.rmtree(path)
    sys.path.remove(path)


code = """
import sys
sys.path.append('{path}')

import pyqtgraph as pg

class C(pg.QtCore.QObject):
    sig = pg.QtCore.Signal()
    def fn(self):
        print("{msg}")

"""

def remove_cache(mod):
    if os.path.isfile(mod+'c'):
        os.remove(mod+'c')
    cachedir = os.path.join(os.path.dirname(mod), '__pycache__')
    if os.path.isdir(cachedir):
        shutil.rmtree(cachedir)


def test_reload():
    py3 = sys.version_info >= (3,)

    # write a module
    mod = os.path.join(path, 'reload_test.py')
    open(mod, 'w').write(code.format(path=path, msg="C.fn() Version1"))

    # import the new module
    import reload_test

    c = reload_test.C()
    c.sig.connect(c.fn)
    if py3:
        v1 = (reload_test.C, reload_test.C.sig, reload_test.C.fn, c.sig, c.fn, c.fn.__func__)
    else:
        v1 = (reload_test.C, reload_test.C.sig, reload_test.C.fn, reload_test.C.fn.__func__, c.sig, c.fn, c.fn.__func__)



    # write again and reload
    open(mod, 'w').write(code.format(path=path, msg="C.fn() Version2"))
    remove_cache(mod)
    pg.reload.reloadAll(path, debug=True)
    if py3:
        v2 = (reload_test.C, reload_test.C.sig, reload_test.C.fn, c.sig, c.fn, c.fn.__func__)
    else:
        v2 = (reload_test.C, reload_test.C.sig, reload_test.C.fn, reload_test.C.fn.__func__, c.sig, c.fn, c.fn.__func__)

    if not py3:
        assert c.fn.im_class is v2[0]
    oldcfn = pg.reload.getPreviousVersion(c.fn)
    if oldcfn is None:
        # Function did not reload; are we using pytest's assertion rewriting?
        raise Exception("Function did not reload. (This can happen when using py.test"
            " with assertion rewriting; use --assert=plain for this test.)")
    if py3:
        assert oldcfn.__func__ is v1[2]
    else:
        assert oldcfn.im_class is v1[0]
        assert oldcfn.__func__ is v1[2].__func__
    assert oldcfn.__self__ is c


    # write again and reload
    open(mod, 'w').write(code.format(path=path, msg="C.fn() Version2"))
    remove_cache(mod)
    pg.reload.reloadAll(path, debug=True)
    if py3:
        v3 = (reload_test.C, reload_test.C.sig, reload_test.C.fn, c.sig, c.fn, c.fn.__func__)
    else:
        v3 = (reload_test.C, reload_test.C.sig, reload_test.C.fn, reload_test.C.fn.__func__, c.sig, c.fn, c.fn.__func__)

    #for i in range(len(old)):
        #print id(old[i]), id(new1[i]), id(new2[i]), old[i], new1[i]

    cfn1 = pg.reload.getPreviousVersion(c.fn)
    cfn2 = pg.reload.getPreviousVersion(cfn1)

    if py3:
        assert cfn1.__func__ is v2[2]
        assert cfn2.__func__ is v1[2]
    else:
        assert cfn1.__func__ is v2[2].__func__
        assert cfn2.__func__ is v1[2].__func__
        assert cfn1.im_class is v2[0]
        assert cfn2.im_class is v1[0]
    assert cfn1.__self__ is c
    assert cfn2.__self__ is c

    pg.functions.disconnect(c.sig, c.fn)

