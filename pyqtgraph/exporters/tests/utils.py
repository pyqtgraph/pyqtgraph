import tempfile
import uuid
import os


def gentempfilename(dir=None, suffix=None):
    """Generate a temporary file with a random name

    Defaults to whatever the system thinks is a temporary location

    Parameters
    ----------
    suffix : str, optional
        The suffix of the file name (The thing after the last dot).
        If 'suffix' does not begin with a dot then one will be prepended

    Returns
    -------
    str
        The filename of a unique file in the temporary directory
    """
    if dir is None:
        dir = tempfile.gettempdir()
    if suffix is None:
        suffix = ''
    elif not suffix.startswith('.'):
        suffix = '.' + suffix
    print('tempfile.tempdir = %s' % tempfile.tempdir)
    print('suffix = %s' % suffix)
    return os.path.join(dir, str(uuid.uuid4()) + suffix)


def gentempdir(dir=None):
    """Generate a temporary directory

    Parameters
    ----------
    dir : str, optional
        The directory to create a temporary directory im. If None, defaults
        to the place on disk that the system thinks is a temporary location

    Returns
    -------
    str
        The path to the temporary directory
    """
    return tempfile.mkdtemp(dir=dir)
