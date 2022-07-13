Detail the reasoning behind the code change.  If there is an associated issue that this PR will resolve add

Fixes #<IssueNumber>

### Other Tasks 

<details>
  <summary>Bump Dependency Versions</summary>

### Files that need updates
    
Confirm the following files have been either updated or there has been a determination that no update is needed.

- [ ] `README.md`
- [ ] `setup.py`
- [ ] `tox.ini`
- [ ] `.github/workflows/main.yml` and associated `requirements.txt` and conda `environemt.yml` files
- [ ] `pyproject.toml`
- [ ] `binder/requirements.txt`

</details>

<details>
    <summary>Pre-Release Checklist</summary>

### Pre Release Checklist

- [ ] Update version info in `__init__.py`
- [ ] Update `CHANGELOG` primarily using contents from automated changelog generation in GitHub release page
- [ ] Have git tag in the format of pyqtgraph-<version>

</details>


<details>
  <summary>Post-Release Checklist</summary>

### Steps To Complete

- [ ] Append `.dev0` to `__version__` in `__init__.py`
- [ ] Announce on mail list
- [ ] Announce on Twitter

</details>
