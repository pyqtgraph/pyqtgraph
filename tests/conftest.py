import pytest
import os
import sys

@pytest.fixture
def tmp_module(tmp_path):
    module_path = os.fsdecode(tmp_path)
    sys.path.insert(0, module_path)
    yield module_path
    sys.path.remove(module_path)
