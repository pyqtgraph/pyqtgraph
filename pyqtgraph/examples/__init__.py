import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run():
    from . import ExampleApp
    ExampleApp.main()
