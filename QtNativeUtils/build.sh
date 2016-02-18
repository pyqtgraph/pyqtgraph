#! /bin/bash

python configure.py

cd sip-generated

make clean
make

cp QtNativeUtils.so ../../pyqtgraph
