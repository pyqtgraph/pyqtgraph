#! /bin/bash

python configure.py

cd sip-generated

make clean
make -j 8

cp QtNativeUtils.so ../../pyqtgraph
