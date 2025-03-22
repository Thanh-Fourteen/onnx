#!/bin/bash
g++ -I/usr/local/onnxruntime/include -I/usr/include/opencv4 src/inference.cpp -L/usr/local/onnxruntime/lib -lonnxruntime \
    $(pkg-config --libs opencv4) -o out/test


LD_LIBRARY_PATH=/usr/local/onnxruntime/lib ./out/test