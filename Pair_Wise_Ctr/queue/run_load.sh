#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Pw @ 2017-03-07 17:52:09

set -e
export PATH=/home/work/Python-4.8.2/bin:/home/work/cuda-8.0/bin$PATH
export LD_LIBRARY_PATH=/home/work/Python-4.8.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5.1/cuda/lib64:/home/work/cuda-8.0/lib64:$LD_LIBRARY_PATH

head -n 100 cache|python load_img.py
