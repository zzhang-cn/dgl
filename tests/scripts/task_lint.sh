#!/bin/bash

# cpplint
echo 'Checking code style of C++ codes...'
python3 third_party/dmlc-core/scripts/lint.py dgl cpp include src || exit 1

# pylint
echo 'Checking code style of Python codes...'
python3 -m pylint --reports=y -v --rcfile=tests/lint/pylintrc python/dgl || exit 1

# flake8
echo 'Checking for syntax errors and undefined names in Python codes...'
python3 -m pip install flake8
python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
