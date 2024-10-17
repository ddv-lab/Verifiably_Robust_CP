#!/bin/bash

# Define the file path
FILE="./venv/lib/python3.9/site-packages/auto_LiRPA-0.5.0-py3.9.egg/auto_LiRPA/operators/pooling.py"

# Use sed to replace 'view' with 'reshape' on line 552
sed -i '552s/view/reshape/' "$FILE"