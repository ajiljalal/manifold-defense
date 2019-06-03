#!/bin/bash

python3 -m venv --system-site-packages env
source env/bin/activate
python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
