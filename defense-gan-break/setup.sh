#!/bin/bash

tar -xvf break/intermediates_1.0/intermediates.tar -C ./break/intermediates_1.0/
tar -xvf break/intermediates_2.5/intermediates_2.5.tar -C ./break/intermediates_2.5
tar -xvf break/intermediates_3.0/intermediates_3.0.tar -C break/intermediates_3.0/
tar -xvf break/intermediates_3.5/intermediates_3.5.tar -C break/intermediates_3.5/
tar -xvf break/intermediates_3.0_200/intermediates_3.0_200.tar -C break/intermediates_3.0_200/
tar -xvf break/intermediates_3.0_300/intermediates_3.0_300.tar -C break/intermediates_3.0_300/
python3 -m venv --system-site-packages env
source env/bin/activate
python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
