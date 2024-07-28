#!/bin/bash

# Install the requirements
pip install -r requirements.txt

# Download the Spacy model
python src/download_spacy_model.py

echo Installed Requirements...
