@echo off
echo Installing requirements...
pip install -r requirements.txt

echo Downloading Spacy model...
python src/download_spacy_model.py

echo Installed Requirements...