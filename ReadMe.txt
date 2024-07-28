1. Create a virtual environment:
   python3 -m venv gg_project_env

2. Activate the environment:
   - On Windows: gg_project_env\Scripts\activate
   - On macOS/Linux: source gg_project_env/bin/activate

### macOS/Linux Setup

3. Install dependencies and download the Spacy model:
   ```bash
   chmod +x setup_mac.sh
   ./setup_mac.sh


#### For Windows

3. Install dependencies and download the Spacy model:
   setup_windows.bat
   
The data shoul be in data folder

4. Run the main script:
   python -m src/gg_main.py
