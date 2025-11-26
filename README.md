## Model for Energy Equity and Climate Compatibility (MEECC_V.1)

==================================================================

This is an interactive dashboard built using Dash (Plotly) to explore energy and emissions projections for countries under different development and climate scenarios. This Readme file contains the information on how to run the dashboard in both Windows and Mac/Linux.

## What does this Dashboard do

- Groups countries into clusters based on energy, economic, health, education, infrastructure, and Emissions indicators.
- Allows selection of different scenarios:
  - GDP growth
  - Energy thresholds
  - Carbon budget allocation
  - Temparature targets (e.g., 1.5°C, 2°C)
- Projects key indicators:
  - Primary and final energy use
  - CO2 emissions (with and without mitigation)
  - Per capita metrics and energy/emission intensities

## Project Files
MEECC-V.1/
1. MEECC_V.1_revision.py # Main dashboard script
2. requirements.txt # All Python package dependencies
3. run.sh
4. run_dashboard.bat
5. LICENSE.txt
6. README.md
7. data/ # Folder for all input data
│ ├── Parent_sheet.xlsx
│ ├── Population.csv

HOW TO RUN (Windows)
---------------------

✅ Recommended: Double-click to launch
--------------------------------------
1. Make sure Python 3.8+ is installed.
2. Open the folder where the project is saved.
3. Double-click on the file named:

       run_dashboard.bat

🔔 This will:
   - Create a virtual environment (if not already created)
   - Install all required packages from requirements.txt
   - Launch the dashboard in your default browser

🕒 Note: Please wait patiently — it may take **few minutes** the first time as dependencies are installed and the server starts.

Once launched, your browser will open to:

    http://127.0.0.1:<some-port>

If not, open your browser and manually visit that address.

------------------------------------------------------------

Alternative: Run from Terminal (for advanced users)
----------------------------------------------------

1. Open a terminal (Command Prompt or PowerShell)
2. Navigate to the project folder:
       cd path\to\your\project

3. Create a virtual environment (if not already created):
       python -m venv env

4. Activate the environment:
       env\Scripts\activate

5. Install dependencies:
       pip install -r requirements.txt

6. Run the dashboard:
       python MEECC_V.1_revision.py

The dashboard will open in your browser automatically.

------------------------------------------------------------

HOW TO RUN (Mac/Linux)
-----------------------

✅ Option 1 (Recommended): Use the run.sh launcher
--------------------------------------------------
1. Make sure Python 3.8+ is installed.

2. Open the **Terminal**:
   - On Mac: Press `Command + Space`, type `Terminal`, and press Enter.
   - On Linux: Press `Ctrl + Alt + T`.

3. Navigate to the folder where this project is saved:
       cd ~/Downloads/MEECC     # Replace with your actual folder path

4. Make the script executable (only once):
       chmod +x run.sh

5. Run the script from the same terminal:
       ./run.sh

🕒 This will:
   - Create a virtual environment (if not already present)
   - Install all required packages
   - Launch the dashboard in your default browser

📌 If the browser doesn't open automatically, open it manually and go to:
       http://127.0.0.1:<some-port>

⚠️ Important: You must run `./run.sh` from the terminal — double-clicking won’t work.

------------------------------------------------------------

🛠️ Option 2 (Manual Method via Terminal)
----------------------------------------

1. Open the Terminal
2. Navigate to the project folder:
       cd ~/Downloads/MEECC     # Update with your folder path

3. Create a virtual environment:
       python3 -m venv env

4. Activate the virtual environment:
       source env/bin/activate

5. Install the required packages:
       pip install -r requirements.txt

6. Launch the dashboard:
       python3 MEECC_V.1_revision.py

✅ The dashboard will open in your browser. If not, visit:
       http://127.0.0.1:<some-port>


Required Python Packages
------------------------
See requirements.txt for exact versions.

