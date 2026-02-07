## Model for Energy Equity and Climate Compatibility (MEECC_V.1)

==================================================================

This is an interactive dashboard built using Dash (Plotly) to explore energy and emissions projections for countries under different development and climate scenarios. 
This README file contains information on how to run the dashboard in both Windows and Mac/Linux.

## What does this dashboard do?

- Groups countries into clusters based on energy, economic, health, education, infrastructure, and emissions indicators.
- Allows selection of different scenarios:
  - GDP growth
  - Energy thresholds
  - Carbon budget allocation
  - Temperature targets (e.g., 1.5°C, 2°C)
- Projects key indicators:
  - Primary energy consumption
  - CO2 emissions (with and without mitigation)
  - Per capita metrics and energy/emission intensities

## Project Files
```
MEECC-V.1/
├── MEECC_V.1_revision.py       # Main dashboard script
├── requirements.txt            # All Python package dependencies
├── run.sh                      # Shell script to launch app (Linux/Mac)
├── run_dashboard.bat           # Batch file to launch app (Windows)
├── LICENSE.txt                 # License (CC BY)
├── README.md                   # This file
└── data/                       # Folder containing all input data
    ├── Parent_sheet.xlsx
    └── Population.csv
```

How to Run (Windows)
---------------------

✅ Recommended: Double-click to launch
--------------------------------------
1. Make sure the latest version of Python is installed.
2. Open the folder where the project is saved.
3. Double-click on the file named:

       run_dashboard.bat

🔔 This will:
   - Create a virtual environment (if not already created)
   - Install all required packages from requirements.txt
   - Launch the dashboard in your default browser

🕒 Note: Please wait patiently — it may take **a few minutes** the first time as dependencies are installed and the server starts.

Once launched, your browser will open to:

    http://127.0.0.1:<some-port>

✅ The dashboard will open in your browser automatically. If not, visit:
       http://127.0.0.1:<some-port> as displayed in the terminal.

------------------------------------------------------------

Alternative: Run from Command Prompt or PowerShell (for advanced users)
----------------------------------------------------

1. Open a Command Prompt or PowerShell
2. Navigate to the project folder:
     cd ~/Downloads/MEECC-V.1     (Update with your folder path)

3. Create a virtual environment (if not already created):
      python -m venv env  

4. Activate the environment:
      env\Scripts\activate

5. Install dependencies:
      pip install -r requirements.txt

6. Run the dashboard:
      python MEECC_V.1_revision.py

✅ The dashboard will open in your browser automatically. If not, visit:
       http://127.0.0.1:<some-port> as displayed in the terminal.

------------------------------------------------------------

How to Run (Mac/Linux)
-----------------------

✅ Option 1 (Recommended): Use the run.sh launcher
--------------------------------------------------
1. Make sure the latest version of Python is installed.

2. Open the **Terminal**:
   - On Mac: Press `Command + Space`, type `Terminal`, and press Enter.
   - On Linux: Press `Ctrl + Alt + T`.

3. Navigate to the folder where this project is saved:
       cd ~/Downloads/MEECC-V.1    (Update with your folder path)

4. Make the script executable (only once):
       chmod +x run.sh

5. Run the script from the same Terminal:
       ./run.sh

🕒 This will:
   - Create a virtual environment (if not already present)
   - Install all required packages
   - Launch the dashboard in your default browser

✅ The dashboard will open in your browser automatically. If not, visit:
       http://127.0.0.1:<some-port> as displayed in the terminal.

⚠️ Important: You must run `./run.sh` from the Terminal — double-clicking won’t work.

------------------------------------------------------------

🛠️ Option 2 (Manual Method via Terminal)
----------------------------------------

1. Open the Terminal
2. Navigate to the project folder:
       cd ~/Downloads/MEECC-V.1     # Update with your folder path

3. Create a virtual environment:
       python3 -m venv env

4. Activate the virtual environment:
       source env/bin/activate

5. Install the required packages:
       pip install -r requirements.txt

6. Launch the dashboard:
       python3 MEECC_V.1_revision.py

✅ The dashboard will open in your browser. If not, visit:
       http://127.0.0.1:<some-port> as displayed in the terminal.


Required Python Packages
------------------------
See requirements.txt for exact versions.
