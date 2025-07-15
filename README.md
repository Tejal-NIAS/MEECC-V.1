## Model for Energy Equity and Climate Compatibility (MEECC_V.1)

This is an interactive dashboard built using Dash (Plotly) to explore energy and emissions projections for countries under different development and climate scenarios.

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

- `MEECC_V.1.py` – Main dashboard code
- `Parent_sheet.xlsx` – Country-level input data
- `Population.csv` – Population projections
- `world-administrative-boundaries.shp` – Map boundaries for spatial data

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/meecc-v1.git
   cd meecc-v1

2. Install the required packages:

   pip install -r requirements.txt

3. Run the app in terminal window
   
   python MEECC_V.1.py

4. Open your browser and go to view the interactive dashboard


## Licence

This project is licensed under the Creative Commons Attribution (CC BY 4.0) License. 
You are free to use, share, and adapt the work with proper attribution.



	 
