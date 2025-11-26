
#pip install dash==2.18.2
#pip install pandas==2.2.2
#pip install plotly==5.22.0
#pip install scikit-learn==1.4.2
#pip install matplotlib==3.8.4
#pip install geopandas==1.0.1
#pip install umap-learn==0.5.7
#pip install seaborn==0.13.2

import os
import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
import umap.umap_ as umap
import seaborn as sns
import logging
import matplotlib
from dash.dash_table.Format import Format, Scheme
matplotlib.use('Agg')
import webbrowser
from threading import Timer
import socket

# This gets the directory where the app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Set the path to the "data" folder inside this directory
DATA_PATH = os.path.join(BASE_DIR, "data")

#Innitialisation of Data files
data = pd.read_excel(os.path.join(DATA_PATH, 'Parent_sheet.xlsx'))
data.loc[data["Countries"]=="Namibia","ISO 3166"]="NA"
print(type(data))
data_norm=data.iloc[:,12:27]  
print(type(data_norm))
len(data_norm)
data_norm.shape
pop = pd.read_csv(os.path.join(DATA_PATH, 'Population.csv'))
population=pd.read_csv(os.path.join(DATA_PATH, 'Population.csv'))
selected_columns=["Countries", "2019"]
population_2019=population[selected_columns]
pop.drop(pop.index[179:185], inplace=True)
pop.drop(columns=["2019"], inplace=True)

# =============================================================================
# #Section 1: Country Clustering
# =============================================================================

#UMAP + K Means
umap_model = umap.UMAP(
    n_neighbors=15,       # Number of neighbors for local structure
    min_dist=0.1,         # Controls compactness of embedding
    n_components=2,       # Dimensionality of the embedding (2D or 3D)
    metric='euclidean',   # Distance metric
    random_state=42       # For reproducibility
)

# Fit and transform the data
data_filled=np.nan_to_num(data_norm)
data_normalised=(data_filled-data_filled.min())/(data_filled.max()-data_filled.min())
data_umap = umap_model.fit_transform(data_normalised)

# Convert embedding to DataFrame for easy plotting
umap_df = pd.DataFrame(data_umap, columns=['UMAP1', 'UMAP2'])


kmeans=KMeans(n_clusters=5, random_state=42)
clusters=kmeans.fit_predict(data_umap)
data['cluster']=clusters
cluster_means=data.groupby("cluster")["PCGDP_2019_$/perons -2017PPP"].mean().sort_values(ascending=False)
reordered_clusters={old:new+1 for new, old in enumerate(cluster_means.index)}
data["cluster_umap"]=data["cluster"].map(reordered_clusters)
data_describe=data.columns[11:26]
descriptive=data.groupby("cluster_umap")[data_describe].agg("mean")

# Other Clustering Methods

#Define a function to assign clusters based on percentiles
def assign_cluster(series):
    p20 = series.quantile(0.20)
    p40 = series.quantile(0.40)
    p60 = series.quantile(0.60)
    p80 = series.quantile(0.80)

    return series.apply(lambda x: 
        5 if x <= p20 else 
        4 if x <= p40 else 
        3 if x <= p60 else 
        2 if x <= p80 else 
        1
    )

data=data.fillna(0)
data_norm=data_norm.fillna(0)
data["capacity_norm"]=(data["PCGDP_2019_$/perons -2017PPP"]-data["PCGDP_2019_$/perons -2017PPP"].min())/(data["PCGDP_2019_$/perons -2017PPP"].max()-data["PCGDP_2019_$/perons -2017PPP"].min())
data["responsibility_norm"]=(data["CCO2-FF1_DFS_2019_GtCO2"]-data["CCO2-FF1_DFS_2019_GtCO2"].min())/(data["CCO2-FF1_DFS_2019_GtCO2"].max()-data["CCO2-FF1_DFS_2019_GtCO2"].min())
data["emission_norm"]=(data["PCO2-FFI_2019_tCO2/person"]-data["PCO2-FFI_2019_tCO2/person"].min())/(data["PCO2-FFI_2019_tCO2/person"].max()-data["PCO2-FFI_2019_tCO2/person"].min())
data["Res_Cap"]=0.5*data["capacity_norm"]+0.5*data["responsibility_norm"]
data_eq_weight=data_norm.iloc[:,0:14]
data_eq_weight_norm=(data_eq_weight-data_eq_weight.min())/(data_eq_weight.max()-data_eq_weight.min())
data_eq_weight_nonpcgdp=data_eq_weight_norm.iloc[:,1:14]
data_eq_weight_pcgdp=data_eq_weight_norm.iloc[:,0]
data["All_indicators"]=((0.05*data_eq_weight_nonpcgdp).sum(axis=1))+0.35*data_eq_weight_pcgdp
data["cluster_PCGDP"] = assign_cluster(data["PCGDP_2019_$/perons -2017PPP"])
data["cluster_PCO2FFI"] = assign_cluster(data["PCO2-FFI_2019_tCO2/person"])
data["cluster_CCO2FF1"] = assign_cluster(data["CCO2-FF1_DFS_2019_GtCO2"])
data["cluster_Res_Cap"]=assign_cluster(data["Res_Cap"])
data["cluster_Eq_weight"]=assign_cluster(data["All_indicators"])

# =============================================================================
# # Section 2: Scenerio Inputs
# =============================================================================

# GDP Inputs
# GDP growth rates for G1, G2, G3, G4, and G5 countries based on the scenerios
gdp_growth_rates={
    "GDP_Uhigh":{1:0.02,2:0.05,3:0.06,4:0.07,5:0.085},
    "GDP_UMod":{1:0.015,2:0.04,3:0.05,4:0.06,5:0.07},
    "GDP_ULow":{1:0.01,2:0.03,3:0.04,4:0.05,5:0.06},
    "GDP_DG1":{1:0.00,2:0.03,3:0.05,4:0.07,5:0.085},
    "GDP_DG2":{1:-0.01,2:0.03,3:0.05,4:0.07,5:0.085},
    "GDP_UMod_TS1":{},
    "GDP_UMod_TS2":{}
    }

gdp_no_degrowth={1:0.00,2:0.01,3:0.02,4:0.025,5:0.03}

# Energy Inputs
energy_threshold={"ECOV_GlAvg75":{1:75,2:75,3:75,4:75,5:75},
                     "ECOV_QPT60":{1:60,2:60,3:60,4:60,5:60},
                     "ECOV_QPT70":{1:70,2:70,3:70,4:70,5:70},
                     "ECOV_QPT80":{1:80,2:80,3:80,4:80,5:80},
                     "ECOV_QPT90":{1:90,2:90,3:90,4:90,5:90},
                     "ECOV_QPT95":{1:95,2:95,3:95,4:95,5:95},
                     "EDIV_High":{1:200,2:160,3:55,4:30,5:15},
                     "EDIV_Med":{1:160,2:120,3:75,4:40,5:25}}
convergence_year=["2030","2040","2050"]
#FEPE_ratio={"EE_1":{1:0.01,2:0.01,3:0.01,4:0.01,5:0.01},
#           "EE_1.5":{1:0.015,2:0.015,3:0.015,4:0.015,5:0.015},
#           "EE_2":{1:0.02,2:0.02,3:0.02,4:0.02,5:0.02},
#            "EE_2.5":{1:0.025,2:0.025,3:0.025,4:0.025,5:0.025},
#            "EE_3":{1:0.03,2:0.03,3:0.03,4:0.03,5:0.03}
#            }
#emission_sce = {
#    "EMIU_BY": {1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#    "EMIU_0.5": {1: -0.005, 2: -0.005, 3: -0.005, 4: -0.005, 5: -0.005},
#    "EMIU_1": {1: -0.01, 2: -0.01, 3: -0.01, 4: -0.01, 5: -0.01},
#    "EMIU_1.5": {1: -0.015, 2: -0.015, 3: -0.015, 4: -0.015, 5: -0.015},
#    "EMID_Low": {1: -0.01, 2: -0.01, 3: -0.005, 4: -0.005, 5: 0.00},
#    "EMID_Med": {1: -0.015, 2: -0.01, 3: -0.01, 4: -0.005, 5: 0.00},
#    "EMI_High": {1: -0.015, 2: -0.015, 3: -0.01, 4: -0.01, 5: -0.01}
#}

# Emissions Inputs
# Temperature Target
temperaturetarget = {
    "1.5_67%": 400,
    "1.5_50%": 500,
    "1.7_67%": 700,
    "1.7_50%": 850,
    "2_67%": 1150,
    "2_50%": 1350
}
variables = [
    "population",
    "Primary Energy Consumption",
    "Baseline CO2-FFI Emissions",
    "Baseline Emissions Intensity",
    "Baseline Net CO2",
    "CO2-LULUCF_Chosen Scenario",
    "Emissions Intensity of Primary Energy_Chosen Scenario",
    "GDP",
    "CO2_Net",
    "Per capita GDP",
    "Per capita Net CO2",
    "Per capita primary energy consumption",
    "Primary Energy Intensity of GDP",
    "Per Capita Net CO2 in Baseline",
]

def get_country_group_mapping(selected_method):
    if selected_method == "equal_weighting":
        cluster_col="cluster_Eq_weight"
    elif selected_method == "umap_kmeans":
        cluster_col= "cluster_umap"
    elif selected_method == "per_capita_GDP":
        cluster_col= "cluster_PCGDP"
    elif selected_method == "per_capita_emission":
        cluster_col = "cluster_PCO2FFI"
    elif selected_method == "historical_emissions":
        cluster_col = "cluster_CCO2FF1"
    elif selected_method == "CBDR&RC":
        cluster_col = "cluster_Res_Cap"

    mapping = {
        "Annex I": data[data["Annex or Non-Annex"] == "Annex I"]["Countries"].tolist(),
        "Non-Annex I": data[data["Annex or Non-Annex"] == "Non-Annex I"]["Countries"].tolist(),
        "World": data["Countries"].tolist()
    }

    for i in range(1, 6):
        mapping[f"Group {i}"] = data[data[cluster_col] == i]["Countries"].tolist()

    return mapping

# =============================================================================
# Section 3: Initialize the App
# =============================================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout
app.layout = html.Div([
    # Title
    html.H1("Model for Energy Equity and Climate Compatibility_Version.1 (MEECC_V.1)", style={"textAlign": "center"}),

    # Layout with two primary sections: Left for Inputs, Right for Graph
    html.Div([
        # 📌 Left Pane (Inputs)
        html.Div([
            # Tabs for Country Selection and Developmental Scenarios
            dcc.Tabs(id="main-tabs", value="clustering_method", children=[
                # 📌 NEW TAB 0: Clustering Method Selection
                dcc.Tab(label="Clustering Method", value="clustering_method", children=[
                    html.Div([
                        html.Label("Choose the clustering method:", style={"textAlign": "left", "fontWeight": "bold"}),
            
                        dcc.RadioItems(
                            id="clustering-method-radio",
                            options=[
                                {"label": "1. All_Dev", "value": "equal_weighting"},
                                {"label": "2. UMAP + k-means", "value": "umap_kmeans"},
                                {"label": "3. PCGDP", "value": "per_capita_GDP"},
                                {"label": "4. PCEmissions", "value": "per_capita_emission"},
                                {"label": "5. HistEmissions", "value": "historical_emissions"},
                                {"label": "6. Res_Cap", "value": "CBDR&RC"}                                
                            ],
                            value="umap_kmeans",  # Default value
                            labelStyle={"display": "block", "margin": "5px 0"}
                        ),
            
                        html.Div(id="clustering-method-output", style={"margin-top": "10px", "color": "#555"})
                    ], style={"padding": "20px"})
                ]),
                # 📌 TAB 1: Country Selection
                dcc.Tab(label="Countries", value="countries", children=[
                    html.Div([
                        html.Label("Please select individual countries or country groups from the list below:", style={"textAlign": "right"}),

                        # Container with Scrollable Checkbox List (4 checkboxes per row)
                        html.Div([
                            dcc.Checklist(
                                id="country-checkbox",
                                value=["World"],  # Default selection
                                inline=True,
                                style={"display": "grid", "grid-template-columns": "repeat(4, 1fr)", "gap": "5px"}
                            ),
                        ], style={"max-height": "300px", "overflow-y": "auto", "border": "1px solid #ccc", "padding": "10px"}),
                    ], style={"padding": "20px"})
                ]),

                # 📌 TAB 2: Developmental Scenarios
                dcc.Tab(label="Developmental Scenarios", value="developmental_scenarios", children=[
                    html.Div([
                        html.Label("Economic Development Scenarios:", style={"textAlign": "right"}),

                        html.Label("GDP Growth:", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="GDP-dropdown",
                            options=[{"label": key, "value": key} for key in gdp_growth_rates.keys()],
                            value="GDP_Uhigh"
                        ),

                        html.Label("Energy Thresholds:", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="Energy-dropdown", 
                            options=[{"label": energy, "value": energy} for energy in energy_threshold.keys()],
                            value="ECOV_GlAvg75"
                        ),

                        html.Label("Year of Convergence if the Convergence Scenario is Chosen:", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="convergence-dropdown",
                            options=[{"label": year, "value": year} for year in convergence_year],
                            value="2050"
                        ),

                        html.Label("CO2-LULUCF Emissions:", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="LULUCF",
                            options=[
                                {"label": "LULUCF_BY", "value": "LULUCF_BY"},
                                {"label": "LULUCF_PartShift", "value": "LULUCF_PartShift"},
                                {"label": "LULUCF_Shift", "value": "LULUCF_Shift"}
                            ],
                            value="LULUCF_BY"
                        ),

                        html.Label("Carbon Budget Allocation:", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="budget-dropdown",
                            options=[
                                {"label": "PCFS_GF", "value": "PCFS_GF"},
                                {"label": "PCFS_Hist", "value": "PCFS_Hist"},
                                {"label": "PCFS_Cap", "value": "PCFS_Cap"},
                                {"label": "PCFS_HistCap", "value": "PCFS_HistCap"},
                                {"label": "PCUS_CurrAnn", "value": "PCUS_CurrAnn"}
                            ],
                            value="PCFS_GF",
                        ),

                        html.Label("Temperature Target:", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="temperature-dropdown",
                            options=[{"label": temp, "value": temp} for temp in temperaturetarget.keys()],
                            value="1.5_67%"
                        )
                    ], style={"padding": "20px"})
                ])
            ])
        ], style={"width": "45%", "padding": "20px", "background-color": "#f8f8f8"}),

        # 📌 Right Pane (Graph Display)
        html.Div([
            html.Label("Select the Figure to Display:", style={"textAlign": "center"}),
            dcc.Dropdown(
                id="figure-dropdown",
                options=[{"label": var, "value": var} for var in variables],
                value="GDP"
            ),
            dcc.Graph(id="result-graph", style={"height": "600px"}),
            html.Div(id="country-limit-warning", style={"color": "red", "font-size": "14px", "margin-top": "10px"}),
        ], style={"width": "50%", "padding": "20px", "background-color": "#ffffff"})
    ], style={"display": "flex", "justify-content": "space-between"}),

    # 📌 Bottom Pane (Details and Data Table)
    html.Div([
        html.Div([
            html.H3("Selected Variable Details:"),
            dcc.Dropdown(
                id="variable-dropdown",
                options=[
                    {"label": "Cumulative Baseline Emissions (2020-2100)", "value": "Cumulative_Baseline_Emissions"},
                    {"label": "Cumulative Emissions (2020-2100)", "value": "Cumulative_Emissions"},
                    {"label": "Emissions Intensity of PE Annual Change", "value": "PE_Intensity_Change"}
                ],
                value="Cumulative_Baseline_Emissions"  # Default selection
            ),
            dcc.Graph(id="variable-bar-graph", style={"margin-top": "10px", "height": "400px"})
        ], style={"width": "45%", "padding": "20px", "background-color": "#f0f0f0"}),

        html.Div([
            html.H3("Data Table for the Selected Figure:"),
            dash_table.DataTable(id="data-table")
        ], style={"width": "50%", "padding": "20px", "background-color": "#f0f0f0"})
    ], style={"display": "flex", "justify-content": "space-between", "margin-top": "20px"})
], style={"font-family": "Arial, sans-serif", "background-color": "#fafafa", "padding": "20px"})

print(app.layout)

# =============================================================================
# # Section 4: Call Back
# =============================================================================
@app.callback(
    Output("country-checkbox", "options"),
    Input("clustering-method-radio", "value")
)
def update_country_options(selected_method):
    group_mapping = get_country_group_mapping(selected_method)
    group_options = [{"label": k, "value": k} for k in group_mapping.keys()]
    country_options = [{"label": c, "value": c} for c in data["Countries"].unique()]
    return group_options + country_options
  
@app.callback(
    [Output('result-graph', 'figure'), Output("data-table", "data"), 
     Output("data-table", "columns"), Output("variable-bar-graph", "figure"), 
     Output("country-limit-warning", "children"), Output("clustering-method-output", "children")],
    [Input('country-checkbox', 'value'),
     Input('GDP-dropdown', 'value'),
     Input("Energy-dropdown","value"),
     Input("convergence-dropdown","value"),
     Input("LULUCF","value"),
     Input("budget-dropdown","value"),
     Input("temperature-dropdown","value"),
     Input("figure-dropdown","value"),
     Input("variable-dropdown","value"),
     Input("clustering-method-radio", "value")
     ]
)
def update_graph(selected_countries, selected_gdpgrowth, selected_energythreshold,
                 selected_convergenceyear,
                 selected_LULUCFsce, selected_budget,
                 selected_temp_target, selected_figure, selected_variable, selected_method):
    
    if not selected_countries:
        # Return empty/default outputs when nothing is selected
        empty_fig = px.line(title="Please select at least one country or group.")
        empty_bar = px.bar(title="No data available.")
        return empty_fig, [], [], empty_bar, "", ""

    print(f"Selected Countries: {selected_countries}")
    print(f"Selected GDP Growth Scenario: {selected_gdpgrowth}")
    print(f"Selected Energy Threshold: {selected_energythreshold}")
    print(f"Selected Convergence Year: {selected_convergenceyear}")
    print(f"Selected LULUCF Scenario: {selected_LULUCFsce}")
    print(f"Selected Carbon Budget: {selected_budget}")
    print(f"Selected Temperature Target: {selected_temp_target}")
    print(f"Selected Figure Type: {selected_figure}")
    
    logging.info(f"Selected Countries: {selected_countries}")
    logging.info(f"Selected GDP Growth Scenario: {selected_gdpgrowth}")
    logging.info(f"Selected Energy Threshold: {selected_energythreshold}")
    logging.info(f"Selected Convergence Year: {selected_convergenceyear}")
    logging.info(f"Selected LULUCF Scenario: {selected_LULUCFsce}")
    logging.info(f"Selected Carbon Budget: {selected_budget}")
    logging.info(f"Selected Temperature Target: {selected_temp_target}")
    logging.info(f"Selected Figure Type: {selected_figure}")
    
    # Dynamically update country groups based on selected clustering method
    group_mapping = get_country_group_mapping(selected_method)

    # Initialize outputs
    infeasible_countries = []  # Track countries that can't reach NZY feasibly
    warning_message = "" 
    combined_results = []  # Store results for all selected countries/groups
    all_selected_countries = {}  # Dictionary to track country -> selected group(s)
    group_aggregations = {}  # Store separate aggregations for groups
    cumulative_emissions_bar_data = []
    cumulative_emissions_by_group = {}
    emission_intensity_change_bar = []

    # Step 1: Identify if selection includes groups or individual countries
    for selection in selected_countries:
        if selection in group_mapping:  # ✅ Use dynamic group mapping
            group_aggregations[selection] = []
            for country in group_mapping[selection]:
                if country in all_selected_countries:
                    all_selected_countries[country].append(selection)
                else:
                    all_selected_countries[country] = [selection]
        else:
            all_selected_countries[selection] = ["Individual Country"]

    
# Step 2: Process each selected country separately
    for country, groups in all_selected_countries.items():
        country_data = data[data["Countries"] == country]
        country_data=country_data.iloc[0]
        country_data=country_data.fillna(0)
        if country_data.empty:
            print(f"Warning: No data found for {country}. Skipping.")
        # Retrieve all the input parameters
        E_threshold=energy_threshold[selected_energythreshold]
        convergence_year = int(selected_convergenceyear)
        #efficiency_ratio=FEPE_ratio[selected_FE_PE]
        #emission_scenerio=emission_sce[selected_emissionsce]
        temp_goal=temperaturetarget[selected_temp_target]

        required_columns = ["POP_2019_No. ","PE_2019_EJ",
        "GDP_2019_$-2017 PPP", "PE_2019_EJ", "CO2-FFI_2019_GtCO2",
        "CO2-LULUCF_2019_GtCO2", "CCO2-FFI_1850-2019_GtCO2",
        "CO2-LULUCF_1850-2019_GtCO2"
        ]
        if not all(col in country_data for col in required_columns):
            raise KeyError("Missing required columns in country_data.")
            
        GDP_2019 = country_data["GDP_2019_$-2017 PPP"]
        print(f"GDP value for the {country} in 2019 is {GDP_2019}")
        energy_2019 = country_data["PPE_2019_GJ/p"]
        primary_energy_2019=country_data["PE_2019_EJ"]
        CO2_FFI_2019=country_data["CO2-FFI_2019_GtCO2"]
        CO2_LULUCF_2019=country_data["CO2-LULUCF_2019_GtCO2"]*10E8
        #Population data transformation
        pop_country=pop[pop["Countries"] == country]
        pop_country_long=pop_country.melt(id_vars=["Countries"],var_name="Years", value_name="Population")
        pop_country_long["Years"] = pop_country_long["Years"].astype(int)
        
        #retrieve values from dictionaries
        if selected_method == "equal_weighting":
            cluster=country_data["cluster_Eq_weight"]
        elif selected_method == "umap_kmeans":
            cluster = country_data["cluster_umap"]
        elif selected_method == "per_capita_GDP":
            cluster = country_data["cluster_PCGDP"]
        elif selected_method == "per_capita_emission":
            cluster = country_data["cluster_PCO2FFI"]
        elif selected_method == "historical_emissions":
            cluster = country_data["cluster_CCO2FF1"]
        elif selected_method == "CBDR&RC":
            cluster = country_data["cluster_Res_Cap"]
        print(f"You have selected {selected_method}")
        print(f"{country} belongs to Group:{cluster}")
        
        # creating TS1 and TS2 scnerio growth rates
        gdp_growth={
            "GDP_UMod":{1:0.015,2:0.04,3:0.05,4:0.06,5:0.07},
            "GDP_DG2":{1:-0.01,2:0.03,3:0.05,4:0.07,5:0.085}
                }
        gr_umod=gdp_growth["GDP_UMod"]
        gr_dg2=gdp_growth["GDP_DG2"]
        gdp_growth=pd.DataFrame()
        gdp_growth["Countries"]=data["Countries"]
        gdp_growth["GDP_2019_$-2017 PPP"]=data['GDP_2019_$-2017 PPP']
        gdp_growth["PCGDP_2019_$/perons -2017PPP"]=data['PCGDP_2019_$/perons -2017PPP']
        gdp_growth["POP_2019_No. "]=data["POP_2019_No. "]
        gdp_growth["POP_2050_No. "]=pop["2050"]
        gdp_growth['growth_rate_umod']=data["cluster_umap"].map(gr_umod)
        gdp_growth['growth_rate_dg2']=data["cluster_umap"].map(gr_dg2)
        year_diff=2050-2019
        gdp_growth['PCGDP_2050_UMod_TS1']=(gdp_growth["GDP_2019_$-2017 PPP"]*((1+gdp_growth['growth_rate_umod'])**year_diff))/gdp_growth["POP_2050_No. "]
        gdp_growth['PCGDP_2050_DG2_TS2']=(gdp_growth["GDP_2019_$-2017 PPP"]*((1+gdp_growth['growth_rate_dg2'])**year_diff))/gdp_growth["POP_2050_No. "]

        # Calculate new growth rate for countries with PCGDP_2050 < 28000
        def calculate_new_growth_rate(current_gdp_2019,population, target_pcgdp_2050=28000, year_diff=31):
            return (((target_pcgdp_2050*population )/ current_gdp_2019) ** (1 / year_diff)) - 1

        # Determine new growth rates where PCGDP_2050 < 28000
        gdp_growth['growth_rate_UMod_TS1'] = gdp_growth.apply(
            lambda row: calculate_new_growth_rate(row["GDP_2019_$-2017 PPP"],row["POP_2050_No. "]) 
            if row["PCGDP_2050_UMod_TS1"] < 28000 else row["growth_rate_umod"], 
            axis=1
        )

        gdp_growth['growth_rate_DG2_TS2'] = gdp_growth.apply(
            lambda row: calculate_new_growth_rate(row["GDP_2019_$-2017 PPP"],row["POP_2050_No. "]) 
            if row["PCGDP_2050_DG2_TS2"] < 28000 else row["growth_rate_dg2"], 
            axis=1
        )
        # Update GDP_UMod_TS1 and GDP_UMod_TS2 based on dynamic rates from gdp_growth DataFrame
        gdp_growth_rates["GDP_UMod_TS1"] = dict(zip(gdp_growth["Countries"], gdp_growth["growth_rate_UMod_TS1"]))
        gdp_growth_rates["GDP_UMod_TS2"] = dict(zip(gdp_growth["Countries"], gdp_growth["growth_rate_DG2_TS2"]))
        
        #selecting the gdp growth rates
        growth_rate = gdp_growth_rates[selected_gdpgrowth]
                
        if selected_gdpgrowth not in gdp_growth_rates:
            raise ValueError(f"Invalid GDP growth rate: {selected_gdpgrowth}")
        
        if selected_gdpgrowth == "GDP_UMod_TS1" or selected_gdpgrowth == "GDP_UMod_TS2":
            rate=growth_rate[country]
        else:
            rate = growth_rate[cluster] # selected GDP growth rate
        print(f"selected GDP growth rate of {country} till 2050 is {rate}")
        no_degrowth_rate=gdp_no_degrowth[cluster]  #GDP no degrowth rate
        print(f"selected GDP growth rate of {country} between 2050 and 2100 is {no_degrowth_rate}")
        
        threshold_value = E_threshold[cluster]  # Energy threshold for the cluster
        print(f"Energy threshold value for {country} is chosen as {threshold_value}")  
        
        #ratio=efficiency_ratio[cluster] #FE PE ratio for the cluster
        #print(f"Ratio of Final and Primary Energy in 2019 for {country} is assummed as {ratio}")
        #emission_rate=emission_scenerio[cluster] # emission density growth rate till 2050
        #print(f"selected rate of change in fossil CO2 emissions for {country} is {emission_rate}")
        
        #GDP projection
        years_2050 = range(2020, 2051)
        gdp_projected_values_2050 = [GDP_2019 * (1 + rate) ** (year - 2019) for year in years_2050]
        GDP_2050=gdp_projected_values_2050[-1]
        years_2100=range(2051,2101)
        gdp_projected_values_2100=[GDP_2050*(1+no_degrowth_rate)**(year-2050) for year in years_2100]
        # Combine projections
        combined_years = list(years_2050) + list(years_2100)
        combined_gdp_values = gdp_projected_values_2050 + gdp_projected_values_2100
        #GDP per capita
        gdp_project_growth_df = pd.DataFrame({"Countries":country, "Years": combined_years, "Projected_GDP": combined_gdp_values})
        pop_country_long["Years"]=pop_country_long["Years"].astype(int)
        gdp_project_growth_df["Years"]=gdp_project_growth_df["Years"].astype(int)
        gdp_per_capita_df=pd.merge(gdp_project_growth_df,pop_country_long,how="inner",on=["Countries","Years"])
        gdp_per_capita_df["GDP_per_capita"]=gdp_per_capita_df["Projected_GDP"]/gdp_per_capita_df["Population"]
        
        #primary energy per capita projection
        current_year = 2019
        years_to_convergence = int(convergence_year) - current_year
        growth_rate = (threshold_value / energy_2019) ** (1 / years_to_convergence) - 1
        print(f"Primary energy per capita growth rate till the convergence year is {growth_rate}")
        growth_rate_2100=(70/threshold_value)**(1/(2100-int(convergence_year)))-1
        print(f"Primary energy per capita growth rate between convergence year and 2100 is {growth_rate_2100}")
        years_con = range(2020, int(convergence_year)+1)
        energy_projected_values_con = [energy_2019 * (1 + growth_rate) ** (year - current_year) for year in years_con]
        years_2100_con=range(int(convergence_year)+1, 2101)
        energy_cov=energy_projected_values_con[-1]
        energy_projected_values_2100=[energy_cov * (1 + growth_rate_2100) ** (year - convergence_year) for year in years_2100_con]
        #combination
        combined_years=list(years_con)+list(years_2100_con)
        combined_energy=energy_projected_values_con+energy_projected_values_2100
    
        #Primary energy
        energy_project_df = pd.DataFrame({
            "Countries":country,
            "Years": combined_years,
            "Projected_Primary_Energy_per_Capita": combined_energy
        })
        energy_project_df["Years"]=energy_project_df["Years"].astype(int)
        energy_project_df=pd.merge(energy_project_df,pop_country_long,how="inner",on=["Countries","Years"])
        energy_project_df=pd.merge(energy_project_df,gdp_project_growth_df, how="inner",on=["Countries","Years"])
        energy_project_df["Primary_Energy"]=energy_project_df["Projected_Primary_Energy_per_Capita"]*energy_project_df["Population"]
        
        #Primary energy density wrt GDP
        energy_project_df["primary_energy_density"]=energy_project_df["Primary_Energy"]/energy_project_df["Projected_GDP"]
        
        #Final Energy calculation
        #FE_PE_standard={1:0.85,2:0.8,3:0.75,4:0.7,5:0.65}
        #threshold=0.93
        #FE_PE_2019=FE_PE_standard[cluster]
        #print(f"FE_PE_ratio for {country} is {FE_PE_2019}")
        #full_years=range(2019,2101)
      
        # Initialize an empty list to store the series
        #FE_PE_series = []
        # Loop through the years to compute the series
        #for year in full_years:
        #    if not FE_PE_series:  # First year (initialize with FE_PE_2019)
        #        current_value = FE_PE_2019
        #    else:
        #        # Compute the value for the current year
        #        current_value = FE_PE_series[-1] * (1 + ratio)
        #        # Check if the threshold has been reached or crossed
        #        if current_value >= threshold:
        #            current_value = threshold  # Freeze the value at the threshold
        #    FE_PE_series.append(current_value)
    
        #FE_PE_df=pd.DataFrame({
        #    "Countries":country,
        #    "Years":full_years,
        #    "Projected_FE_PE_Ratio":FE_PE_series})
        #energy_project_df=pd.merge(energy_project_df,FE_PE_df,how="inner",on=["Countries","Years"])
        #energy_project_df["Final_energy"]=energy_project_df["Primary_Energy"]*energy_project_df["Projected_FE_PE_Ratio"]
        
        #Final energy per capita
        #energy_project_df["Final_energy_per_capita"]=energy_project_df["Final_energy"]/energy_project_df["Population"]
        #Final energy density
        #energy_project_df["Final_energy_density"]=energy_project_df["Final_energy"]/energy_project_df["Projected_GDP"]
        
        #CO2 FFI emissions
        Emission_intensity_energy_2019=CO2_FFI_2019/primary_energy_2019
        print(f"Emission Intensity of energy for {country} in 2019 was {Emission_intensity_energy_2019}")
        years=range(2020,2101)
        EI_projection_2100 = [Emission_intensity_energy_2019* (1+0) ** (year - 2019) for year in years]
        Emission_intensity_df = pd.DataFrame({"Countries":country, "Years": years, "Projected_Emission_Intensity": EI_projection_2100})
        Emission_intensity_df["Years"]=Emission_intensity_df["Years"].astype(int)
        energy_project_df=pd.merge(energy_project_df,Emission_intensity_df,how="inner",on=["Countries","Years"])
        energy_project_df["CO2_FFI_Baseline"]=energy_project_df["Projected_Emission_Intensity"]*energy_project_df["Primary_Energy"]
    
        #CO2_LULUCF_emissions scenerios
        if selected_LULUCFsce == "LULUCF_BY":
            CO2_LULUCF_2050 = CO2_LULUCF_2019
            CO2_LULUCF_2100 = CO2_LULUCF_2019
        elif selected_LULUCFsce == "LULUCF_PartShift":
            if CO2_LULUCF_2019 < 0:
                CO2_LULUCF_2050 = CO2_LULUCF_2019
                CO2_LULUCF_2100 = CO2_LULUCF_2019
            else:
                CO2_LULUCF_2050 = 0
                CO2_LULUCF_2100 = 0
        elif selected_LULUCFsce == "LULUCF_Shift":
            if CO2_LULUCF_2019 < 0:
               CO2_LULUCF_2050 = CO2_LULUCF_2019
               CO2_LULUCF_2100 = CO2_LULUCF_2019
            else:
               CO2_LULUCF_2050 = 0
               CO2_LULUCF_2100 = -0.005*10E9
           
        LULUCF_rate_2050=((CO2_LULUCF_2050 - CO2_LULUCF_2019)/(2050-2019))
        LULUCF_rate_2100=((CO2_LULUCF_2100 - CO2_LULUCF_2050)/(2100-2050))
        print(f"The rate of change of LULUCF emissions in {country} between 2020 and 2050 is {LULUCF_rate_2050}")
        print(f"The rate of change of LULUCF emissions in {country} between 2051 and 2100 is {LULUCF_rate_2100}")
        
        years_2050=range(2020,2051)
        LULUCF_projection_2050 = [CO2_LULUCF_2019+LULUCF_rate_2050 * (year - 2019) for year in years_2050]
        years_2100=range(2051,2101)
        LULUCF_projection_2100=[CO2_LULUCF_2050+LULUCF_rate_2100*(year-2050) for year in years_2100]
        combined_years = list(years_2050) + list(years_2100)
        combined_LULUCF_values = (LULUCF_projection_2050 + LULUCF_projection_2100)
        LULUCF_emissions_df = pd.DataFrame({"Countries":country, "Years": combined_years, "Projected_LULUCF_emissions": combined_LULUCF_values})
        LULUCF_emissions_df["Years"]=LULUCF_emissions_df["Years"].astype(int)
        energy_project_df=pd.merge(energy_project_df,LULUCF_emissions_df,how="inner",on=["Countries","Years"])
        energy_project_df["Net_CO2_Baseline"]=energy_project_df["CO2_FFI_Baseline"]+energy_project_df["Projected_LULUCF_emissions"]
        energy_project_df["percapita_Net_CO2_Baseline"]=energy_project_df["Net_CO2_Baseline"]/energy_project_df["Population"]
      
        # Carbon Budget and temp target scenerios
        data["Hist_Net_CO2"]=data["CCO2-FFI_1850-2019_GtCO2"]+data["CO2-LULUCF_1850-2019_GtCO2"]
        total_emission_1850_2019=data["Hist_Net_CO2"].sum()
        data["actual_share"]=data["Hist_Net_CO2"]/total_emission_1850_2019
        total_population=data["POP_2019_No. "].sum()
        total_gdp_per_capita=data["PCGDP_2019_$/perons -2017PPP"].sum()
        data["Pop_share"]=data["POP_2019_No. "]/total_population
        data["F/N_share"]=data["Pop_share"]/data["actual_share"]
        data["Population Share Weighted Per Capita GDP"]=(total_gdp_per_capita-data["PCGDP_2019_$/perons -2017PPP"])*data["Pop_share"]
        total_population_share_weighted_pcgdp=data["Population Share Weighted Per Capita GDP"].sum()
        data["Fair Share Weighted by Per Capita GDP"]=data["Population Share Weighted Per Capita GDP"]/total_population_share_weighted_pcgdp
        #Hist
        data["per_capita_hist"]=data["Hist_Net_CO2"]/data["POP_2019_No. "]
        total_per_capita_hist=data["per_capita_hist"].sum()
        data["Population Share Weighted Per Capita historical emissions"]=(total_per_capita_hist-data["per_capita_hist"])*data["Pop_share"]
        total_Population_Share_Weighted_Per_Capita_historical_emissions=data["Population Share Weighted Per Capita historical emissions"].sum()
        data["Fair Share Weighted by Per Capita Historical emissions"]=data["Population Share Weighted Per Capita historical emissions"]/total_Population_Share_Weighted_Per_Capita_historical_emissions
        total_emissions_2019=(data["CO2-FFI_2019_GtCO2"]+data["CO2-LULUCF_2019_GtCO2"]).sum()
        data["Unfair Share"]=(data["CO2-FFI_2019_GtCO2"]+data["CO2-LULUCF_2019_GtCO2"])/total_emissions_2019
        carbon_data=data
        carbon_data["PCFS_GF"]=carbon_data["Pop_share"]*temp_goal
        carbon_data["PCFS_Hist"]=carbon_data["Fair Share Weighted by Per Capita Historical emissions"]*temp_goal
        carbon_data["PCFS_Cap"]=carbon_data["Fair Share Weighted by Per Capita GDP"]*temp_goal
        carbon_data["PCFS_HistCap"]=(0.5*carbon_data["Fair Share Weighted by Per Capita Historical emissions"]+0.5*carbon_data["Fair Share Weighted by Per Capita GDP"])*temp_goal
        carbon_data["PCUS_CurrAnn"]=carbon_data["Unfair Share"]*temp_goal
       
        # Determine RCB based on selected_budget
        if selected_budget == "PCFS_GF":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_GF"].values[0]
            print(f"Remaining Carbon Budget (RCB) for {country} in the chosen carbon budget allocation is {RCB}")
        elif selected_budget == "PCFS_Hist":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_Hist"].values[0]
            print(f"Remaining Carbon Budget (RCB) for {country} in the chosen carbon budget allocation is {RCB}")
        elif selected_budget == "PCFS_Cap":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_Cap"].values[0]
            print(f"Remaining Carbon Budget (RCB) for {country} in the chosen carbon budget allocation is {RCB}")
        elif selected_budget == "PCFS_HistCap":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_HistCap"].values[0]
            print(f"Remaining Carbon Budget (RCB) for {country} in the chosen carbon budget allocation is {RCB}")
        elif selected_budget == "PCUS_CurrAnn":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCUS_CurrAnn"].values[0]
            print(f"Remaining Carbon Budget (RCB) for {country} in the chosen carbon budget allocation is {RCB}")
        else:
            print("Error: Invalid budget type selected.")
            RCB = None
        print(f"selected_remaining_carbon_budget for the selected carbon budget allocation {selected_budget} for {country} is {RCB}")
        Annual_reduction_rate=None
        # Peak_year_endogeneous calculation
        F_A_share=data.loc[data["Countries"]==country,"F/N_share"].values[0]
        if F_A_share < 1:
            peak_year = 2020
        elif (F_A_share >= 1) and (F_A_share < 2):
            peak_year = 2025
        elif (F_A_share >= 2) and (F_A_share < 3):
            peak_year = 2030
        elif F_A_share >= 3:
            peak_year = 2035

        # Now recalculate NZY
        peak_year_df=energy_project_df.copy()
        Peakyear_emissions = peak_year_df.loc[peak_year_df["Years"] == peak_year, "Net_CO2_Baseline"].values[0]
        cum_peakyear_emissions = peak_year_df.loc[
            (peak_year_df["Years"] >= 2020) & (peak_year_df["Years"] <= peak_year),
            "Net_CO2_Baseline"
        ].sum()
        NZY = peak_year + ((RCB - cum_peakyear_emissions * 1e-9) * (2 / (Peakyear_emissions * 1e-9)))
        NZY = round(NZY)
        
        # Initialize column once to ensure it exists
        peak_year_df["Emissions_with_mitigation"] = 0.0
        
        #Checking whether the difference between net zero year and peak year is atleast 10 years
        if (NZY - peak_year) <= 10:
            Annual_reduction_rate = None
            NZY=None
            infeasible_countries.append(country)
        else:
            Annual_reduction_rate = Peakyear_emissions / (NZY - peak_year)
        
        if NZY is not None and Annual_reduction_rate is not None:
            # Loop through the years to calculate emissions
            for year in peak_year_df["Years"]:
                if year <= peak_year:
                    # Use the emission value in the particular year from the "Net CO2 Baseline"
                    peak_year_df.loc[peak_year_df["Years"] == year, "Emissions_with_mitigation"] = peak_year_df.loc[peak_year_df["Years"] == year, "Net_CO2_Baseline"].values[0]
                else:
                    # Calculate annual emission reduction
                    previous_emission = peak_year_df.loc[peak_year_df["Years"] == year - 1, "Emissions_with_mitigation"].values[0]
                    if previous_emission > Annual_reduction_rate:
                        reduced_emission = previous_emission - Annual_reduction_rate
                        peak_year_df.loc[peak_year_df["Years"] == year, "Emissions_with_mitigation"] = reduced_emission
                    else:
                        # If emissions fall below the annual reduction rate, set to zero
                        peak_year_df.loc[peak_year_df["Years"] == year, "Emissions_with_mitigation"] = 0
        else:
          peak_year_df["Emissions_with_mitigation"]=0.0
        #calculating per capita gdp
        peak_year_df["GDP_per_capita"]=peak_year_df["Projected_GDP"]/peak_year_df["Population"]
        #calculating per capita emission_with mitigation
        if NZY is not None:
            peak_year_df["per_capita_emission_with_mitigation"]=peak_year_df["Emissions_with_mitigation"]/peak_year_df["Population"]
            #calculating emission with mitigation intensity wrt primary energy
            peak_year_df["emission_intensity_primaryenergy_with_mitigation"]=peak_year_df["Emissions_with_mitigation"]/peak_year_df["Primary_Energy"]
        else:
            peak_year_df["per_capita_emission_with_mitigation"]=0.0
            peak_year_df["emission_intensity_primaryenergy_with_mitigation"]=0.0
                
        #calculating the cumulative emissions between 2020 and 2100 with and without (Baseline) mitigation
        cumulative_baseline_emissions=peak_year_df["Net_CO2_Baseline"].sum()
        cumulative_withmitigation_emissions=peak_year_df["Emissions_with_mitigation"].sum()
        print(f"Cumulative Net Baseline Emission between 2020 and {NZY} is {cumulative_baseline_emissions}")
        print(f"Cumulative Net Emission with mitigation between 2020 and {NZY} is {cumulative_withmitigation_emissions}")
        
        #default values
        rate_2020_2030=None
        rate_2030_2040=None
        rate_2040_2050=None     
             
        #calculating the annual change in the emissions intensity of primary energy
        if cumulative_withmitigation_emissions is not None:
            EI_2020=peak_year_df.loc[peak_year_df["Years"]==2020,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            EI_2030=peak_year_df.loc[peak_year_df["Years"]==2030,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            EI_2040=peak_year_df.loc[peak_year_df["Years"]==2040,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            EI_2050=peak_year_df.loc[peak_year_df["Years"]==2050,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            print(f"EI_2020 is {EI_2020}")
            print(f"EI_2030 is {EI_2030}")
            print(f"EI_2040 is {EI_2040}")
            print(f"EI_2050 is {EI_2050}")
            if not pd.isna(EI_2020) and not pd.isna(EI_2030) and EI_2020 != 0:
                rate_2020_2030 = round(((EI_2030 - EI_2020) * 10) / EI_2020, 2)
                emission_intensity_change_bar.append({"Entity": country, "Period": "2020–2030", "Change (%)": rate_2020_2030})
        
            if not pd.isna(EI_2030) and not pd.isna(EI_2040) and EI_2030 != 0:
                rate_2030_2040 = round(((EI_2040 - EI_2030) * 10) / EI_2030, 2)
                emission_intensity_change_bar.append({"Entity": country, "Period": "2030–2040", "Change (%)": rate_2030_2040})
        
            if not pd.isna(EI_2040) and not pd.isna(EI_2050) and EI_2040 != 0:
                rate_2040_2050 = round(((EI_2050 - EI_2040) * 10) / EI_2040, 2)
                emission_intensity_change_bar.append({"Entity": country, "Period": "2040–2050", "Change (%)": rate_2040_2050})
        else:
            print("Emissions_with mitigation projection is infeasible for the chosen scenerio")
        country_results = peak_year_df.copy()
        # Assign "Individual Country" if no predefined group is assigned
        country_results["Selected_Group"] = country if groups == ["Individual Country"] else " & ".join(groups)     
        print(f"Appending data for {country} (infeasible: {country in infeasible_countries})")
        # Store results
        combined_results.append(country_results)

        # If the country belongs to a group selection, also store it for aggregation
        for group in groups:
            if group in group_aggregations:
                group_aggregations[group].append(country_results)

        # ✅ Collect cumulative emissions for bar chart
        if groups == ["Individual Country"]:
            # Store individual country data
            cumulative_emissions_bar_data.append({
                "Name": country,
                "Type": "Country",
                "Cumulative_Baseline_Emissions": cumulative_baseline_emissions,
                "Cumulative_Mitigation_Emissions": cumulative_withmitigation_emissions if cumulative_withmitigation_emissions is not None else 0
            })
        else:
            # Accumulate emissions per group
            for group in groups:
                if group not in cumulative_emissions_by_group:
                    cumulative_emissions_by_group[group] = {"baseline": 0, "mitigation": 0}
                cumulative_emissions_by_group[group]["baseline"] += cumulative_baseline_emissions
                if cumulative_withmitigation_emissions is not None:
                    cumulative_emissions_by_group[group]["mitigation"] += cumulative_withmitigation_emissions

    # 📌 Combine results after filtering
    if combined_results:
        final_df = pd.concat(combined_results)
        final_df = final_df.fillna(0)
    else:
        final_df = pd.DataFrame()
    
    for group_name, values in cumulative_emissions_by_group.items():
     cumulative_emissions_bar_data.append({
         "Name": group_name,
         "Type": "Group",
         "Cumulative_Baseline_Emissions": values["baseline"],
         "Cumulative_Mitigation_Emissions": values["mitigation"]
     })

    cumulative_emissions_bar_df = pd.DataFrame(cumulative_emissions_bar_data)
    
    # ✅ Filter out infeasible countries from aggregation
    aggregated_group_results = []
    group_mapping = get_country_group_mapping(selected_method)
    selected_groups = [grp for grp in selected_countries if grp in group_mapping]
    selected_individual_countries = [cnt for cnt in selected_countries if cnt not in group_mapping]
    
    # 🚨 Generate warning message
    warning_parts = []
    
    # Map group → list of its infeasible countries
    infeasible_by_group = {grp: [] for grp in selected_groups}
    infeasible_individuals = []
    
    for country in infeasible_countries:
        assigned = False
        for grp in selected_groups:
            if country in group_mapping[grp]:
                infeasible_by_group[grp].append(country)
                assigned = True
        if not assigned and country in selected_individual_countries:
            infeasible_individuals.append(country)
    
    # For group selections
    for grp, countries in infeasible_by_group.items():
        if countries:
            warning_parts.append(f"No solution for {len(countries)} countries in {grp}")
    
    # For individual selections
    if infeasible_individuals:
        names = ", ".join(infeasible_individuals)
        warning_parts.append(f"No solution for: {names}")
    
    # Final message
    warning_message = "; ".join(warning_parts) if warning_parts else ""
    
    # Show warning only for certain figure types
    warn_figures = [
        "CO2_Net", 
        "Emissions Intensity of Primary Energy_Chosen Scenario", 
        "Primary Energy Intensity of GDP",
        "Per capita Net CO2"
    ]
    
    if selected_figure not in warn_figures:
        # Clear warning message for other figures
        warning_message = ""
        # Also clear infeasible_countries so they are included in the graph
        infeasible_countries = []

    custom_agg_figures = {
        "Per capita GDP": ("Projected_GDP", "Population", "GDP_per_capita"),
        "Per capita Net CO2": ("Emissions_with_mitigation", "Population", "per_capita_emission_with_mitigation"),
        "Per capita primary energy consumption": ("Primary_Energy", "Population", "Projected_Primary_Energy_per_Capita"),
        "Primary Energy Intensity of GDP": ("Primary_Energy", "Projected_GDP", "primary_energy_density"),
        "Per Capita Net CO2 in Baseline": ("Net_CO2_Baseline", "Population", "percapita_Net_CO2_Baseline"),
        "Emissions Intensity of Primary Energy_Chosen Scenario":("Emissions_with_mitigation","Primary_Energy","emission_intensity_primaryenergy_with_mitigation")
    }
    
    for group, country_data_list in group_aggregations.items():
        filtered_list = [df for df in country_data_list if df["Countries"].iloc[0] not in infeasible_countries]
        if filtered_list:
            group_concat = pd.concat(filtered_list)
    
            if selected_figure in custom_agg_figures:
                group_df = group_concat.groupby("Years").sum().reset_index()

                # 🔁 Compute all derived indicators in custom_agg_figures
                for _, (numerator_col, denominator_col, result_col) in custom_agg_figures.items():
                    if numerator_col in group_df.columns and denominator_col in group_df.columns:
                        group_df[result_col] = group_df[numerator_col] / group_df[denominator_col]
           
            else:
                group_df = group_concat.groupby("Years").sum().reset_index()
         
            group_df["Countries"] = group
            group_df["Selected_Group"] = group
            aggregated_group_results.append(group_df)

    # Step 5: Merge aggregated groups into final dataframe
    if aggregated_group_results:
        aggregated_df = pd.concat(aggregated_group_results, ignore_index=True)  # Grouped aggregations
        final_df = pd.concat([final_df, aggregated_df], ignore_index=True)  # Merge all data
    
    # 🔹 Ensure that only the correct data is shown:
    if selected_groups:
        # If any groups are selected, **only show the group(s)** and remove the individual countries
        final_df = final_df[final_df["Countries"].isin(selected_groups)]
    elif selected_individual_countries:
        # If only individual countries are selected, show them
        final_df = final_df[final_df["Countries"].isin(selected_individual_countries)]
    
    # 🔹 Ensure pivot column is set correctly
    pivot_column = "Selected_Group" if "Selected_Group" in final_df.columns and selected_groups else "Countries"
    
    if pivot_column == "Countries":
        final_df = final_df.drop_duplicates(subset=["Years", pivot_column])

    result_col = "emission_intensity_primaryenergy_with_mitigation"

    if result_col in final_df.columns:
        for entity in final_df[pivot_column].unique():
            subset = final_df[final_df[pivot_column] == entity]
            try:
                EI_2020 = subset.loc[subset["Years"] == 2020, result_col].values[0]
                EI_2030 = subset.loc[subset["Years"] == 2030, result_col].values[0]
                EI_2040 = subset.loc[subset["Years"] == 2040, result_col].values[0]
                EI_2050 = subset.loc[subset["Years"] == 2050, result_col].values[0]
    
                if EI_2020 != 0:
                    rate_2020_2030 = round(((EI_2030 - EI_2020) * 10) / EI_2020, 2)
                    emission_intensity_change_bar.append({"Entity": entity, "Period": "2020–2030", "Change (%)": rate_2020_2030})
                if EI_2030 != 0:
                    rate_2030_2040 = round(((EI_2040 - EI_2030) * 10) / EI_2030, 2)
                    emission_intensity_change_bar.append({"Entity": entity, "Period": "2030–2040", "Change (%)": rate_2030_2040})
                if EI_2040 != 0:
                    rate_2040_2050 = round(((EI_2050 - EI_2040) * 10) / EI_2040, 2)
                    emission_intensity_change_bar.append({"Entity": entity, "Period": "2040–2050", "Change (%)": rate_2040_2050})
            except Exception:
                continue

    emission_change_df = pd.DataFrame(emission_intensity_change_bar)
    print(emission_change_df)

    # Debugging Prints (Remove in final version if unnecessary)
    print("Final DF After Aggregation:")
    print(final_df.head(8))
    print("Columns in final_df just before plotting:", final_df.columns.tolist())
    print("Unique Countries in Final DF:", final_df["Countries"].unique() if "Countries" in final_df else "MISSING")
    print("Unique Selected Groups in Final DF:", final_df["Selected_Group"].unique() if "Selected_Group" in final_df else "MISSING")

# =============================================================================
# Section 5: Results visualisation
# =============================================================================
    # Placeholder for figure and table data
    fig = None
    table_data = []
    table_columns = []
    
    # 🔹 Define Figure Mapping for Efficiency
    figure_mapping = {
        "population": {"y": "Population", "title": "Population Growth Projection", "unit": "Population (Millions)", "scale": 1e6},
        "Primary Energy Consumption": {"y": "Primary_Energy", "title": "Primary Energy Consumption Projection", "unit": "Primary Energy (EJ)", "scale": 1e9},
        "Baseline CO2-FFI Emissions": {"y": "CO2_FFI_Baseline", "title": "Baseline CO2-FFI Emissions Projection", "unit": "CO2-FFI Value (GtCO2)","scale":1e9},
        "Baseline Emissions Intensity": {"y": "Projected_Emission_Intensity", "title": "Baseline Emissions Intensity Projection", "unit": "Emissions Intensity (GtCO2/EJ)"},
        "Baseline Net CO2": {"y": "Net_CO2_Baseline", "title": "Baseline Net CO2 Projection", "unit": "Net CO2 (GtCO2)","scale":1e9},
        "CO2-LULUCF_Chosen Scenario": {"y": "Projected_LULUCF_emissions", "title": "CO2-LULUCF Projection", "unit": "CO2-LULUCF (GtCO2)","scale":1e9},
        "Emissions Intensity of Primary Energy_Chosen Scenario": {"y": "emission_intensity_primaryenergy_with_mitigation", "title": "Emissions Intensity of Primary Energy Projection", "unit": "Primary Energy Intensity (GtCO2/EJ)"},
        "GDP": {"y": "Projected_GDP", "title": "GDP Projection", "unit": "GDP million$-2017 PPP", "scale":1e6},
        "CO2_Net": {"y": "Emissions_with_mitigation", "title": "Net CO2 Emissions Projection", "unit": "Net CO2 (GtCO2)","scale":1e9},
        "Per capita GDP": {"y": "GDP_per_capita", "title": "Per Capita GDP Projection", "unit": "Per Capita GDP ($ 2017 PPP)"},
        "Per capita Net CO2": {"y": "per_capita_emission_with_mitigation", "title": "Per Capita Net CO2 Emissions", "unit": "Per Capita CO2 (tCO2/p)"},
        "Per capita primary energy consumption": {"y": "Projected_Primary_Energy_per_Capita", "title": "Per Capita Primary Energy Consumption", "unit": "Per Capita Primary Energy (GJ/p)"},
        "Primary Energy Intensity of GDP": {"y": "primary_energy_density", "title": "Primary Energy Intensity of GDP", "unit": "Energy Intensity (MJ/$)","scale":1e-3},
        "Per Capita Net CO2 in Baseline": {"y": "percapita_Net_CO2_Baseline", "title": "Per Capita Net CO2 in Baseline", "unit": "Baseline per capita Net CO2 (tCO2/p)"}
        }

    
    # 🔹 Ensure selection exists
    if selected_figure in figure_mapping:
        y_column = figure_mapping[selected_figure]["y"]
        title = figure_mapping[selected_figure]["title"]
        unit = figure_mapping[selected_figure]["unit"]
        scale = figure_mapping[selected_figure].get("scale", 1)  # Default scale = 1 if not specified
    
        # Apply scaling if required
        final_df[y_column] = final_df[y_column] / scale
    
        # 🔹 Create Figure
        fig = px.line(
            final_df,
            x="Years",
            y=y_column,
            color=pivot_column,
            title=title,
            labels={y_column: unit, pivot_column: "Country / Group"},
            hover_data={"Years": True, y_column: ":.2f", pivot_column: True}
            )
    
        # 🔹 Prepare Data Table
        dummy_df = pd.DataFrame({
            "Years": final_df["Years"],
            pivot_column: final_df[pivot_column],
            y_column: final_df[y_column]
        })
    
        # 🔹 Handle duplicates by aggregating first (sum all duplicate entries)
        dummy_df_grouped = dummy_df.groupby(["Years", pivot_column], as_index=False).sum()
    
        # 🔹 Now pivot safely
        pivot_df = dummy_df_grouped.pivot(index="Years", columns=pivot_column, values=y_column)
    
        # 🔹 Convert Pivot Table to Dict
        df = pivot_df.reset_index()
        df=df.round(2)
        table_data = df.to_dict("records")
        table_columns = [{"name": "Years", "id": "Years"}] + [{"name": col, "id": col} for col in pivot_df.columns]
    
    else:
        fig = px.line(title="No Figure Selected")

    # Create bar chart based on selected variable
    # 🔹 Create the Bar Graph based on the variable selected
    if selected_variable == "Cumulative_Baseline_Emissions":
        bar_df = pd.DataFrame(cumulative_emissions_bar_data)
        variable_bar_figure = px.bar(
            bar_df, x="Name", y="Cumulative_Baseline_Emissions",
            title="Cumulative Baseline Emissions (2020-2100)",
            labels={"Name": "Country / Group", "Cumulative_Baseline_Emissions": "tCO₂"}
        )

    elif selected_variable == "Cumulative_Emissions":
        bar_df = pd.DataFrame(cumulative_emissions_bar_data)
        variable_bar_figure = px.bar(
            bar_df, x="Name", y="Cumulative_Mitigation_Emissions",
            title="Cumulative Mitigation Emissions (2020-2100)",
            labels={"Name": "Country / Group", "Cumulative_Mitigation_Emissions": "tCO₂"}
        )
    elif selected_variable == "PE_Intensity_Change":
        bar_df = pd.DataFrame(emission_intensity_change_bar)
    
        if not bar_df.empty and "Entity" in bar_df.columns:
            bar_df = bar_df.drop_duplicates(subset=["Entity","Period"])
            if selected_groups:
                bar_df = bar_df[bar_df["Entity"].isin(selected_groups)]
            elif selected_individual_countries:
                bar_df = bar_df[bar_df["Entity"].isin(selected_individual_countries)]
    
            variable_bar_figure = px.bar(
                bar_df,
                x="Entity", y="Change (%)",
                color="Period",
                barmode="group",
                title="Annual Change in Emissions Intensity of Primary Energy",
                labels={"Entity": "Country / Group", "Change (%)": "Annual Change (%)"}
            )
        else:
            # ✅ Show an empty bar chart with a custom title
            variable_bar_figure = px.bar(
                title="No solution for Emissions Intensity Change",
                labels={"Entity": "Country / Group", "Change (%)": "Annual Change (%)"}
            )
    else:
        # ✅ Show an empty bar chart with a custom title
        variable_bar_figure = px.bar(
            title="No solution",
            labels={"Entity": "Country / Group", "Change (%)": "Annual Change (%)"}
        )
    
    return fig, table_data,table_columns, variable_bar_figure, warning_message,""

def find_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # OS assigns a free port
        return s.getsockname()[1]

# =============================================================================
# Section 6: Auto-Launch of the Dashboard Application
# =============================================================================
def open_browser(port):
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except:
        print(f"Please open your browser manually and go to http://127.0.0.1:{port}")

if __name__ == "__main__":
    port = find_free_port()
    Timer(1, lambda: open_browser(port)).start()
    print(f"Launching MEECC Dashboard at http://127.0.0.1:{port}")
    app.run(debug=False, port=port)





 
