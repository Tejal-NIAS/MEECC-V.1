import os
import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from dash import dcc, html, Input, Output
import geopandas as gpd
from ipyleaflet import Map
from ipyleaflet import Marker
from ipyleaflet import TileLayer, basemaps, basemap_to_tiles
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import WMTSTileSource
import folium
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import pearsonr
from branca.colormap import StepColormap
from bokeh.io import curdoc
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, Select
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.palettes import Viridis256
from bokeh.models import WMTSTileSource
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import logging
import matplotlib
matplotlib.use('Agg')

os.chdir(r"C:\Users\lavanyaa\OneDrive - National Institute of Advance Studies (NIAS)\NDC Tracking\Emission model of Tejal\Further work")
os.getcwd()
data=pd.read_excel('Parent_sheet.xlsx')
data.loc[data["Countries"]=="Namibia","ISO 3166"]="NA"
#print(type(data))
data_norm=data.iloc[:,12:27]  
#print(type(data_norm))
#len(data_norm)
#data_norm.shape
pop=pd.read_csv('Population.csv')
population=pd.read_csv("Population.csv")
selected_columns=["Countries", "2019"]
population_2019=population[selected_columns]
pop.drop(pop.index[179:185], inplace=True)
pop.drop(columns=["2019"], inplace=True)
basemap=gpd.read_file("world-administrative-boundaries.shp")
print(basemap.head())
basemap.plot()
print(basemap.crs)
basemap=basemap.rename(columns={"iso_3166_1_":"ISO 3166"})

#normalisation
data_filled=np.nan_to_num(data_norm)
data_normalised=(data_filled-data_filled.min())/(data_filled.max()-data_filled.min())
print(data_normalised)
numerical_columns_d = data_norm.columns  # Only numerical columns
data_print=pd.DataFrame(data_filled,columns=numerical_columns_d)

#Step 1
# Initialize UMAP
umap_model = umap.UMAP(
    n_neighbors=15,       # Number of neighbors for local structure
    min_dist=0.1,         # Controls compactness of embedding
    n_components=2,       # Dimensionality of the embedding (2D or 3D)
    metric='euclidean',   # Distance metric
    random_state=42       # For reproducibility
)

# Fit and transform the data
data_umap = umap_model.fit_transform(data_normalised)

# Convert embedding to DataFrame for easy plotting
umap_df = pd.DataFrame(data_umap, columns=['UMAP1', 'UMAP2'])

# Plot the UMAP results
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='UMAP1', y='UMAP2',
    data=umap_df,
    alpha=0.8
)
plt.title('UMAP Projection')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()

#step 1 PCA
#pca_c=PCA(n_components=0.95)
#data_pca=pca_c.fit_transform(data_normalised)
#step 2 kmeans
kmeans=KMeans(n_clusters=5, random_state=42)
clusters=kmeans.fit_predict(data_umap)
data['cluster']=clusters
cluster_means=data.groupby("cluster")["PCGDP_2019_$/perons -2017PPP"].mean().sort_values(ascending=False)
reordered_clusters={old:new+1 for new, old in enumerate(cluster_means.index)}
data["reordered_cluster"]=data["cluster"].map(reordered_clusters)
data_describe=data.columns[11:26]
descriptive=data.groupby("reordered_cluster")[data_describe].agg("mean")

# spatial analysis
gdf=basemap.merge(data,how="inner", on="ISO 3166")
gdf = gdf.to_crs(epsg=3857)
#Define a discrete colormap
colors = ['#FF0000', '#FFA500', '#FFFF00', '#00FF00', "#0000FF"]  # Example colors
cmap = ListedColormap(colors)  # Create a colormap with these colors

# Define discrete boundaries
boundaries = [1, 2, 3, 4, 5,6]  # Boundaries for the clusters
norm = BoundaryNorm(boundaries, len(colors))  # Normalize data to the boundaries

#map generation
fig, ax=plt.subplots(1,1,figsize=(20,10))
gdf.plot(column="reordered_cluster",
          ax=ax,
          cmap=cmap,
          norm=norm)
legend_labels = ['1', '2', '3', '4','5']  # Legend labels
legend_colors = [plt.Line2D([0], [0], marker='o', color=color, markersize=10, linestyle='') for color in colors]
ax.legend(legend_colors, legend_labels, loc="lower left", title="Clusters", frameon=False)
plt.title("Clusters")
plt.show()
  
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
gdp_growth['growth_rate_umod']=data["reordered_cluster"].map(gr_umod)
gdp_growth['growth_rate_dg2']=data["reordered_cluster"].map(gr_dg2)
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

#GDP growth rates for G1, G2, G3, G4, and G5 countries based on the scenerios
gdp_growth_rates={
    "GDP_Uhigh":{1:0.02,2:0.05,3:0.06,4:0.07,5:0.085},
    "GDP_UMod":{1:0.015,2:0.04,3:0.05,4:0.06,5:0.07},
    "GDP_ULow":{1:0.01,2:0.03,3:0.04,4:0.05,5:0.06},
    "GDP_DG1":{1:0.00,2:0.03,3:0.05,4:0.07,5:0.085},
    "GDP_DG2":{1:-0.01,2:0.03,3:0.05,4:0.07,5:0.085}
    }
    # Update GDP_UMod_TS1 and GDP_UMod_TS2 based on dynamic rates from gdp_growth DataFrame
gdp_growth_rates["GDP_UMod_TS1"] = dict(zip(gdp_growth["Countries"], gdp_growth["growth_rate_UMod_TS1"]))
gdp_growth_rates["GDP_UMod_TS2"] = dict(zip(gdp_growth["Countries"], gdp_growth["growth_rate_DG2_TS2"]))

gdp_no_degrowth={1:0.00,2:0.01,3:0.02,4:0.025,5:0.03}
energy_threshold={"ECOV_GlAvg75":{1:75,2:75,3:75,4:75,5:75},
                     "ECOV_QPT60":{1:60,2:60,3:60,4:60,5:60},
                     "ECOV_QPT70":{1:70,2:70,3:70,4:70,5:70},
                     "ECOV_QPT80":{1:80,2:80,3:80,4:80,5:80},
                     "ECOV_QPT90":{1:90,2:90,3:90,4:90,5:90},
                     "ECOV_QPT95":{1:95,2:95,3:95,4:95,5:95},
                     "EDIV_High":{1:200,2:160,3:55,4:30,5:15},
                     "EDIV_Med":{1:160,2:120,3:75,4:40,5:25}}
convergence_year={"2030","2040","2050"}
FEPE_ratio={"EE_1":{1:0.01,2:0.01,3:0.01,4:0.01,5:0.01},
            "EE_1.5":{1:0.015,2:0.015,3:0.015,4:0.015,5:0.015},
            "EE_2":{1:0.02,2:0.02,3:0.02,4:0.02,5:0.02},
            "EE_2.5":{1:0.025,2:0.025,3:0.025,4:0.025,5:0.025},
            "EE_3":{1:0.03,2:0.03,3:0.03,4:0.03,5:0.03}
            }
emission_sce = {
    "EMIU_BY": {1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
    "EMIU_0.5": {1: -0.005, 2: -0.005, 3: -0.005, 4: -0.005, 5: -0.005},
    "EMIU_1": {1: -0.01, 2: -0.01, 3: -0.01, 4: -0.01, 5: -0.01},
    "EMIU_1.5": {1: -0.015, 2: -0.015, 3: -0.015, 4: -0.015, 5: -0.015},
    "EMID_Low": {1: -0.01, 2: -0.01, 3: -0.005, 4: -0.005, 5: 0.00},
    "EMID_Med": {1: -0.015, 2: -0.01, 3: -0.01, 4: -0.005, 5: 0.00},
    "EMI_High": {1: -0.015, 2: -0.015, 3: -0.01, 4: -0.01, 5: -0.01}
}
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
    "Emissions Intensity of Final Energy_Chosen Scenario",
    "Emissions Intensity of Primary Energy_Chosen Scenario",
    "Final Energy Consumption",
    "GDP",
    "CO2_Net",
    "Per capita final energy consumption",
    "Per capita GDP",
    "Per capita Net CO2",
    "Per capita primary energy consumption",
    "Primary Energy Intensity of GDP",
    "Per Capita Net CO2 in Baseline",
]
country_group_mapping={"Annex I": data[data["Annex or Non-Annex"]=="Annex I"]["Countries"].tolist(),
                       "Non-Annex I":data[data["Annex or Non-Annex"]=="Non-Annex I"]["Countries"].tolist(),
                       "Group 1": data[data["reordered_cluster"]==1]["Countries"].tolist(),
                       "Group 2": data[data["reordered_cluster"]==2]["Countries"].tolist(),
                       "Group 3": data[data["reordered_cluster"]==3]["Countries"].tolist(),
                       "Group 4": data[data["reordered_cluster"]==4]["Countries"].tolist(),
                       "Group 5": data[data["reordered_cluster"]==5]["Countries"].tolist(),
                       "World":data["Countries"].tolist()}

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import dash_table
import dash_bootstrap_components as dbc  # For better styling
from dash.dependencies import Input, Output

# Initialize the App
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
            dcc.Tabs(id="main-tabs", value="countries", children=[

                # 📌 TAB 1: Country Selection
                dcc.Tab(label="Countries", value="countries", children=[
                    html.Div([
                        html.Label("Select Countries or Groups:", style={"textAlign": "right"}),

                        # Container with Scrollable Checkbox List (4 checkboxes per row)
                        html.Div([
                            dcc.Checklist(
                                id="country-checkbox",
                                options=(
                                    [{"label": group, "value": group} for group in country_group_mapping.keys()] +  # Groups
                                    [{"label": country, "value": country} for country in data["Countries"].unique()]  # Individual countries
                                ),
                                value=["World"],  # Default to "World"
                                inline=True,  # Display checkboxes inline
                                style={"display": "grid", "grid-template-columns": "repeat(4, 1fr)", "gap": "5px"}  # 4 per row
                            ),
                        ], style={"max-height": "300px", "overflow-y": "auto", "border": "1px solid #ccc", "padding": "10px"}),

                        html.Div(id="country-limit-warning", style={"color": "red", "font-size": "12px"}),

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

                        html.Label("Energy Efficiency Improvement: FE to PE Ratio", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="FE/PE-dropdown",
                            options=[{"label": ratio, "value": ratio} for ratio in FEPE_ratio.keys()],
                            value="EE_1"
                        ),

                        html.Label("Emission Scenarios:", style={"textAlign": "right"}),

                        html.Label("CO2-FFI Emission Intensity of Energy:", style={"textAlign": "right"}),
                        dcc.Dropdown(
                            id="emission-dropdown",
                            options=[{"label": emission, "value": emission} for emission in emission_sce.keys()],
                            value="EMIU_BY"
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
            dcc.Graph(id="result-graph"),
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
                    {"label": "Net Zero Year", "value": "Net_Zero_Year"},
                    {"label": "Cumulative Emissions (2020-2100)", "value": "Cumulative_Emissions"},
                    {"label": "Emissions Intensity of PE Annual Change", "value": "PE_Intensity_Change"}
                ],
                value="Cumulative_Baseline_Emissions"  # Default selection
            ),
            html.Div(id="variable-text-output", style={"margin-top": "10px", "padding": "10px", "background-color": "#f8f8f8"})
        ], style={"width": "45%", "padding": "20px", "background-color": "#f0f0f0"}),

        html.Div([
            html.H3("Data Table for the Selected Figure:"),
            dash_table.DataTable(id="data-table")
        ], style={"width": "50%", "padding": "20px", "background-color": "#f0f0f0"})
    ], style={"display": "flex", "justify-content": "space-between", "margin-top": "20px"})
], style={"font-family": "Arial, sans-serif", "background-color": "#fafafa", "padding": "20px"})

print(app.layout)
    
@app.callback(
    [Output('result-graph', 'figure'), Output("data-table", "data"), 
     Output("data-table", "columns"), Output("variable-text-output", "children"), Output("country-limit-warning", "children")],
    [Input('country-checkbox', 'value'),
     Input('GDP-dropdown', 'value'),
     Input("Energy-dropdown","value"),
     Input("convergence-dropdown","value"),
     Input("FE/PE-dropdown","value"),
     Input("emission-dropdown","value"),
     Input("LULUCF","value"),
     Input("budget-dropdown","value"),
     Input("temperature-dropdown","value"),
     Input("figure-dropdown","value"),
     Input("variable-dropdown","value"),
     ]
)
def update_graph(selected_countries, selected_gdpgrowth, selected_energythreshold,
                 selected_convergenceyear,selected_FE_PE,selected_emissionsce,
                 selected_LULUCFsce, selected_budget,
                 selected_temp_target, selected_figure, selected_variable):
    print(f"Selected Countries: {selected_countries}")
    print(f"Selected GDP Growth Scenario: {selected_gdpgrowth}")
    print(f"Selected Energy Threshold: {selected_energythreshold}")
    print(f"Selected Convergence Year: {selected_convergenceyear}")
    print(f"Selected FE/PE Scenario: {selected_FE_PE}")
    print(f"Selected Emission Scenario: {selected_emissionsce}")
    print(f"Selected LULUCF Scenario: {selected_LULUCFsce}")
    print(f"Selected Carbon Budget: {selected_budget}")
    print(f"Selected Temperature Target: {selected_temp_target}")
    print(f"Selected Figure Type: {selected_figure}")
    
    logging.info(f"Selected Countries: {selected_countries}")
    logging.info(f"Selected GDP Growth Scenario: {selected_gdpgrowth}")
    logging.info(f"Selected Energy Threshold: {selected_energythreshold}")
    logging.info(f"Selected Convergence Year: {selected_convergenceyear}")
    logging.info(f"Selected FE/PE Scenario: {selected_FE_PE}")
    logging.info(f"Selected Emission Scenario: {selected_emissionsce}")
    logging.info(f"Selected LULUCF Scenario: {selected_LULUCFsce}")
    logging.info(f"Selected Carbon Budget: {selected_budget}")
    logging.info(f"Selected Temperature Target: {selected_temp_target}")
    logging.info(f"Selected Figure Type: {selected_figure}")
    
    # Initialize outputs
    Cum_Bas_Emi =0
    Cum_Miti_Emi=0
    Net_Zero_Year=10000
    warning_message = "" 
    combined_results = []  # Store results for all selected countries/groups
    all_selected_countries = {}  # Dictionary to track country -> selected group(s)
    group_aggregations = {}  # Store separate aggregations for groups
    
    # Step 1: Identify if selection includes groups or individual countries
    for selection in selected_countries:
        if selection in country_group_mapping:  # If it's a group selection
            group_aggregations[selection] = []  # Prepare for aggregation
            for country in country_group_mapping[selection]:
                if country in all_selected_countries:
                    all_selected_countries[country].append(selection)  # Append group name
                else:
                    all_selected_countries[country] = [selection]  # Initialize with group
        else:  # If it's an individual country selection
            all_selected_countries[selection] = ["Individual Country"]
    
    # Step 2: Process each selected country separately
    for country, groups in all_selected_countries.items():
        country_data = data[data["Countries"] == country]
        country_data=country_data.iloc[0]
        country_data=country_data.fillna(0)
        if country_data.empty:
            print(f"Warning: No data found for {country}. Skipping.")
        # Retrieve all the input parameters
        growth_rate = gdp_growth_rates[selected_gdpgrowth]
        E_threshold=energy_threshold[selected_energythreshold]
        convergence_year = int(selected_convergenceyear)
        efficiency_ratio=FEPE_ratio[selected_FE_PE]
        emission_scenerio=emission_sce[selected_emissionsce]
        temp_goal=temperaturetarget[selected_temp_target]
        
        if selected_gdpgrowth not in gdp_growth_rates:
            raise ValueError(f"Invalid GDP growth rate: {selected_gdpgrowth}")
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
        cluster = country_data["reordered_cluster"]
        print(f"{country} belongs to Group:{cluster}")
        if selected_gdpgrowth == "GDP_UMod_TS1" or selected_gdpgrowth == "GDP_UMod_TS2":
            rate=growth_rate[country]
        else:
            rate = growth_rate[cluster] # selected GDP growth rate
        
        print(f"selected GDP growth rate of {country} till 2050 is {rate}")
        no_degrowth_rate=gdp_no_degrowth[cluster]  #GDP no degrowth rate
        print(f"selected GDP growth rate of {country} between 2050 and 2100 is {no_degrowth_rate}")
        threshold_value = E_threshold[cluster]  # Energy threshold for the cluster
        print(f"Energy threshold value for {country} is chosen as {threshold_value}")    
        ratio=efficiency_ratio[cluster] #FE PE ratio for the cluster
        print(f"Ratio of Final and Primary Energy in 2019 for {country} is assummed as {ratio}")
        emission_rate=emission_scenerio[cluster] # emission density growth rate till 2050
        print(f"selected rate of change in fossil CO2 emissions for {country} is {emission_rate}")
        
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
        FE_PE_standard={1:0.85,2:0.8,3:0.75,4:0.7,5:0.65}
        threshold=0.93
        FE_PE_2019=FE_PE_standard[cluster]
        print(f"FE_PE_ratio for {country} is {FE_PE_2019}")
        full_years=range(2019,2101)
      
        # Initialize an empty list to store the series
        FE_PE_series = []
        # Loop through the years to compute the series
        for year in full_years:
            if not FE_PE_series:  # First year (initialize with FE_PE_2019)
                current_value = FE_PE_2019
            else:
                # Compute the value for the current year
                current_value = FE_PE_series[-1] * (1 + ratio)
                # Check if the threshold has been reached or crossed
                if current_value >= threshold:
                    current_value = threshold  # Freeze the value at the threshold
            FE_PE_series.append(current_value)
    
        FE_PE_df=pd.DataFrame({
            "Countries":country,
            "Years":full_years,
            "Projected_FE_PE_Ratio":FE_PE_series})
        energy_project_df=pd.merge(energy_project_df,FE_PE_df,how="inner",on=["Countries","Years"])
        energy_project_df["Final_energy"]=energy_project_df["Primary_Energy"]*energy_project_df["Projected_FE_PE_Ratio"]
        
        #Final energy per capita
        energy_project_df["Final_energy_per_capita"]=energy_project_df["Final_energy"]/energy_project_df["Population"]
        #Final energy density
        energy_project_df["Final_energy_density"]=energy_project_df["Final_energy"]/energy_project_df["Projected_GDP"]
        
        #CO2 FFI emissions
        Emission_intensity_energy_2019=CO2_FFI_2019/primary_energy_2019
        print(f"Emission Intensity of energy for {country} in 2019 was {Emission_intensity_energy_2019}")
        years=range(2020,2051)
        EI_projection_2050 = [Emission_intensity_energy_2019* (1 + emission_rate) ** (year - 2019) for year in years]
        Emission_intensity_energy_2050=EI_projection_2050[-1]
        years_2100=range(2051,2101)
        EI_projection_2100=[Emission_intensity_energy_2050*(1+0)**(year-2050) for year in years_2100]
        # Combine projections
        combined_years = list(years_2050) + list(years_2100)
        combined_emission_intensity_values = EI_projection_2050 + EI_projection_2100
        Emission_intensity_df = pd.DataFrame({"Countries":country, "Years": combined_years, "Projected_Emission_Intensity": combined_emission_intensity_values})
        Emission_intensity_df["Years"]=Emission_intensity_df["Years"].astype(int)
        energy_project_df=pd.merge(energy_project_df,Emission_intensity_df,how="inner",on=["Countries","Years"])
        energy_project_df["CO2_FFI_Baseline"]=energy_project_df["Projected_Emission_Intensity"]*energy_project_df["Primary_Energy"]
    
        #CO2_LULUCF_emissions
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
        print(CO2_LULUCF_2019)
        print(CO2_LULUCF_2050)
        print(CO2_LULUCF_2100)            
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
      
        # Carbon Budget and temp target
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
        carbon_data["PCFS_CurrAnn"]=carbon_data["Unfair Share"]*temp_goal
       
        # Determine RCB based on selected_budget
        if selected_budget == "PCFS_GF":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_GF"].values[0]
        elif selected_budget == "PCFS_Hist":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_Hist"].values[0]
        elif selected_budget == "PCFS_Cap":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_Cap"].values[0]
        elif selected_budget == "PCFS_HistCap":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_HistCap"].values[0]
        elif selected_budget == "PCFS_CurrAnn":
            RCB = carbon_data.loc[carbon_data["Countries"] == country, "PCFS_CurrAnn"].values[0]
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
        if ((NZY - peak_year) > 10):
            Annual_reduction_rate = Peakyear_emissions / (NZY - peak_year)
        else:
            Annual_reduction_rate=0
        print(f"NZY recalculated from fallback peak year: {NZY}")
        
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
           
        #calculating per capita gdp
        peak_year_df["GDP_per_capita"]=peak_year_df["Projected_GDP"]/peak_year_df["Population"]
        #calculating per capita emission_with mitigation
        if NZY is not None:
            peak_year_df["per_capita_emission_with_mitigation"]=peak_year_df["Emissions_with_mitigation"]/peak_year_df["Population"]
            #calculating emission with mitigation intensity wrt primary energy
            peak_year_df["emission_intensity_primaryenergy_with_mitigation"]=peak_year_df["Emissions_with_mitigation"]/peak_year_df["Primary_Energy"]
            #calculating emission with mitigation intensity wrt final energy
            peak_year_df["emission_intensity_finalenergy_with_mitigation"]=peak_year_df["Emissions_with_mitigation"]/peak_year_df["Final_energy"]
        else:
            peak_year_df["per_capita_emission_with_mitigation"]=0.0
            peak_year_df["emission_intensity_primaryenergy_with_mitigation"]=0.0
            peak_year_df["emission_intensity_finalenergy_with_mitigation"]=0.0
                
        #calculating the cumulative emissions between 2020 and 2100 with and without (Baseline) mitigation
        cumulative_baseline_emissions=peak_year_df["Net_CO2_Baseline"].sum()
        cumulative_withmitigation_emissions=peak_year_df["Emissions_with_mitigation"].sum()
        print(f"Cumulative Net Baseline Emission between 2020 and {NZY} is {cumulative_baseline_emissions}")
        print(f"Cumulative Net Emission with mitigation between 2020 and {NZY} is {cumulative_withmitigation_emissions}")
        #calculating the annual change in the emissions intensity of primary energy
        if cumulative_withmitigation_emissions is not None:
            EI_2020=peak_year_df.loc[peak_year_df["Years"]==2020,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            EI_2030=peak_year_df.loc[peak_year_df["Years"]==2030,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            EI_2040=peak_year_df.loc[peak_year_df["Years"]==2040,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            EI_2050=peak_year_df.loc[peak_year_df["Years"]==2040,"emission_intensity_primaryenergy_with_mitigation"].values[0]
            print(f"EI_2020 is {EI_2020}")
            print(f"EI_2030 is {EI_2030}")
            print(f"EI_2040 is {EI_2040}")
            print(f"EI_2050 is {EI_2050}")
            # Ensure EI_2020 and EI_2030 are not NaN before performing calculations
            if not (pd.isna(EI_2030) or pd.isna(EI_2020)) and EI_2020 != 0:
                annual_change_2020_2030 = round(((EI_2030 - EI_2020) * 10) / EI_2020)
            else:
                annual_change_2020_2030 = None  # Assign None or a default value
                print("Warning: EI_2030 or EI_2020 is NaN or EI_2020 is zero. Skipping calculation.")
                    # Similarly, check for EI_2040 and EI_2030 for the next calculation
            if not (pd.isna(EI_2040) or pd.isna(EI_2030)) and EI_2030 != 0:
                annual_change_2030_2040 = round(((EI_2040 - EI_2030) * 10) / EI_2030)
            else:
                annual_change_2030_2040 = None
                print("Warning: EI_2040 or EI_2030 is NaN or EI_2030 is zero. Skipping calculation.")
            # Similarly, check for EI_2050 and EI_2040 for the next calculation
            if not (pd.isna(EI_2050) or pd.isna(EI_2040)) and EI_2040 != 0:
                annual_change_2030_2050 = round(((EI_2050 - EI_2040) * 10) / EI_2040)
            else:
                annual_change_2040_2050 = None
                print("Warning: EI_2040 or EI_2030 is NaN or EI_2030 is zero. Skipping calculation.")
        else:
            print("Emissions_with mitigation projection is infeasible for the chosen scenerio")
        country_results = peak_year_df.copy()
        # Assign "Individual Country" if no predefined group is assigned
        if groups == ["Individual Country"]:
            country_results["Selected_Group"] = country  # Keep the country name for clarity
        else:
            country_results["Selected_Group"] = " & ".join(groups)
    
        # Store results
        combined_results.append(country_results)
    
        # If the country belongs to a group selection, also store it for aggregation
        for group in groups:
            if group in group_aggregations:
                group_aggregations[group].append(country_results)

        Cum_Bas_Emi=Cum_Bas_Emi+cumulative_baseline_emissions
        if cumulative_withmitigation_emissions is not None:
            Cum_Miti_Emi=Cum_Miti_Emi+cumulative_withmitigation_emissions
        if NZY is not None:
            if NZY < Net_Zero_Year:
                Net_Zero_Year=NZY
    
    # Step 3: Combine individual country results
    # Step 3: Combine individual country results
    if combined_results:
        final_df = pd.concat(combined_results)
        final_df=final_df.fillna(0)
    else:
        final_df = pd.DataFrame()  # Empty DataFrame if nothing is selected
    
    # Step 4: Aggregate separately for each selected group
    aggregated_group_results = []
    selected_groups = [grp for grp in selected_countries if grp in country_group_mapping]  # Selected country groups
    selected_individual_countries = [cnt for cnt in selected_countries if cnt not in country_group_mapping]  # Individual countries
    
    for group, country_data_list in group_aggregations.items():
        if country_data_list and group in selected_groups:
            group_df = pd.concat(country_data_list).groupby("Years").sum().reset_index()
            group_df["Countries"] = group  # Assign the group name
            group_df["Selected_Group"] = group  # Ensure consistency with column names
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
    
    # Debugging Prints (Remove in final version if unnecessary)
    print("Final DF After Aggregation:")
    print(final_df.head(8))
    
    print("Unique Countries in Final DF:", final_df["Countries"].unique() if "Countries" in final_df else "MISSING")
    print("Unique Selected Groups in Final DF:", final_df["Selected_Group"].unique() if "Selected_Group" in final_df else "MISSING")
    
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
        "Emissions Intensity of Final Energy_Chosen Scenario": {"y": "emission_intensity_finalenergy_with_mitigation", "title": "Emissions Intensity of Final Energy Projection", "unit": "Final Energy Intensity (GtCO2/EJ)"},
        "Emissions Intensity of Primary Energy_Chosen Scenario": {"y": "emission_intensity_primaryenergy_with_mitigation", "title": "Emissions Intensity of Primary Energy Projection", "unit": "Primary Energy Intensity (GtCO2/EJ)"},
        "Final Energy Consumption": {"y": "Final_energy", "title": "Final Energy Consumption Projection", "unit": "Final Energy (EJ)","scale":1e9},
        "GDP": {"y": "Projected_GDP", "title": "GDP Projection", "unit": "GDP million$-2017 PPP", "scale":1e6},
        "CO2_Net": {"y": "Emissions_with_mitigation", "title": "Net CO2 Emissions Projection", "unit": "Net CO2 (GtCO2)","scale":1e9},
        "Per capita final energy consumption": {"y": "Final_energy_per_capita", "title": "Per Capita Final Energy Consumption Projection", "unit": "Per Capita Final Energy (GJ/p)"},
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
            labels={y_column: unit, pivot_column: "Country / Group"}
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
        table_data = df.to_dict("records")
        table_columns = [{"name": "Years", "id": "Years"}] + [{"name": col, "id": col} for col in pivot_df.columns]
    
    else:
        fig = px.line(title="No Figure Selected")

       
    # Variable text logic
    if selected_variable == "Cumulative_Baseline_Emissions":
        variable_text = f"Cumulative Baseline Emissions (2020-2100): {Cum_Bas_Emi} tCO2"
    elif selected_variable == "Net_Zero_Year":
        variable_text = f"Net Zero Year: {Net_Zero_Year}"
    elif selected_variable == "Cumulative_Emissions":
        variable_text = f"Cumulative Emissions (2020-2100): {Cum_Miti_Emi} tCO2"
    elif selected_variable == "PE_Intensity_Change":
        variable_text = f"Annual Change in Emissions Intensity of Primary Energy: {annual_change_2020_2030}% (2020-2030)"
    else:
        variable_text = "No details available for the selected variable."
    return fig, table_data,table_columns, variable_text, warning_message
# Run the App
if __name__ == "__main__":
    app.run_server(debug=True)
 
