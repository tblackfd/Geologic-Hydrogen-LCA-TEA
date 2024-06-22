#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/tblackfd/Thesis/blob/main/BrandtModelReplication_v0_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Background / Introduction
# This notebook intends to replicate the Lifecycle Assessment (LCA) of geologic hydrogen produced by Brandt (2023) in order to perform a review of its methods and assumptions, as well as the conclusions published in the associated paper.
# 
# The Brandt paper considers a comprehensive range of potential emissions sources throughout the whole 'well pad' process, from the wellhead to the 'gate', including emissions embodied in the efforts required to discover the resource and produce/install the associated equipment. The paper displays the process in the following block diagram:
# 
# ![image.png](attachment:image.png)
# 
# This notebook also builds on top of Brandt's work in three ways. First, it incorporates assessments of sensitivity to uncertainty regarding key assumptions. Second, it estimates costs of development, aligned with the scenarios assessed in the LCA portion. Finally, it estimates the value of the Production Tax Credits (PTC) that may be associated with each of the scenarios, under the United States' current PTC regime. 

# # Import Python Packages

# In[221]:


#Import relevant packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import statistics
import multiprocessing as mp
from itertools import repeat
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import concurrent.futures
from itertools import repeat
import os


# In[222]:


# Set seaborn whitegrid style
sns.set_style("whitegrid")


# # Input Data Specific to Geologic H2
# 
# The following section establishes input data specific to the estimation of emissions from production of geologic hydrogen.
# 
# Key assumptions are:
# * Well pressure calculated as depth (in ft) by a factor 0.42
# * Initial Baseline raw gas rate is given (23,100 MSCFD) and declines at a manually-defined rate.
# * Low pressure case is assumed to be 75% of baseline
# * High pressure case is assumed to be 125% of baseline
# * "High productivity" wellhead pressure is assumed to be 200% of baseline
# * "Deep" and "Shallow" pressures are calculated based on assumed depths of 12,000 and 1,500 ft, respectively.

# # Constants/Assumptions for Calculations

# In[223]:


GWP_H2_default = 5 #Global Warming Potential of H2, relative to CO2, to enable CO2e calculations. From Brandt OPGEE file: "Low value from Derwent et al. 2020, of 5.0. High value of 10.9 from Warwick et al. 2022."
GWP_CH4 = 25 #Global Warming Potential of methane. 100-year basis, IPCC 2007 AR4, per Brand OPGEE file.

LHV_H2 = 113.958057395143 #mmbtu/tonne. Lower Heating Value of hydrogen. OPGEE source listed as Mechanical engineer's handbook - Energy & Power (3rd edition). John Wiley & Sons. Chapter 17: Gaseous Fuels, Table 2
LHV_CH4 = 47.5055187637969 #mmbtu/tonne. Lower Heating Value of methane. 

number_well_workovers = 4 # Number of well workovers per well, over the lifetime of the well. Default assumption in OPGEE model.

#LHV Energy Densities for gases:
LHV_data = {
    'Gas': ['N2', 'O2', 'CO2', 'H2O', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'CO', 'H2', 'H2S', 'SO2'],
    'LHV (MJ/kg)': [
        0.0000000000,  # LHV for N2, Ar
        0.0000000000,  # LHV for O2
        0.0000000000,  # LHV for CO2
        0.0000000000,  # LHV for H2O
        50.1206975717,  # LHV for CH4
        47.5867143488,  # LHV for C2H6
        46.4501483444,  # LHV for C3H8
        45.8352847682,  # LHV for C4H10
        10.1242877483,  # LHV for CO
        120.2314484547,  # LHV for H2
        15.2434928256,  # LHV for H2S
        0.0000000000   # LHV for SO2
    ]
}
# Create the DataFrame
LHV_density_gases_metric = pd.DataFrame(LHV_data)
# Set 'Gas' as the index instead of a column
LHV_density_gases_metric.set_index('Gas', inplace=True)

# Molecular weights of gases:
MW_data = {
    'Gas': ['N2', 'O2', 'CO2', 'H2O', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'CO', 'H2', 'H2S', 'SO2'],
    'Molecular Weight (g/mol)': [
        28.0134,  # Molecular weight of N2
        31.9988,  # Molecular weight of O2
        44.0095,  # Molecular weight of CO2
        18.0153,  # Molecular weight of H2O
        16.0425,  # Molecular weight of CH4
        30.0690,  # Molecular weight of C2H6
        44.0956,  # Molecular weight of C3H8
        58.1222,  # Molecular weight of C4H10
        28.0101,  # Molecular weight of CO
        2.01588,  # Molecular weight of H2
        34.0809,  # Molecular weight of H2S
        64.0638   # Molecular weight of SO2
    ]
}
# Create the DataFrame
molecular_weights_gases = pd.DataFrame(MW_data)
# Set 'Gas' as the index instead of a column
molecular_weights_gases.set_index('Gas', inplace=True)

# Constants required to calculate psuedo-critical properties of gas mixtures:
pseudo_crit_data = {
    'Gas': ['N2', 'O2', 'CO2', 'H2O', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'CO', 'H2', 'H2S', 'SO2'], # N2 is actually "N2 / Ar" 
    'Tc_Pc': [0.461366863, 0.380013661, 0.511577965, 0.363255567, 0.515156062, 0.778343949, 1.081331169, 1.390483109, 0.471270936, 0.317056323, 0.517230769, 0.678388792],
    'sqrt_Tc_Pc': [0.679239915, 0.616452481, 0.715246786, 0.602706867, 0.71774373, 0.882238034, 1.039870746, 1.179187478, 0.68649176, 0.563077546, 0.719187576, 0.823643607],
    'Tc_Pc_Constant_K': [10.24504569, 10.28145905, 16.74197022, 20.56873505, 13.29861182, 20.68843488, 26.83793382, 32.62750172, 10.61667885, 4.349569036, 18.6490206, 22.92512518]
}
# Create the DataFrame
pseudo_crit_constants = pd.DataFrame(pseudo_crit_data)
# Set 'Gas' as the index instead of a column
pseudo_crit_constants.set_index('Gas', inplace=True)

# Specific heat constants:

# Define the data
specific_heat_data = {
    'Gas': ['N2', 'O2', 'CO2', 'H2O', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'CO', 'H2', 'H2S', 'SO2'],
    'Specific heat C_p': [1.04, 0.919, 0.844, 1.97, 2.22, 1.75, 1.67, 1.67, 1.02, 14.32, 1.01, 0.64],
    'Specific heat C_v': [0.793, 0.659, 0.655, 1.5, 1.7, 1.48, 1.48, 1.53, 0.72, 10.16, 0.76, 0.51]
}
# Create the DataFrame
specific_heat_df = pd.DataFrame(specific_heat_data)
# Set 'Gas' as the index instead of a column
specific_heat_df.set_index('Gas', inplace=True)

#Various physical constants:

steel_density = 0.30 #lb/in^3
mmbtu_to_MJ = 1055.05585 #conversion factor
btu_per_MJ = 947.817 #conversion factor
Pounds_per_kg = 2.20462 #conversion factor
mol_per_SCF = 1.1953 # At standard conditions

# Various assumptions regarding gas processing and equipment:

dehy_reflux_ratio = 2.25 #Dehydration reflux ratio. Default assumption in OPGEE model.
dehy_regen_temp = 200 #F. Regeneration temperature for dehydration. Default assumption in OPGEE model.

eta_compressor = 0.75 #Compressor efficiency. Default assumption in OPGEE model. For the time being, assume all compressor types have the same efficiency (this is the default assumption in OPGEE model).

ng_engine_efficiency_data = {
    'bhp': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800],
    'Efficiency btu LHV/bhp-hr': [7922.40, 7862.05, 7801.70, 7741.35, 7681.00, 7620.65, 7560.30, 7499.95, 7439.60, 7379.25, 7318.90, 7258.55, 7198.20, 7137.85, 7077.50, 7017.15, 6956.80, 6896.45, 6836.10, 6775.75, 6715.40, 6655.05, 6594.70, 6534.35, 6474.00, 6413.65, 6353.30, 6292.95, 6232.60]
}
ng_engine_efficiency_data_df = pd.DataFrame(ng_engine_efficiency_data)
ng_engine_efficiency_data_df.set_index('bhp', inplace=True)

reciprocating_compressor_ng_emissions_factor = 68193.5860604127 #gCO2eq./mmbtu. OPGEE quoting GREET. This is the emissions factor for reciprocating compressors powered by natural gas.

#Factors related to drilling calculations:
heavy_duty_truck_diesel_intensity = 969 #btu LHV/ton mi (Paper does not cite source of this figure)
weight_land_survey = 25 #tonnes. Weight of land survey vehicle. Default assumption of OPGEE model.
distance_survey = 10000 #miles. Distance of travel for survey. Default assumption of OPGEE model. "Estimate accounting for long-distance travel of specialized equipment"
emissions_factor_trucks = 78908.518237706 #gCO2eq./mmbtu. OPGEE quoting GREET.
emissions_factor_diesel_exploration = 78823.3589186562 #g GHGs/mmbtu LHV. OPGEE quoting GREET, but using the values for "Barge diesel" rather than "Truck Diesel". It is unclear to me why this is the case.
emissions_factor_diesel_drilling = 78490.5078472298 #g GHGs/mmbtu LHV. OPGEE quoting GREET, but using the values for "Barge diesel" rather than "Truck Diesel". It is unclear to me why this is the case.  

#Pre-production Wells:
number_dry_wells = 1 #Number of dry wells drilled per discovered field.
number_exploration_wells = 3 #Number of exploratory/scientific wells drilled after discovery of the field.
diesel_energy_density = 128450 #LHV btu/gal. Source: GREET1_2016, obtained from "Fuel_Specs" worksheet.

#Production Wells:
number_production_wells_default = 50 #Key assumption
#Injection Wells:
number_injection_wells_default = math.ceil(0.25*number_production_wells_default) #Assumption is the number of injection wells is 25% of the number of production wells. Rounding up, as you can't drill a fraction of a well.
#Total Wells:
total_number_wells_default = number_production_wells_default + number_injection_wells_default 
# print(total_number_wells)

#Liquid unloading considerations:
wells_LUnp = 0.068996621 #Fraction of wells with non-plunger liquids unloadings. Default assumption in OPGEE model, "from US EPA (2020) Greenhouse Gas Inventory, based on RY2015"
wells_LUp = 0.1 #Fraction of wells with plunger liquids unloadings

PSA_unit_slippage_rate = 0.1 #Brandt assumes that the PSA unit only separates 90% of the H2 from the gas stream.

drilling_fuel_per_foot_vertical = 0.325964356060972 #gal diesel fuel/ft. This figure taken direct from OPGEE model and assumes Moderate complexity wells drilled at Medium efficiency.

steel_emissions_intensity = 2747.8545357015 / 2.204 #gCO2/lb. This is the emissions intensity of steel production, as calculated in the OPGEE model. Conversion factor of 2.204 is used to convert from kg to lb.

cement_emissions_intensity = 36587.7935725105 #gCO2/ft^3. This is the emissions intensity of cement production, as calculated in the OPGEE model.


# # Key Variables / Inputs

# In[224]:


### Gas Densities. Define a dataframe:

# Define the data as a dictionary
data = {
    'Gas': ['N2','CO2', 'CH4', 'H2','H2O'],
    'Density tonne/MMSCF': [33.480353, 52.605153, 19.1738073, 2.4097248, 21.527353]
}
# Create the DataFrame
gas_densities = pd.DataFrame(data)
# Set 'Gas' as the index instead of a column
gas_densities.set_index('Gas', inplace=True)

#Brandt's OPGEE model assumes all cases produce 0.1 bbl/day of oil. The significance of this assumption will be checked via sensitivity analysis.
oil_production_default = 0.1 #bbl/day
water_production_default = 1 #bbl/mmscf of gas

field_lifespan_default = 30 #years

small_source_emissions_percentage_default = 10 #%


# # Define/Assume Reservoir Conditions for Analysis

# In[274]:


# Data for the Reservoir Conditions DataFrame
reservoir_data = {
    'Case': ['Baseline', 'Low Productivity', 'High Productivity', 'High CH4', 'Mixed', 'Low H2', 'Low H2 w/ CH4', 'Low H2 w/ N2', 'High H2', 'Deep', 'Shallow','Exponential Approx Baseline'],
    'Raw Gas EUR, BCF': [67, 33, 167, 67, 67, 33, 67, 67, 67, 67, 33, 67],
    'H2 EUR, BCF': [57, 28, 142, 57, 57, 28, 57, 57, 57, 57, 28, 57],
    'Depth, ft': [6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 12000, 1500, 6000],
    'Initial Reservoir Pressure, psi': [2520, 2520, 2520, 2520, 2520, 2520, 2520, 2520, 2520, 5040, 630, 2520]
}

# Data for the Gas Composition DataFrame
gas_composition_data = {
    'Case': ['Baseline', 'Low Productivity', 'High Productivity', 'High CH4', 'Mixed', 'Low H2', 'Low H2 w/ CH4', 'Low H2 w/ N2', 'High H2', 'Deep', 'Shallow', 'Exponential Approx Baseline'],
    'H2': [85.0, 85.0, 85.0, 85.0, 85.0, 75.0, 75.0, 75.0, 95.0, 85.0, 85.0, 85.0],
    'N2': [12.0, 12.0, 12.0, 1.5, 8.5, 20.0, 0.0, 22.5, 4.0, 12.0, 12.0, 12.0],
    'CH4': [1.5, 1.5, 1.5, 12.0, 5.0, 2.5, 22.5, 0.0, 0.5, 1.5, 1.5, 1.5],
    'C2+': [0.0] * 12,
    'CO2': [0.0] * 12,
    'Ar/oth inert': [1.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 0.5, 1.5, 1.5, 1.5]
}

# Data for the Development Parameters DataFrame
development_params_data = {
    'Case': ['Baseline', 'Low Productivity', 'High Productivity', 'High CH4', 'Mixed', 'Low H2', 'Low H2 w/ CH4', 'Low H2 w/ N2', 'High H2', 'Deep', 'Shallow', 'Exponential Approx Baseline'],
    'Total Producing Wells': [50] * 12,
    'No. of Compressors': [2] * 12,
    'No. of Purification Plants': [1] * 12,
    'Water Cut (bbl/mmscf)': [1] * 12,
    'H2 purification loss rate': [10] * 12,
    'BCF per well': [1.3, 0.33, 2.0, 1.0, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.0, 1.3]
}

#Extract the number of total producing wells, water cut, and H2 purification loss rate as default values from development_params_data:
number_production_wells_default = development_params_data['Total Producing Wells'][0]
water_cut_default = development_params_data['Water Cut (bbl/mmscf)'][0]
h2_purification_loss_rate_default = development_params_data['H2 purification loss rate'][0]

# Creating the DataFrames & setting 'Case' as the index to make slicing based on Case easier
reservoir_df = pd.DataFrame(reservoir_data)
reservoir_df.set_index('Case', inplace=True)

gas_composition_df = pd.DataFrame(gas_composition_data)
gas_composition_df.set_index('Case', inplace=True)

development_params_df = pd.DataFrame(development_params_data)
development_params_df.set_index('Case', inplace=True)

#Extract the list of cases into a separate list to help with iteration in future functions:
cases = reservoir_data['Case']

# # Display the DataFrames (Optional)
# print("Reservoir Conditions DataFrame:")
# print(reservoir_df)
# print("\nGas Composition DataFrame:")
# print(gas_composition_df)
# print("\nDevelopment Parameters DataFrame:")
# print(development_params_df)


# In[277]:


#Now define the assumed production profile over the life of each well. This comes from the OPGEE model, and is based on the assumption that the well will produce 1.3 BCF over its lifetime.

# Constants
pressure_coefficient = 0.42
pressure_decline_rate = 0.95
pressure_decline_rate_default = pressure_decline_rate
rate_decline_factors = np.array([
    0.3, 0.2, 0.15, 0.125, 0.125, 0.1, 0.1, 0.09, 0.08, 0.07,
    0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03,
    0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02
])
rate_decline_factors = rate_decline_factors[:30]  # Ensure it has exactly 30 elements

#plot the rate decline factors
plt.plot(range(0, 29),rate_decline_factors)
plt.xlabel('Year')
plt.ylabel('Rate Decline Factor')
plt.title('Rate Decline Factors Over 30 Years')

#Add a plot of a line showing 5% exponential decline starting at 0.3 for comparison
# This demonstrates visually that the empirical rate decline factors are much more severe than the assumed exponential pressure decline.
plt.plot(range(0, 29), [0.3/0.95 * (0.95 ** i) for i in range(1, 30)], label='5% Exponential Decline')
plt.legend()
plt.show()

# Extract depths for Baseline, Deep, and Shallow from the reservoir_df
depths = reservoir_df['Depth, ft']

# Initial values calculated by depth
initial_wellhead_pressure_baseline = depths['Baseline'] * pressure_coefficient
initial_deep_pressure = depths['Deep'] * 0.43 # The OPGEE model uses a slightly different coefficient for deep and shallow wells.
initial_shallow_pressure = depths['Shallow'] * 0.43

# DataFrame initialization
production_profile_df = pd.DataFrame({
    'Year': range(1, 31),
    'Baseline Raw Gas Rate, MSCFD': [23100] * 30,  # Baseline raw gas rate
    'Baseline Wellhead Pressure, PSI': [initial_wellhead_pressure_baseline] * 30,
    'Deep Wellhead Pressure, PSI': [initial_deep_pressure] * 30,
    'Shallow Wellhead Pressure, PSI': [initial_shallow_pressure] * 30
})

# Calculate the baseline raw gas rate and wellhead pressure for each year
for year in range(1, 30):  # Skip the first year as the initial values are already set
    production_profile_df.loc[year, 'Baseline Raw Gas Rate, MSCFD'] = production_profile_df.loc[year - 1, 'Baseline Raw Gas Rate, MSCFD'] * (1 - rate_decline_factors[year-1])
    production_profile_df.loc[year, 'Baseline Wellhead Pressure, PSI'] = production_profile_df.loc[year - 1, 'Baseline Wellhead Pressure, PSI'] * pressure_decline_rate
    production_profile_df.loc[year, 'Deep Wellhead Pressure, PSI'] = production_profile_df.loc[year - 1, 'Deep Wellhead Pressure, PSI'] * pressure_decline_rate
    production_profile_df.loc[year, 'Shallow Wellhead Pressure, PSI'] = production_profile_df.loc[year - 1, 'Shallow Wellhead Pressure, PSI'] * pressure_decline_rate

# Update other pressures based on the baseline wellhead pressure
production_profile_df['Low Pressure Wellhead Pressure, PSI'] = production_profile_df['Baseline Wellhead Pressure, PSI'] * 0.75
production_profile_df['High Pressure Wellhead Pressure, PSI'] = production_profile_df['Baseline Wellhead Pressure, PSI'] * 1.25

# Update the low and high productivity raw gas rate values
production_profile_df['Low Productivity Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD'] * 0.25
production_profile_df['High Productivity Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD'] * 2

# Calculate the baseline Gas Oil Ratio (GOR)
production_profile_df['Baseline GOR, SCF/BBL'] = production_profile_df['Baseline Raw Gas Rate, MSCFD'] * 1000 / oil_production_default
# Calculate the low productivity Gas Oil Ratio (GOR)
production_profile_df['Low Productivity GOR, SCF/BBL'] = production_profile_df['Low Productivity Raw Gas Rate, MSCFD'] * 1000 / oil_production_default
# Calculate the high productivity Gas Oil Ratio (GOR)
production_profile_df['High Productivity GOR, SCF/BBL'] = production_profile_df['High Productivity Raw Gas Rate, MSCFD'] * 1000 / oil_production_default
# Assume the Gas Oil Ratio (GOR) for the remaining cases ('High CH4', 'Mixed', 'Low H2', 'Low H2 w/ CH4', 'Low H2 w/ N2', 'High H2', 'Deep', 'Shallow') are the same as the baseline GOR
production_profile_df['Low H2 GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']
production_profile_df['High CH4 GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']
production_profile_df['Mixed GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']
production_profile_df['Low H2 w/ CH4 GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']
production_profile_df['Low H2 w/ N2 GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']
production_profile_df['High H2 GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']
production_profile_df['Deep GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']
production_profile_df['Shallow GOR, SCF/BBL'] = production_profile_df['Baseline GOR, SCF/BBL']

# Similarly assume the raw gas rates for the remaining cases ('High CH4', 'Mixed', 'Low H2', 'Low H2 w/ CH4', 'Low H2 w/ N2', 'High H2', 'Deep', 'Shallow') are the same as the baseline raw gas rate
production_profile_df['High CH4 Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']
production_profile_df['Mixed Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']
production_profile_df['Low H2 Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']
production_profile_df['Low H2 w/ CH4 Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']
production_profile_df['Low H2 w/ N2 Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']
production_profile_df['High H2 Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']
production_profile_df['Deep Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']
production_profile_df['Shallow Raw Gas Rate, MSCFD'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']


# Calculate baseline water production in barrels per day
production_profile_df['Water Production, BBL/D'] = production_profile_df['Baseline Raw Gas Rate, MSCFD']/1000 * water_production_default

# Calculate the baseline Water Oil Ratio (WOR)
production_profile_df['Baseline WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default

# Similarly calculate the WOR for the remaining cases ('High CH4', 'Mixed', 'Low H2', 'Low H2 w/ CH4', 'Low H2 w/ N2', 'High H2', 'Deep', 'Shallow')
production_profile_df['Low H2 WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['High CH4 WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['Mixed WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['Low H2 w/ CH4 WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['Low H2 w/ N2 WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['High H2 WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['Deep WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['Shallow WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['Low Productivity WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default
production_profile_df['High Productivity WOR'] = production_profile_df['Water Production, BBL/D'] / oil_production_default


# Add another column to the dataframe to capture the flowrates for the Baseline case approximated via exponential decline. This is calculated in Section 11 and stored in fitted_production_profile_df
# The values from the dataframe are copied here to avoid circular references in the code.
production_profile_df['Exponential Approx Baseline Raw Gas Rate, MSCFD'] = ([17340.29288031, 15783.52096137, 14366.51247228, 13076.71977129,
       11902.7217153 , 10834.12252535,  9861.45973181,  8976.12038396,
        8170.26478214,  7436.75705704,  6769.10198141,  6161.38745468,
        5608.23215117,  5104.73786834,  4646.4461531 ,  4229.29882208,
        3849.60202639,  3503.99354243,  3189.41299938,  2903.07477951,
        2642.44335151,  2405.21081827,  2189.27648044,  1992.72823462,
        1813.82564173,  1650.98451532,  1502.76289359,  1367.84827077,
        1245.04597487,  1133.26858882])

# Print the DataFrame
production_profile_df.head()


# # Sources of Emissions per Brandt Paper:
# Brandt's paper considers that the following categories of emissions contribute to the total emissions associated with a geologic hydrogen development:
# 
# 1.  Operational Combustion
# 1.  Operational Venting, Flaring and Fugitive Emissions (VFF)
# 1.  Drilling energy-use and VFF
# 1.  Emissions embodied in wellbore construction materials and surface equipment
# 1.  "Other" offsite emissions
# 1.  "Small sources" of emissions, not significant enough to be modelled individually but included as course, aggregated estimates.
# 
# The following sections will consider each of these emissions categories and replicate the accounting for these sources.
# 
# 
# 
# 
# 
# 

# # 1: Operational Combustion Emissions
# 
# Total operational combustion emissions are those associated with combustion during operation. These are only relevant in certain cases. In the baseline case, it is assumed that a portion of the produced H2 is used to provide power for compression (both upstream of gas treatment and upstream of waste gas re-injection), dehydration, and pressure-swing adsorption (PSA) separation.
# 
# Hydrogen combustion is assumed to have no CO2e emissions, so the baseline case has no Operational Combustion Emissions.
# 
# Other cases (e.g. high CH4 fields, where the CH4 is "self used" for the processes listed above) will have Operational Combustion Emissions. These will be assessed after replicating the baseline conditions.

# In[227]:


# Define a function to calculate the operational combustion emissions for a given case and sensitivities

def calculate_operational_combustion_emissions(case, sensitivity_variables=None):
    """
    Calculate the operational combustion emissions for a given case and sensitivity values.

    Parameters:
    case (str): The case for which to calculate the operational combustion emissions.
    sensitivity_variables (dict): A dictionary containing sensitivity values for the parameters. The keys are the parameter names and the values are the sensitivity values.

    Returns:
    float: The operational combustion emissions in kg CO2e/day.
    """
    # Call the HC reinjection compressor function to get the emissions from the reinjection compressor. 
    # Note, Brandt assumes in the baseline case that the reinjection compressor uses H2 only (and so has no combustion emissions),
    # however, the OPGEE calculations include CH4 in the fuel gas stream, so there must be some combustion emissions.

    # Calculate the emissions from the reinjection compressor
    reinjection_compressor_emissions = calculate_HC_gas_reinjection_compressor_fugitives(case, sensitivity_variables)['HC_gas_reinjection_compressor_combustion_emissions']

    return reinjection_compressor_emissions #tonne/day

# Test Usage
# calculate_operational_combustion_emissions('Baseline')


    


# # 2: Operational Venting, Flaring and Fugitive Emissions (VFF)
# 
# Brandt paper: "VFF emissions include all purposeful (vented) and un-purposeful (fugitive) emissions from process units and piping."
# 
# Each of sub-category (i.e. venting, flaring, or fugitive) is calculated individually, per each of the following process stages:
# 
# 1.   Production & Extraction
# 2.   Surface Processing
# 
# The following two sub-categories are associated with VFF-type emissions, however these are separated out into a separate section. See Section 3, below.
# 
# 3.   Exploration
# 4.   Drilling & Development
# 
# The following sub-categories are included in the full OPGEE analysis model, but are not considered relevant to geologic H2 production:
# 
# 5.   Liquefied Natural Gas
# 6.   Maintenance
# 7.   Waste disposal
# 8.   Crude oil transport and storage
# 9.   Other Gas transport, storage, and distribution
# 10.  Electricity generation
# 
# Note also that the baseline case does not include any flaring, so it is sufficient to only consider Venting and Fugitive emissions for this specific case.
# 
# The following sections detail Venting and Flaring calculations for each of the relevant items above (i.e. Exploration and Drilling & Development).
# 
# 

# ### 2.1.1 Production & Extraction Venting
# 
# The OPGEE tool considers the following potential sources of vented emissions.
# 
# ![image.png](data:image/png;base64,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F42yZMmSjLvWNttso3fAGTNm6HG/vKeccoqWxWxF+VdYYYXs6KOPVptQRnH2ntddCzqi/0pbeHFubcXpz++6665uu+22azQK1113nRPHdtdff72TLo+TbpD71re+5d785je7G2+8Ue8Gq622mrv33nsbrXXj4sDOzTffrP1yWl7SoVX99a9/7WbOnKmx99prL83HLo3la+d5ZviHf/gH/fjII4/YYd3+53/+p5Mfn5MflX7+3e9+57bYYgv3pS99KRfPPrzlLW/R5w+eQd7+9rfrYVr0Qw45xC1atMi97nWvcxdeeKEe98v7+te/Xo+Zrf7iL/5CW/gXXnjB/exnP3M//vGP3csvv+x6WXer06hugw+tZgy6Kh/5yEccXYy11lrLDut22rRpuv3FL36hX7i0dvrgh/Nz7KWXXtLjK664ottoo40aD4U2wMVtnC/ewr//+7870pgwYYL+aN7//verQ0p/3aLktrF8LdKaa66p+dtnf/vNb37Tkd+jjz7qpBV2p59+up7mRx0Khx9+uNt///3HnCIPAmU25x4TSQ6YrS666CL9UcndQLs1zzzzjEbvZd1D5RulY6UOTysWcwK5Naud9txzT23lcRxadL68ww47TPvH0t1xZ555pvbB5Q6q8eUBVLfnnXee+8lPftKw9Yc//GFtkU866SQ9Tl9cHg4b5+ln84OwEMvXzpdt5SFUT3Mn2X777fWO9KY3vcm94x3vCF7GHeupp55qnKOlXmWVVRx3QHmIdU8++aQ+01x55ZWNOJSXHzXBbCUPofpc8dWvftXttttueg679LLumuko/7OunNjAdrNvf/vbeGc2Z86cxjHbOfvss/WcdAv0kHQ7MvrpxOdPfiTZf/3Xf2mf+EMf+pAek25AJg6SzZo1K5NWLaPvanHZfuELX9A3QB/96EczaS31nHQXNH259WfSkuqxE044Qbfyii6L5Tt79myNIz+m7P/+7/90X36AVnzd8kZGfsiaF31qnj/e+MY3ZltvvbVeY5GtD291sy1vTHiuWWmllbLvfe972q8nHXllmvnlPeqoozR/sxVvhuROoNdhJ9J78MEHe1p3q9uobl9DxcXw2uV4ZZePbQfepNAXtj6uJUB/mRbVWjk7Lj8K7SZZF8eOMwZAN4hnAQu84ZEHWLfGGms0ukZ2LpavnS/b0o/mTQ2te52hrLzUAzuH8uxl3eus7yClVZvDD1KlU1lH1wKlb2lG1yyp5sNqgdxDa7F7MayVTvUaXQvkHL6TPvzomjDVfJAskLo0g/RtpbJ2bIHk8B2bMCUwSBbIdWnqKDg4gI/UMuzOSKONStaRR5U0/u3f/k1fbU6dOrXK5emaIbFA7S08o44Mw8Ot83fTTTcpMQjbMp4BypGRzhRG2wK1OzzmZNhdRjz1DxyY4fvbb79dB1w+97nPKQIsfL27++671fqgwgy1cxfYYYcdFKziAToU9+tf/7rbZ599dCIJmLIQnJoGW3Blw5IZuBLW3d122216nkEdUAb/ehlJ1XPp3+hYoCsODw9/6aWX6p8gxu673/2u+9jHPubuu+8+JSGhGPfYYw8HkwJvwo+CHwlsOaOpf/d3fxeN+9xzzym7I2iy22yzzRwkIgEgjB8A1Ob3v/99pTX33ntvJ9iCnmdUlVFZ//ptt91Wz6V/o2OByn14JoWA/gqDoq2mbzIcHscj0LIy1A5ie80117h11llHW3am6DEhAigLAAugCiSYVp0pd8KhBOOS5gc+8AGNRwsN0UkAVPv7v/97d/HFFzth5xX3BfllUsrTTz+tceyfXW+f03Z0LFDZ4aEeaVHBgYsBZ4ZEtIDj0ZenhYWpgSchMGsKwvLjH/+4Mi201sxQgkKMxeU642xAl0mL5wP4Gx6OmVEEWmxBALIGtWjH7Hr7nLajY4HKXRomUoP+4mDNAm9G6ErIjCh1bJn15OiSMMUNh2e6HNPw6JYcccQROoE8FtfPCyeXGUja2sOYE0Btubvg6P/0T//kJk2a5AxJ9q9N+6NpgcoO3465IAMFg3XMbQVfgKjcaqutdM4rry2ZzSTT3vRBlG4PP4BYXPL1EQgeYLkb2AQNngvor6+//vqarnV5/PL61/vH0/7wW2BcaEneljBDiH69BVp43pXz9sZ3yFBcu6Zsy0wpHoDrkPMoyyedGywLjIvDD5aJUmmHyQI96dIMk8FSXQbbArm3NH5XYrCrlUqfLBC2QM7hEx4cNlI6OjwWSF2a4fkuU01asEBy+BaMlKIMjwVqd3j0J1EqsMDwP1o1Forn7TjbH4sSFyphtvXPdbpvzE1ZOq3EKbs+dA569KGHHgqdCh7rRt2DGY3qQem3a5D6225HWzRe0I8hoE9Dup/+9Kf184svvqh6jGixhILwNJkMHGW2DcWpegxdnGahlTjN0iieF0HV7IADDigejn7uRt2jmY3gidpb+A9+8IMKldGAoJkIGvyd73yHj8rHMxCEdF8I/dVIkX+gAvvuu69q3HA9uDEBNWNYHAasUDBGiZgAwMZxjoFBoJlTDLE4McQYulKEl9yzzz6r6DEtN3GhNqFDQ+Ugz8cee0zjUBbgNkIsDz0p/8QXm9oI9TYoVJimP/7jP26oEVMWFI0J6P+I4KvuYytGsNH7R/5QRLW0Hox8ox8aS08vHpJ/tTs8bAvS2mC/dGeQrebL5VYN/y7qXipAClxWxITLbAoJyaIATOIAI0DKm4DGJDOs0G1EA/IrX/mKHr/22mu1HPA54AtwN8UQixNDjBkZvuuuu5TBp6uCk/zLv/yLSgDaTK9iOcjziSee0LkB4BPnnnuuFiOWh5UxhlLbebY4M5NsYJLOOOMMd/LJJ+tpfpCcI2B761JiK8ovCstuyy23VBvyAxTFN7VfLD1NaEj+VXZ48GCALVBcP9D6ouaFY4IPm+owtCRfosjvqaOHMGE/neL+n/3ZnylFiX4jdw6+VAvcVSAk0WYHRyZwB4CzQfE4BrjF4uCMhhiLxLU6DIgxE0xQC4bvh9RkHwdi0gohVA6O41zo0JOmOWIsD+ITaAxasRHlQAMUGE+6jIpn/CGF8H++H+pBWaFY2ec7sXK1m144l/49WtnhDQ9+5zvfmasdg1eiNamtCC0if3zZkIuo9dKa+OgveK9hwrmECh+YFILDwND7+C/RYPIJovWoXQH2cXxjdYDXQoNqsTg8dPtzcA0xpux0afgx8yNinx+16EySZbAcHGcBBQINAV0VQiwPPSn/WrWRL9lHt80c19LxF5HgmOh66ikwbf9aPSj//GOh9CzeoG4rO3wZHkyLwRsPJloQaOXpdrAuEtPuWkF/iwbl7Q550iUADDPHKcazz7RUtKJMPkFLPhQ/FieGGKMuTFcNyI1nE2Zp4fRWT8u7lW0sD7u2VRvxg6NMbOm60V/HaVFeNkTa0mxlG0qvlesGJU5lhy+roElsW8vHhAtuu8xkIrSC/hZbZJbPoT/O4mrLly/XLg39+mKw6+jO0NrxMMdsKl9q266JxSlDjKkDdSHwPMIDK/rwxWDliB0vy4Nry2zkp8ldih8iE2eOOeYYvZNxJ+JHwEMyz0qhH7ufBvtW3lB6xbiD/Hlcacl20V8Q4t/85jc6bY+HRByNVq0s0NLxFqKoXuxfE4vTC8S4WR5lNpLXv7qCCPOGsUMRhaY7087srmbp+TYb1P0cS9PrSjBJo51Al8iWiyx+ubF0Yosc+PFjcfihdDs0y6MVG9mzSrGs7Ti7f20sPT/OoO6Paws/qEbrl3JztyPwLFFHqDu9OspUdxq5Ft76cXVnktJLFugXC+QcvpWHm34peCpHskAVC3TlLU2VgqRrkgV6YYHk8L2wcsqjbyzQNYdn2B5BJUZYGQCpGupAdhkoYoAoobdVv4Xhua52h8e5GfiA1uM9MKOdCDE9/PDDlaz2iU98otJ1/kWmHMwgDAxQs8Bq4JCPKQyfBWp3eIRRAa1AZ3Eu6MFLLrlEW1hfuReFsRDCGkN2eaBuhhTHkFtGDxnACYWQcjHMDj9URi5jafp1IW4IXaYujOaut956CnfZSHMrdQmVNR3r3AK1OzxdB1tq3YoHR/7JT34yp9wL72Ggk4+wxpDdVnBZHK9MOdjK429DysWIuk6ePFlx21iavgoxP6gQuswdhbuLLPSsPP7SpUs161bq4pcx7ddngdxryXaSjakHw52DAMSCKfeC3YaCj+yutdZaKpRKvBguC4xmAec0rDemHGxxbRtSLgZ0g2xkVLcsTasLgBrMDugyzL+hyzy/HHTQQTrhAroShWNCK3Wx8qVtvRao3MLH8GBaRlowPzDJADKREBru9hFWWksb2uYuYINhreCyzZBbv0y2T1+dVp67DQAWs4X8UJam1SWGLsPoGOtDnQC6CK3UxS9D2q/PApUdPoYHz5w5U7su9HEJ9JGZhSTzOvWz/YshrDFktxVcthlya3n725ByMedptQmtpBlDl7l7cIfA0efOndsA2Fqpi2ae/tVugcoOHysJt3ZaTKaRMWMHOIruBTOWCNZixxDWGLLbCi5bhtxaeS1/+xxSLgZB5sGbKXllaVpaMXSZ1Ufo62OHOXPmWJYto7+NC9JOfRaQNwYaJEXbrW0r778zmdFUmp5IWwfPy8ogmbziHHNO3qVn0u0Zc9w/0Eq+fnxBjTN52M2kJW4cljcs2UsvvdT43CxN0hBOX+NTZ66Vll//SOuGG27IZKpfIz12WqlL7oL0oWMLJFqyvrZjTEqoFdBF47Uk61fRyqf3+2PM1NMDyeG7bG4G4pYsWaJTG0PLA3U5+5R8wQK515LWJy3ESR+TBYbGAjmHlw7S0FQsVSRZIGSB2t/ShDJJx5IF+sUCyeH75ZtI5eiJBXJdmjpzRIIOhS6WjUSnxgSAquYB2otUHANWiDdBYMKvcBx5PwugwHbejqVtsoBZoCstPJJ3wGJoGgJzoVPDQFQnwdBeQ33RdoQ+RInMx3ntfCd5pWuH1wK1OzxgFCKjUJOo0fIHLWi8DK/pQihtCNMNmd1QX0ZuZXBHxUR9nNfOl6kKh5BdPy9DlGFlkPVDDDWGCVdV5PXxYiCzEPocsknoWKxsfh4I26bgXO0OzwLEiIdCG9K1QB2MY0g0v/zyy/o5hNKGMN3QF8Siw8BXzKYCWwDl9XFeOx9TFY4hu35e/ECXLVumP9opU6Y40XhXJiaEHldV5PXxYghTykVjscceeyjOAH8TsknoWCsIM4h2CqI/WtUIMTz4hRdeUMcmXTBZviBedy5YsEAdlZYZodIiShvCdMvKRl+eHxVL19u+iTTZdabmCxtz+umn6+EYsmvXsMXhYWggQsGYcX4mssTQY1PkRUWMbhyKvHS16Mahw85oK/J8/MHX8JxBMLz4qKOOCioFh2yC3CDSgfzIuSvQrUOSPFY2y0MzTP+qt/AxPBgVL1p0Aqq5LIZwyimnNEwdQ2mbYbqNBNrYCakKx5BdP1mclh8SAU1K4K8yTNgeyNtV5DW8OIYLh2wSOlZWNsvDr98o71fu0sTwYFakQKecFpxAn9p/YI2htDFMt9Uvx3DeZvFjyK5/HfLe9Hl53kCtmKl+rWDCfhr+fjNF3hguHLJJ6FgnZfPLOQr7lbs0MeMASsG/c5s+6aSTdPYTt1VUdkEXQGlh45lVhLot3R76+WC6tF5cy4wpHgaLIYY++Djv2muvXbxMP9u1dCmYhUSr7WvA+xdRRsoH6kwLL6SjW2ONNbSfjdYjD+CzZ8/2LwnuW5786Lnz8Uxz5plnNhBpOw/6zAMmzyTEoavCxJGQTfhhF+1E2jwDhMpmeQQLOIoHjbeUuttuLVt5c5AJLZiJXuGY9EIoLZFCmO6YiyMHijhvJFpTZNe/Tt6I+B91vxkmXLxAJqpnsuCDIs3izMXTuc8ypjAGfQ7ZJHSMhNotWy7zEfkwcrRkr5Fdk6D2u3Wj2LD2S51HzuExfC+R3VFQ5O0XZ26lHLk+fOrvtWKyFGeQLZBzeOnGDXJdUtmTBZpaoPJryaYppwjJAn1ogeTwffilpCJ1zwK5Lk0d2SBvAb1oAfjKhJUGBd1tVs5m563uadt/Fqi9hWdl6iOPPFJXqV64cKGuyWoKwIbu+jhv/5nEqR4kiw/HgtUjdj4d718L1O7wVJURVAhDRk0POeQQ1VvkuKG7Ps6LtF1IRbgZdssDdgipjWHB5G+hWdpWzlhazc6DFzdDkK0sadtbC3TF4bnl77333qoTj3rX/vvvr7UydNfHeUELQirCzbDbmAJvDAv2zdosbStnLK1m51tBkP3ypP3eWaCyw4MHz5gxQ7HUYnEhDZnlBNtNHx4xVSYpWIjhvHbetobdgvki1wd2y3RBfiAxBV6uNSyYmVe0xqFQlrYfv1laofNFBNlPL+2PrwUqP7QaHhwSFwK0Yoofgf47UBOYcLNgs6IsXhl26yO1xBdepYH0hrBgS9O2ZWlbHLbN0gqdB0HefvvtNRlfNdhPN+2PjwUqt/AxPNivBkP4LHAA/cjkBz8Yzktrj4MQl5ax1RBDalu9vpvxWkGQu5l/SjtugcoOH0sSPAHFAiZDMCPp5JNPVkwYbNiCj/PGVIQtbmhLHq2oCXNtu7hEWfyyc35eMdXgUF3Ssd5aYNzgMfr0zNu0BQPozlSZncPaTa9//esb7/p7a75wbrfddpue2GGHHfSuBQNvs8DCV6SjvbJA5T58pwWcMGFCLokqzk4CPB/0W1h33XXHqAb3WxlHtTzj1sIPu8F7iSAPuy3rrF+uhW/WR60z45RWssB4WCDn8AkPHo+vIOXZSwvU/paml4VPeSULtGuB5PDtWizFH2gL1O7w6M4gDGQBfReEjSwUz9vxTrfwO+hZxgLnkcqrGpqlXzXddF1vLVC7w0NJws4Q4FgYWEIHkoD46Z577tmAxfSg968TbLgZsouepc/pe9lGd/3yNEs/mkg60VcWqN3hgalQ2iLcfPPNypQYR3PPPfe4iRMnqrBqSEHYx4Zj+G9MEbcZsgvhyJ0nlm5IMdgvj6UfU+qNocR99W2nwlTXlozZjsUJFi9erKOodGcQ+sRJRGRIufitt97aXXfddbqYAZMsQIdPPfVUTc7HhmP4r6+66yviNkN2zeFj6YYUg/3yWPoxpd4YShyzUzo+PhbIvZZspwgx9WCwWxganJmWnu4NLfxNN93kcDa6NDEFYR8bLsN/myniGrLrqwZb3WLphhSDUQCmLr4qMQ4fU+oty9fyT9vxtUDlLo3hwe985ztzNWDw6r3vfa8KqILG8oc4KSTko48+qn36mIKwn5CP/6K36OO/zTCEELJracfSDSkG2zX+tkyptyxfP420P34WqOzwZXgwkzTottASE5gMgjgpLeWGG27oYgrCxDVsuFv4byzdkGKwXx72CUmp9w92GNT/lR2+rMI4OGHnnXfWLS0yPDzi/QTUeRH2p8uxfPnyhoKwjw2X4b/tIBDFuLF0KROyeCgGX3bZZe64447T8qHCwDRFCyyUQH8eaA11Xyash0Ix31CcdKz3Fhg3eIxJITgY8taihuugJ0GFi9hwXfgvD6BPPfVUQ4Y7li7O7MtoF8tjXxGrfTCzi7dOKQyOBcbN4XtpokWLFumDMsvX8MCZwuhaYCQcnnfsv/vd79wqq6wyut90qrlaIPdaMvU7k1cMuwVyDp/w4GH/ulP9uvKWJpk1WaBfLZAcvl+/mVSurligdodnaXQGmfzACCtrK1m4/fbblaBk9JW5n/0QmuG/CS/uh2+p8zLU7vBo0hx66KG5kl144YXukksuUecGF0Y8FVkOuJSpU6e6hx9+OBd/PD40w38TXjwe30r9edbu8GVFZBl6Ri4feughB3zGcu78EIoTN2Lqu9wNQlgxd5XddttNB4zQgmHwyA8+UsxdJaQ6bPhvDPMlzfHGi0MIcxmuzKjy5ptv7jbaaCN39tlnK9bBWrB33nmnO++88xyLSG+yySY6ujxv3jw1WSu2Ctk7dCxWNj8PiNpehq44PGKnOJ793XrrrVonHHvatGm5+oH4mg6lnYip77KAMS1tESvmh8Q79scff1xHP4HT/OAjxSuuuKIu4gs1idgrqACCUDg0YFkM8zWHH0+8OIQwl+HKaPWzXOaWW26pCDbkKndYGhq+I+Yn4OhnnHGGKsRhs1ZsFbJ36FisbH4ePuLtf2fd2s+9lmwnkxgeTBo4H2yKhblz5+ouzgZS0CzQCrNaNq3TiSee6OwHE8OK0XL86le/6sAHaL2N2fHzMaSYFcJZhfvuu+9WLAAEGOTAD2WY73jixSGEGeeN4cqmkAwGAQ2K+jL15UdAq/+Rj3xEGSc4J+QBeU4hNLNVyN6wUcXvgLkQsbJZHr7de7FfuYWP4cEUGiqSVtv+ICQJkydPViZeP7zyj1YHw/sBcVWT4PPVd2NYMUux08LQak2fPt2xyEIxGFIcw4P9+GWYb+z6XuDFoTzKcOVmCsnMP7DASDT2IzSzVcjeoWNlZbM8LP9ebSs7fBkeHCv8zJkz9ZZJH45Av49VQg444AD9bP9i6rsxrPgzn/mM3jloZY444gjto1paxW0MDy7Gi32OXd8LvDiURye4MhN06M6xpYHZZpttctWO1TVk79CxTsqWK0iNHyp3acrKEEMUQG9piZnUfeyxxyolaTOg/PS4vdKloevhk4t0k/hx8NqTZXVYPYR+PZguLQw/HrpMPHgWg5WJBzl+cDy8MbGELpDdTYrX8Nmus3Ox6ykb5aKOr33ta90NN9ygk18ML4YKJfDMwDMKeDFvqmbPnm1J57bFfDkZygNis5X0/MQtbR7U3/GOd6gdEHy147aN1TVkb+YxFL8D0o6VzfLwy9WTfcEJNEhmttuTrfQXM3G4YF7Skmf8yVuJTBwnk4euRjxBiTNh6PUz14sSgu5zXGQ4MmmxGnHLdmSObSZfeFmU3DlxiEwWd2gci10vd61GHHaog5XRP1FWfz9eaL+YB3HaTU9eDWcyi0xtEPseLO9QXUP2Dh2rUjbLtxvbhpf32uHLKrNkyZJMWvdM3uhkssJIJn33suhdPydvZjK5o2TXXHNN1/PqVQbm8L3Kr1/y6Vs8uJ/Ud3n/PWx4MZNvCKOGTOf68OPWr1LTp3/JAt23QM7h5bbT/RxTDskC42iByq8lx7HMKetkgcoWSA5f2XTpwkG0QK5LU2cFQIChIN/97nerXAdYgYUFCxYoD8NiZGAADIH7gQEp3i/71/jne73PkDvv7CE7Q4Hz8krOTZkyJXS66bFm6TdNIEVo2QK1t/C8XYkhwNBzaE8edthhDpkMfhAQe7NmzdICL1u2TH8gLFvPKF1M86Xl2tUUMaHDNRmyH5Kx96NSFtvtaCvseyatei4NGbrORNwoO/fcczMBlXKDQ0InZpMmTcqE7MsEC8g+/elP67Xy2iwTrRodUPETe/nllzMZrdT34gImZULb6Wl5bZjts88++v5eWuJMlo7U4wwuyQ8wE3YjE2ApKw7a8G6d9ARUywTO0sEYWYUv22yzzTK5E2kaV111VcZ76zlz5mSCVOg54ghi2zgvo8NaLwZzitdTZq6Tu1Ymd7RMqM5M8IlMXglmIvKUWfoMUgn+rPE4z+AbIZavnnzlXyiPsvRkhFu/J+GcMhn51vIIe5PJaHEmEFgmjU628cYbZ8JMZTKarbm0YquQvUPHYmXz8xCI0K9iLfu1t/BlCDAtOkuy+68/V199dYc+JYsVMLwNhUkAGVhrrbWcAVB6UP4ldPhwRSjMHrZN6LBZonxbuQ8fw4PLEGAGb6QlGlMimBL4EwZBOM+MKRwe7sb/cXBhQoff5U4//fQxNkzo8BiTBA9UbuFjeHAZAgxMtHDhwlxBQFKffPJJ1XF88cUXte++dOlSXbkaSKkYEjq8En3PolmUd19ttdX0OI0H4F0Znmt3zhVWWMH5mLAl7B8bJnS4ssPH8OAyBBg+Hqb7yiuvVLvyxTHBg2lmdHWgHQlMQbMvRA94/xI67BnD203osGeMkt3KXZpYmmUIMIsLyAOYMuvHH3+8SmMTn6lgBCaAf/e733W0OhYeeeSR3Ou+hA6PRZaxVUKHzWOabO3RV6LZbm3bGLIqry4zke3InnnmmbbzSuhwucmKb6GIHfseYikZScnblWFDhxte3g2Hjxm0k+MJHe7Eeq1daw7fWuzBitW3eHDZjSmhw2XW6fzcMKPDuT588RVg56ZLKSQL9JcFcg4vN6f+Kl0qTbJAzRZ49XVIzQmn5JIF+tECyeH78VtJZeqaBbri8HSNGDxCgoJR02EIILxFDUy/XpyHB6oamqVfNd10Xd4CtTs8XDjLT8LCgA2ItIUuKpzPdvA+JUR48L6zUIlrd3hGSxEkpXVHsZZl55nMQasfU5NFGk/e/Wr5+JGYAlZMZRYVAUGB3Xrrref222+/hpYkryuTuvDHVSrv/e9/v3viiSeiNkesahTVhWt3eHQiwXqBy2bMmOGEb3YImPLKM6Ymi4KY6Rryo4C3IcRUZhMinBBhdZAK/3KvJdu5PoYHkwb67zfeeKM6OOwL6sEwMjh8TE02lrdM2lA5PP98QoQTIuz7Qzv7lVv4GB6MoOk3vvENXQj46quvVvRXcABd8qYMV7VCw8b7IaQymxDhhAj7PtLOfmWHj+HB9KvRab///vu1HOiREyAgY2qysNc4MX1wWu9mISHCYQslRDhsF/9o5S6Nn4i/z8PSVltt5XbddVd9WGUGE1LKMO8TJ04Mqsky6Zu7AorA4ML+iG8Id0iIcEKEfZ9ra1+cS4NcZLu1bJlUDAIcUvON4aqyrExLeSdEuNxMCRGO26fh5XU7fDzLzs8kRLhzGzZLYVgR4YHEg7mFJUQYK3QvDCsinOvDh/rL3TNpSjlZoPcWyDm83OZ6X4KUY7JADy1Q+bVkD8uYskoWqM0CyeFrM2VKaBAs0DWHByIDPwBSkkWxarEFq3CzSDHpId/Rabjuuus0iYTmdmrJwbm+Kw5/8MEH66LEQGAw4gxCsfpzJwF1MqS1GbkFHjMNyk7SBF0mNEN/O8kjXdtfFqjd4VmanckfTJYQdV39Q/fQGJmqCC/Lzsvyj+6ee+5pWDCGG8eOgxWzligjuqARSMgRRH9F5bu5G3EeTXvinH/++Xo+hiPryVf+WdqwPwnN9S3TX/u1O/yDDz7oZF1Vh8oYiwgwEYRjIketQql8pmvywAMPuP3339+deuqpahGEUxFTFSlpRRJYZt4PLGbMYsJwOhZiuHHs+LXXXusWL17sANxYjNgWXJARXseS8nD8LC580UUXucMPfxXBjeHIVg62Sb3Xt0Yf79uImxTRdlvannPOOZlMIMhE4i0XX5R/sz322EOPCRCWffCDH8x22GEH3ndm4lSZtL6q3X7aaadlO+64YyZS2RqXtVjRT0cjXlDiXJp8WLRokWq4s49+OZruLBSMtrkF4fAzITKjx7lGlHc1uiy+oGvA8oG6UA+2skCDnke3Hv12gqz+ncmPQPfBJUTiW/f9fwceeGAmE170EAv0yo88Wg7ywSYEdPPljqD78+fPV717Rjl9u6LRji2ptwB4GldWENfryJM/WU0lo04hO4aOxWzn56EZDdm/yi18DA9GIZgWnbDBBhvojKdTTjlFP/OPlhsmnr44t34LdCVo5ZkIMn36dMcsqGYhhhvHjtN1ectb3qLJQmiGBtre+MY36vmVVnoVwY3hyH75eF5J6r2+Rfpzv7LDx/BglrBB9toUgnEy/4FVwC/tP8tqIO5HP/oRtxW1DETl888/r90NaeX1OaCZyWK4cew4s7Ho7vz2t791119/fSPvZvnEcGT/uoTm+tbo3/3cSGsdxYSHR/aaaX0nnXSSOjGzlmQJGW1RUbmVLoL2lXkwZHof/Xq04GnluRbH5wGyLNA6SxcliBvHjjMPFkcHQaY1Rke9LNgdIIYj+9dSL/r/pE26N9xwgz6L0P9ff/319aGdeb7NguVJQ8HdkucgVkax47ZlPipzfnmuIY50cfS5JGRHfuBF25J2rGyWR7OyDuR566JJ4W23li19dfq7rNVUDPRxly9frodRp5W3L7rPcXmNGUSKi2n4n2O4cez4U089lcnbIj+J0v0yHLl4YUJzixbpr88DS0v2snWRH66jO8Tdi7dIaNzTYnYrnHDCCfpGy+8KdiuvUUs3OXyL33gvceRhRXNbNHVXo+X68EPdd+uqGVPig2KBnMNLb2tQyp3KmSxQyQKVX0tWyi1dlCwwzhZIDj/OX0DKvrcWqN3h5bWcvo/2q/Hoo4+qEJMdu/322x1sDBo0PAx2OzTDfzk/iMq/hknbttt2HIb0a3d4OHhW0vbDhRde6C655BJ1bjRoEE6FnmTUc+rUqY4l6bsZmuG/wGwmHNVqOYhvryabpd9qmu3GM0zatu1eP4rxa3f4MiPCyjz99NOqPQnPDpXID6Goux7Dcauixc3wX2hJ+Bse2hmxZAQYRPjuu+/W6oTQX3ggfrDHHHNMAy+OYckx7Ni31bbbbqt2YOR5ww03VBuRHiPUiFmFyuVfzz531912282tueaaTuA0R71SyFugKw4PAIbB7Y9ZSgQce9q0abkS8EWzQrcfaLFoNUFuYdZtUYWqaHEz/Ncc/r777tPhdph+IT4VXRAhqSD6C6Y8efJkd/LJJ6tjgRfHsOQYduzXGajtrrvucgsXLlQ9TuYUcLcEU+AOiE2K5fKvZ78ZYl2MP4qfc68l2zFAmXowXDtsiQXUgwnw53AyzUI31IEtT8GVldJk0QZBhe2wbnGoddZZR1t2ZAEhOgVDUIeHz4EQPeuss5xguHoO5n/VVVdtpIHDx9SRy/IlAUGl1dm5AzKqKzi0tuw0GrFyNTJ+ZQfITfBenTPAHYEZYinkLVC5hY/hwSSPE9Bq2x+3aAItIq2oHy6++GL9gv1jMRy3DrQ4hP9a3rTSiL4CY/H32c9+ViGzEPpr1/jbGJZMnLJ8Oc+zDXdAJsaceOKJun/vvfe6nXfeWSenhMrFdX7gmaJdxNq/fhT2Kzt8DA8uM9rMmTN1kQMoPwJ9TuhI6Ek/xHDcOtFiPz/b33333XWmlkxO0ZVF5s2bpw4fQn+5BgrRDzEs2Y8T24depPvEG6Ptt99eZ3/xA4A0jZWrmFYVxLqYxrB/ruzwZYaJIQqgs7RAQFF0HUBbt9hiC9WS99MDx2X1D+IAalmgm8T0PLojQlvm0OKjjz7aTZkyRVcc4YtvJRTLCXLLMcqFAjL5MRWQLXwL5ZcZSu64447TMtD9IF8LdHt4HgAHBtM98sgj7VRuW8zXTtIF2XTTTfXj1ltvrQ+sEyZM0KVpQuUioqXFljyr2MHyH4mtwZtSWdvtyTaG7pJ5GY5bN1pMfsKb63Q89gnyXjuTNzt/+OD9L6K/8vamgTZ70bKyuvnx2t2PlctPpypi7acxzPsNL++1w5cZtZfqwPJMkclryEzWoiorUjo3JBboWzy4Vzgu79h59cmbpRSG3wK515LWHxz+aqcajqoFcg4vd61RtUOq94hYoCtvaUbEdqmaA2iB5PAD+KWlIle3QK5LUz2ZsVeCAMOAAGExWmiydsRcsGCBjiiKWpYOf7/nPe/JJcCAFMvV+9fkItT4gYEeRlWhNkOB8/KqT9/xh843O9Ys/WbXVz0PMgzDM2nSJN0WbVw13UG/rvYWnrcrMQQY+m+XXXZxhx12mIqX8oNAuGnWrFlqRxgVfiB77723rukaG7ip0+jN0N6EDtdp7fFPq3aHL0OAL7jgAsWDETSFkb/qqqv0LoBOO60+XA3yeygEA2LxGXUyPyR0OKHDvj+0u1+7w5chwLTocCL+608RJnUiqKozjlDYMt13UGARR3Vve9vbcnVK6PCTKkOY0OGcW7T8oXIfPoYHlyHADPAwmaEYmP0E983gD+eZMYXDc7fwfxxcl9DhhA4X/aedz5Vb+BgeXIYAQwQywcEPTBZhdQ+AMERYIQ6Z8IECMTBUMSR0OKHDRZ9o53Nlh4/hwWUIMHw8bLkpCzPQBfu9ySabaFcHVJjAbJ9iV8YqldDhhA6bL1TZVnb4WGZlCDB68eC+IKz0z3n1SL+eB1QC/VJZDEEnYTDhgb+imkBCh51L6HDM+1o4bhCcRLXd2rYxTFZeXaqy8DPPPNN2XgkdDpssocNhuxSPNry8Gw5fzKyOzwkdrsOKo5tG3+LBZTenhA6XWSedK7NA7rVk8RVg2YXpXLLAIFog5/ByoxvEOqQyJwu0bIHa39K0nHOKmCwwDhZIDj8ORk9Zjp8Fcl2aOoth79TBUz/0oQ9FB5LayRNyEewV9gb0tVPkFWiNJe0TwtvOtzDYcbvSwh988MGqOsaoKgNHu+66a26t1iomAz9AtwX5u7rUcmU1ai1KM0S4SnlbucbqYdtWrklxOrNA7Q6PDiJoANTkeeedp3+IogKIEXiluO+++7p1111XW2gmihCaKd/SEsvylooO6wXyL6bWGztuKsCoA4NGALMRmqkLx5BkvfiVf0n917dG/+7X7vBAX1tuuaVK1jGTCOqRY5tvvrmSkFUVgFlAAUUwVHstxNR6Y8evvfZaB4uPehmKYjajqpm6cAxJtnKwTeq/vjX6d79yHz6GB7/wwgsNBBitcxBfXncywQOxUlpq9MsByNBfJw6hmfLtm970Jv0RsYK2BRw7pNYbO87dhNW4t9tuO2V55s+fb0k1tiGV3xiS3LhIdpL6r2+N/t2v3MLH8GAQYFp0ArDYd77zHXfKKac0LFCHArAlFlPrjR2n60JLTOAHFBpoC6n8xpBkKwfbpP7rW6N/9ys7fAwPZo4qXLshwDiZv6J0nQrAMbXe2HF012n9Uf29/vrr9c7TylcTQ5L9a5P6r2+N/t2v7PCxKrE8O1w7CDD7pg3P0i20qHUpAJNWTK03dpzuDBNOQJhZOIBZVmXB7gAxJLl4bVL/LVqkDz8bNydFs91atvKmRBFgkZkek17dCsAxDDl2XFb1yECUWw1lSHKraRAvIbztWKs7cQeSlux1u/HYY4/pKiXcsR5//HGdxGIr+PW6LCm/ziyQHL5F+/UKSW6xOClaRQvkXktan7ViWumyZIG+t0DO4aXX1PcFTgVMFujEArW/pemkMOnaZIFuWyA5fLctnNLvKwt0xeHpGgGQzZ49u7GKdi9rDe4LvBYLnC/Kf8Tiho43Sz90TTrWHxao3eGRlkZFDEiMQR4QXBb4Jdx///2uF6/zmuG+g6oI3B8uM9ilqN3hmfjB5Axa97PPPltZGgSXaPUBvRjaP+aYY/Qzy6OD6iKRDUhGiOHDV1xxha5XCnW50UYbadqM4kJQcjfxQzPcFzoS3oYyhcpgGPGb3/xmVTN+4okncmW39GMYMmXlh029qN/555/vFy/tj6cF5EvXIGWw3Y638kVnogqcyUK/mUheZ8LWaJoywygTh80YaRWFsUz4k0x+IJnI7WWyCLGOfn7jG9/IZCZTJmRlJohCJsP1eu0555yTTZw4MbvjjjsyQQQyWUxB92UR4kw0KHNlJq4gDBlb+bFlIr+dCV6cCczWSEtW/46WQTibTBY5zmQCS3bIIYdkxPXLbukTT+5m2UMPPZQdf/zx2bRp0xrph/LNFTJ9GBcL5F5LtvPDi+HBpCEO4G688UZtzeFQ5s6dqxJ6hviuuuqqjokirLRNyy6OrDOZZMg/ig+TLlP6wHDRjGc2FfvMgPLhNOL5IYT72vlYGZiwAo8DEXrWWWc5Fmogn5VXXtlRdgsxDJnzZfna9WnbewtU7tLE8GAmV0grrY579dVXqzKwqIU5huf9ABuPdiSTRPijnw/rHsOHufZtr2jFcx0/nlZCCPe162Jl4Mdk3D2AGT/MUIhhyMQtyzeUVjrWGwtUdvgYHgxvwqwkHlAJtIwEnJQAmkvYfffdtcU87bTT3H777efmzZunThbDh/Wimv/FyrDTTju5W265RZ8nzj33XH3mIGsruxUjhiHb+bTtPwtUdvhYVViyZqutttKJ23DkrPgh/WyVxObtzdNPP63osPTvFRfmoZP4YMNMu4vhw7H8ON4qElGMV1YGoTwVI77sssvccccdp2+erOxWlhiGbOdtW8zXjqdt7y3QNXiMNx30ydF+979w3mz8/ve/V+emuj/5yU8cq/nZTCSOPf/88w6HW3vttXUFvQkTJjTic77TwB2IsvE2hRAqA8d5m8N0RAvFsttxnil4E8WzSAr9bYGuOXy/VnvRokX6fMHDKK9JUxgtC4ycw3PnQZ6D9aRSGD0L5F5L+l2P0TNFqvEoWCDn8DISMAp1TnUcYQvU/pZmhG2Zqj4AFkgOPwBfUipifRbIdWnqSJaBJsSX/MBrx4033rhlxV/wW0Zfp06d6ieT9pMFOrZA7Q7PrP4DDzxQJe2sdGAAqP9CUqLv2CyA9/JuPDl8M0ul8+1aoCtdGjiSiy66qPHnC6DG8N+vf/3r+iMRStEJEakOH8NsW1HzbdcQKf5oWKArDk93hGF3+4OPsRBTD37uuedU/g6uhjsCYBdcPc7Pj+fwww9XRTPSaUXN1/JL22QB3wKVHR48eMaMGcHRSjBaVumwP4bdLaAejPQ12pM333xzQz2Y8x/4wAd0QgZMjQXDbFlkgYkXhKKar8VN22SBZhao3Ic3PPgNb3jDmDw45isGE+G+++7TeOC/xx57rC6KAGhGi2+BGUbFEMJsUfMFSiPA4MDmpJAs0IoFKjs8eDB/7QYf/91///11ml27aZiaL9czucTQ43bTSfFHzwKVuzQxU4EnxBAFjpfhv7HrLC8736qar12XtskCZoFxgcc6xX9vu+02Lf8OO+yg/fkzzzyzsQiDVSxtkwVCFqjcpQkl1uox5oXa3NAqDDkLorG4ga/m22reKd5oW2BcWvg6TJ7UfOuw4uilkWvhrY88emZINR4VC+QcPuHBo/K1j249a39LM7qmTDUfBAskhx+EbymVsTYLdMXh6RqNp3pw0TpJ7bdokdH9XLvDl6kH99LMvlJxMzXhXpYr5TW+Fqjd4cvUg2n5Q2q95513nmNBYzRsWEMVFTJCKygx+o777ruv49082pMsL0/wlYqT2q+aJP3DAuKEGrxdO1R5G1MPjikGf/7zn88EAstEgzITPcpM5LA175iSsCwqnIk4UyYyeNnll18eVBtOar+Vv76hvjD3WrKdJqCKenBMrZd8GTnddNNN9Q9Whn43KDHKX6DEqAw/++yzjSIaSozeI3eFYhxfqdguSmq/ZonR3Vbu0hgeDPPuhzL14JhaL9f7asAIJbF6SJmSsKHEZXH8crGf1H6LFhm9z5Udvop6cEytF7Pfe++9yrWzZQLINtts43yUGP1GudeO+YbK4iS13zHmGvkDlR0+Zrky9eCYWi9p8WApK4K46dOnqzx1qyhxDDf2lYqtrEnt1ywxutuuwWMx9WBMXVTrPeGEE9zLL7/sZNkYbd19grIVlDgWJ6n9jq5jx2pe+aE1lqAdZ+UMWtlQWH/99UOHc5LZFqEVlDgWB5ntUECTPoXRtEDXWvh2zIkWPCEp+rZjtRS3igVyLXzCg6uYMF0zSBbIOXzoLcggVSaVNVmgmQVqf0vTLMN0PllgPC2QHH48rZ/y7rkFand41INBCOxv8eLFHVVKmJiOri+7uJtpl+Wbzo2fBWp3eNSDDzroICfLvbuFCxe6o446ym277baV1cE+8YlPdM06daXto8jtFraTa/286krHT9P2u5m25dGzraFxkqHtdrSFiHzrW9/aSEMGlDKR3ssEG86uueaaTEY7s8033zy76aabMsF6M9GdzERJLBNEQK8hvmALmeC+2cyZMzN5n6/Hv/a1r2UyMKX7y5cvz7beeutcfNJ53/vel8kPLhPpvUxW9tY0Nttss2zBggW5uMW09eQr/wRpyB555JHs5z//uZbrwQcfzGQQLROwLZPXp9k+++yTycrcmUh5Z6KPo1dRfnmlmh199NHRvP26izZmI0v/Wj/Ot7/97WAdhDNqWoY5c+Zke+65Zwa1uuGGG2ai5al1kZXMMxGnjZaR67A9NsNGgm2PqV+j4AO60/DyOh1ewK5M5K+zq666Sh18rbXWyoRtz3ysly9XBqayhx56SB152rRpakKRyM5k8YRM7g6ZKAnrj4UTIrmdHXDAARpHRmr1C+HD9ddfn4nEdibLxWeHHHKIxolhyLG0NdFX/uEoYMfSJQPcUWcRUjPbYostshiu7KPIsbz9uku3r5Glf60f59Zbb80EtdCG4sQTT9QfGTZspQznnHNOJqPV6tz8QGVBCt2XBaKzvfbaK4uVketoOOTurPXeYIMNtJx+GRsFH9Cd3GvJdm4rZXjwiy++2Fj0V1oYRXtN/9GwXhZNYJKGOJL+sQgxK10zgUO+JLfddts5+aG4+fPnlxZLWkKV5YbeZO3VZcuWKWUprbDmC6bAcwULEbeS9o477qhdMcoCssy6rmAPqJzFcGUfRS5DoK3ufoX8azlucegKhurQShlIh8kw1AXoThoD3ccOX/7yl/X5KpQ215laM6Pkp59+OoeUZEUR2sSz9OCA/qvch4/hwdhh9dVX12VvWPrmkksucTKho2Eew3pjqC4Qma3KjTOEBsOku9FIjy9ztdVW08/gDHyRMQy5lbQ/+tGPuu9973vugQcecNKy6j4E584771yKK1uBYnlz3upucUNbixNLp1UcWrovmjwNjY9eczCWNudCas0cH5ZQ2eFjeHCrhvnwhz/smJDBND703idNmuT4EdGqchy0V7ordLk0Sb40ZLItvuWz0047uVtuuUWPS1dEScsYhhxL29JiC7GJ/DYTUJDkBnfgB0DL2wqKHMvbz6O4X8SYOR9Lp5UyFNMvfo6lXYznfw6V0T8/KPuVHT5WQVrkUKts8e1cDNWlO8PkD2YxSZ/W0WoTaHlxRHkg1hVA7IcAHgyLQ/zLLrvMHXfccS6GIcfStrLZVh5+deYVn+Xh2MlDnANEawVFjuVNWlZ39i0UMWaLE0unlTJY2qEt6cfSLsa3shTLWIw3SJ/HHR6jj8kKIT4SjAFpzaEare9vRqU7Y7d9O8b2Zz/7mU4H9I8VMWQ7F0vbzpdtW0WRY3mH0o5hzMQNpdNqGUJ5+cdCafvn/f2yMvrx+n1/3B2+3w2UyjdcFqi9SzNc5km1GTYL5F5LWp9t2CqZ6pMsYBbIObw9CNrJtE0WGDYLpC7NsH2jqT6lFkgOX2qedHLYLFC7wzMkz/C6PxqK0YQNcehOxgKvCiEtU0gW6KYFanf4u+66y+29997uggsuaJSb4f9dd921wWY0Tng7CKgKEekd6Z9dH4/199stYSfX+nnVlY6fpu13M23LY1y3Br1JIWy3oy2UpMhwKE5rCcmyknpMRkuzGN4KDXnYYYcpuiqjidkRRxwRxVhjqDC0ofzYlLYUTCGTVb6tCI1tp/ivj/PGMGQf800ocMP0fbHT8PI6HV5IR1UAhk0nCLmXzZo1K8PhY3irObxMHlHWXYjLKMYaQ4VjKsS+pTvFf31UNobZ+phvQoF964//fu61ZDu3mjI8mHSQzJOWzn3qU59SPFcmfSimGsNbuYZuDUTjD37wAycMdynGSvxQCKkQC9fdiNop/uvjvAkFbph1YHYq9+HL8GBqDxyGQ0gXR8Evs0gZ3gpPA/146qmnavQyjNXSKz4c+yisqRBbXLad4r9+WmXlC/E+/rXsW5xYOmW28tNKKLBvjfL9yg7fDA9mAgJ048UXX+yIa6EMb91ll13cpZde6pjUAYMew1hjqDB5hFSILW+2neK/pGGobKx8xIkFu9Y/H0unzFahdPw0bT+Wtp0PbVtNO3Rtvx+r7PCxioEnGKKAscFqURS2YzG8lfSIAyFJN+jQQw+NYqwxVJg0QirEHPdDJ/ivj8qWYbZWXz9f/1qOW5xYOjFbFdPx8/D3ST+Wth+PfStLq2kXrx+Uz+NCS8bw1pjRYhhrERUuUyGOpV08Hiubj8f6+1wfK18xbT4Xr/XjhNJppTx+GrH9UNqxuGVljF0zKMfHxeG7ZRxzeOZtppAsELLAUDl8UiEOfcXpmG+B3GtJ68f5EdJ+ssAwWSDn8DIsMEx1S3VJFhhjgdrf0ozJIR1IFugjCySH76MvIxWl+xboisPTNbrzzjvd7Nmz3dKlS7tfiz7PAY0btG1SGH8L1O7wv/71r3UxMyEVVV8GhV4RNtWaDj166n2ffl1/+MMfqpKZdzrtjpcFjF+T/G23o60Mh2eIp77wwguaDoq7olGoqG8raG0MH26miOsXGgViVIpRMRb9ykzkuvV0DOeNqebG4vv4780339xUzRdRWUhOVIhDismx/P06pf16LNDw8rocnmIh0yz6kpkMa2fC0mSgvoRW0NoYPtxMEVczeOWfUJeq9vv9739fHZ+yEGI4b0w1Nxbfx38vv/xyxZ8FAMu+8pWvqGQ3efl1JX0Yf5SOZeh+jGJyLH/SSaFeC+ReS7ZzlynDg0UC2914442qEfmFL3zBzZ0714nz5FRoY2htGT5cpojrlx2tShZlAElGEJXphYRYnpwLqeaWxTeVX0ArZP6uvPJKVSt+9tlnSS5XVz0g/9DMDCkmcz6Uv12XtvVZoHIfPoYHf/Ob33TSSqu09NVXX+2efPJJt2TJEvfYY4/lSl0FiS3DYP3EmR/7ute9Tg+hRAy1SYjlybmQam5ZfEN7W0V4ySOmmBzLn+Mp1GuByg4fw4PXW289JzOSHA9tBDTJCaYRaehpDFstQ2I1oRb+ybOCtqY4OncXyzuWZyzJVuKXldfqaunHFJPtfNp23wJUOW4tAAAMyUlEQVSVHT5WNFDgrbbaSidt43hITsvKE26TTTbRtzeoGsjSMFFsNYbExvLjeBGJoBv13HPPqVa8PBA2Lm0XlS2Lb3nGyhvCbGOKyY0CvrJjaRePp8+dW6Br8Ji8kdBVN3B0/wssoqchbDWGxLZaXVl7SaOyagf9eZlE7mStpsbloTwbJwM7zeLHylusqyUdU0y282nbPQt0zeG7V+TmKfO8wNxWuldo3dDKf/zjH29+YYox9BYYSofnW2OlEB6WWWNKVhEc+i8yVbA1C+ReS/pdj9YuT7GSBQbLAjmHl1f8g1X6VNpkgTYtUPtbmjbzT9GTBXpqgeTwPTV3ymy8LdA1h2cRYFnyXF8L8gBZNQiTUvXSxnWG5/74xz92sqx843hop478Qum2c8zK2841KW5rFqjd4XFudGOEDlTJbPiRqVOnuocffri1EhVigRd3GgzP/da3vuVggMpCHfkV0/dR4eK50Gcrb+hcOtaZBWp3eCSvGU0FIMO5LrroIl2NmwkQyO6xVuqUKVOcqAPrj4Lisy6rqPpqTRiw4p0567GCLyCXR+CB+nOf+5weFxpTQS094f1joIeRTzgXXkfaABTiTAwehUK7+QlqrJM5gMTIg3qSL2u5ChXqBP916667ri79zl2OADDGD/+YY46J1sO3zR133KHlveKKK9QW1Bd7nH/++ZoeZcaOjDPst99+DmGpFFqzQO0Oj2NPmzYtlztO8slPflKH+1ldmy8JuTwcnYDDoCFPuPbaa93ixYsdEBoA2IorrqjH77vvPl2QGIJxjz32UO1Kg8I0gvzDqQQJ1hWz0agHMSCwhisgWCi0mx8wGhr4CxcuVDCOmV0s9MACyhCiCMGybP3+++/f0MiELZo8ebI7+eSTXaweoBC+bSjvL37xC4fz02gcfvjhTvBjrQJ3Ku4CSBLSIKRZZaFvNnysssPTes+YMUNbLz9pHPSll17yD+X2Das1mjF3Uj7QKtJ6ieS2or12vgzVtTg+fnvWWWfpj4i7TVloNz/UhxctWqS4M6O57PMDAGMAbea5BVRYJoa4ECpcVo+QbQwbPvjgg1VGkLoU8eey+qVzeQtUdvgYHkxLRivmBwRVcQ6CYbX+eV8BmO4HrSiBu4ANhpWhupZWGX5rcYrbdvPrVH24rB4h24Sw5Rj+XKxb+jzWApUdPoYHz5w5U7su9EkJP/3pT/VWfMABB+hn+xdTAOaHQUsNWsst3gbDWkF1q+C37ebXqfpwK/UwG8W2Mfw5Fj8df9UClR3+1STye8z+4cEVfcd11llH1YC32GILvd0T01rsmAIw3Rn69qQjU+m0b8x1Zagu5wmt4LeW/x+ucNp9aje/TtSHy+pRLJuV0bZ2PoY/W7y0LbGAzRiUKLZb21beJ2e/+tWvStOTB8rg+aeeeiqTV5xjzsm79Ey6IWOO+wdaydePz34n+flpiWpDtnz5cj1E3eV5RvflzUpjnwOt1EMvDPxjojx/pHnDDTdkW265ZSBWOhSywNDSkiW/8YE/lfDn6l9hcvjqthvXKxP+XM38OVrS+ojVkkpXJQv0vwVyDi99nv4vcSphskAHFqj9LU0HZUmXJgt03QLJ4btu4pRBP1kg16Wps2AM2UNIAj7tvPPODSam3TzAdRmy7ySA28orQieSe8qnoGDWaWByOAE5Dj8U82KAjbwhRqsES6/q9VXyHOZram/hBx0P5stuBedFXc1Gk30HMbTXUGT77Mdptu/nX+X6ZumP8vnaHX7Q8GDQh912282tueaaCoBBVvo4Lz/gEPKL00BMrr/++joqjMIZoYgi22dWGAQntr9jjz1WlRVCafv52/Ux9DmGEGth0r8xFqjd4QcND+YHusoqq6h+zRprrOHQivRxXnTuQ8gvlqS7IdLZ7rDDDmugwEUU2T4TB1LyS1/6krLu8P+xtP387foY+hxDiMd80+mAWqByHz6mHgwejBJXLBgCC74bCj6uKzrzbv78+RothtXSYlrw8WD4nbXXXlsno9j50BYQC2YHJ2OCCZwMDPrKK6/sVl11VX1+oPUvqgOTFugu9eFPJK/1BxDKg2NwRfygmKAya9YsRasB5EJp0++3/C29sroZQszzhGjx2yVpG7BA5RZ+WPBgZlfRygOQTZ8+3V166aU5M5WpA+PEFnDQZuMYhx56qJswYYKCdVxXlrala9sy9DmEENt1aZu3QGWHHxY8GKFX7kjMsGLaITOYCKb8W6YOzMQPZl2hfc+MJ9DhWGBmFTOieOtks7jK0rb8Lb0q6LNdm7avWqCyw7+aRH5v0PDgvfbaS9WMmWdLf5wfgK/8G1MHBsPgdSMPrTvttJOj9S4GH9U4++yzdQYUizTQZWG2WCxtP39LsxX0mbh+nnZt2r5qga7CY62o5DLbKTTTh1k9b3/72xva7lbkZkq+xGslX0uPLS08D6CMGZjD8FaE1pupiDF1YK6l/CuttJI6MZ/bDbG0/fz9NNutm39t2pcGAWYYQ/BFN+uDJoMlCwy6BWrv0gy6QVL5h9sCudeSdjsf7iqn2o2yBXIOn7o0o+wKo1H31KUZje851fIVCySHT64wUhbIdWnqqPmCBQtUB5Ehewuoc8GrsM4qw/YWeOWHLB/D66h0sfDYLrvsYqfdo48+qoM0m266aeNY2kkW6MQCtb+WNEm4m266ScvFcwGin0jQsaIfknToTBIQH4WRQZls0qRJDrEm05jk/Kc+9Skdwbzgggv4mEKyQMcWqL1Lc+CBBzpa+d/85jdaOIbqX3755cYkDlr62bNn6x8MC+u4Aoy1Gkztl8Eq1oR94oknVIw1pBoMOovwEaObG220kWO0E9iMAS3Kdd5557mPfexj+kNkhHjevHlaDF/JFx3HkGpxCCsOHYthvX4et9xyS6vVT/E6tEDtDr/11lsrpWgLC8CQoDJm4qk4LIAWf8cff7xyKDhdqwHF3GXLlqlkNTgAlGIZOgvvggqaiBUpwsvdhPKgyAswds8996ijn3HGGaruSzl8JV+4FyZzcCfyVYtDWHHoWKxsfh6oK6fQGwtU7sPH8GCKDVgG640sNq2XTY7gHA6PExCY3AAkhXiqKHbpseI/A63sOA4PVwKtCWKM8+O8TJoACS5iwUznQ/GXIXm6S+z/8pe/1B8BrT7akjwj8IeEHYgBwTDmo446StHeu+++202cOFGvFZUyF8KKAdCKqDF3vFjZLA+rW9p23wKVW/gYHkyR+ZIhCGHZoQhxMgs4DS0mf0xlo3tBfx8HokvAD8LCM8884zbYYAP7qFucdrXVVtN90gbRLUNneVAmrLDCCkHeBZDLAlrrtPoE43tiar8hrDh0rKxslofln7bdt0Blh4/hwRQZJwPEOvLIIxt991hVmJzM7Z3JGjgyxCLB5nLSbfEDZCJ3DabenXvuubqqRifo7L333quQGFu6XbYSieUZU/sNYcWhY52UzcqQtvVZoHKXplkReIjEAQ466KDSqLSwrKBBoA/M3eG0007THwFdIm77fiBdXm3ykEkLL2KiOpOIOwaoLvQiD8XNgmEUdKvg2EF9zzzzzAYtaed56OUBkzsRcXiA5YcBVkyLzqocEI88INM9Kx4j7VjZLI9mZU3n67NA7a8lOy0aSC79cpy3bMl45nry/t4P7aKzTKzmDRIPzzgx3a1YCGHJIaw4dIw02y1brBzpeGcW6DuH76w67V1tDs9bnBRGwwIj7fA2VsDYQAqjYYFcHz71KUfjSx/lWuYcPuHBo+wKo1H3yq8lR8M8qZbDZoHk8MP2jab6lFqgdocHHHvyySdzmYIHs7p2J4Ehf2T8UkgW6MQCtTs8sNjRRx/dKBPPBcBhsO3thqSi267FUvxmFqjd4cvwYJw/hNrGFHBDKrqxuDA4rPEKe88ILRqRKSQLFC1Qu8OX4cEsSc8wexG1jSnghlR0Y3FJF/4GmhIIbOnSpcW6ps/JAq6yw4MHIxdHK1wMhgeDCQB6MZOJEFMA5pwp4NqMKY6FVHRjcZmoAbfDZI8TTzyRaCkkC4yxQGWHr4IHx1BbStWOAm4oLtJ8NskEvp4fWwrJAkULVHb4KnhwDLUtFsr/XFTR9c/5+/D0TCzB0ZlwAv+eQrJA0QJd8wow3uXLl+fwYFBb8AVQ26222krVc61V9gtmiENIRdePx77FZbYSXD0TQubMmVOMlj4nC6gFxgUeC6G2se8jpqJbjH/bbbfpIVQR6M/DtiP7kUKygG+BHEvjn+jmPqx7q4EVM1oJ6667rs5P5bUkS0qmVr4Vq41enHFp4btlZqb9LVmyRKU4yiaPdCv/lG7/WyDXwlt/uP+LnUqYLFDNAjmHT3hwNSOmqwbHAl17SzM4JkglHSULJIcfpW871dX9P1yVOh+iyJZVAAAAAElFTkSuQmCC)
# 
# Of these, Brandt's baseline case shows emissions from the "Well and downhole pump", "HC gas reinjection compressor", "HC gas injection wells", and "Separation" process stages. Thus, these calculations are replicated below.
# 
# The basis of this calculation is to take the weighted sum (in terms of Global Warming Potential, i.e. ensuring CO2-equivalence) of emissions in each component associated with the "well & downhole pump" part of the system. These components are:
# 
# 
# * Fugitives - Completions
#     * OPGEE labels this as fugitive, but logic actually classifies these emissions as vents
#     * "Total gas lost during completions divided by gas production"
#     * OPGEE uses a database of tonnes of fugitive emissions per activity in terms of CO2 and CH4. For Geologic H2, Brandt assumes that total mass of emissions per activity are consistent with the database values, but that the composition of the emissions are aligned with the assumptions regarding the reservoir composition (i.e. high H2, low CH4 and CO2).
# * Fugitives - Workovers
#     * "Total gas lost during workovers divided by gas production"
#     * Assumes total gas for all workovers over field life and then calculates a loss rate as this total divided by total gas production. Loss is apportioned per reservoir composition.
# * Pneumatic controllers
#     * 0.0000%
#     * Loss rate calculated in same manner as wellhead, above.
# * Liquids unloadings
#     * This loss rate is dependent on gas rate, so needs to be calculated for each year of field life.
# * On-site tanks - vents
#     * 0.000% - No tanks in design.
# 
# 

# In[228]:


# The OPGEE calculations rely on conservation of mass throughout the process flow and calculate vented or fugitive emissions as a proportion of the mass flow of each component of the gas stream.
# Brandt's cases are based on volume flow rates downstream of the production separator, so we need to convert these to mass flow rates to calculate emissions.
# We will later also need to calculate the effective mass flows upstream of the separator (i.e. at the wellhead)

# First define a function to calculate the mass flow rates of each component of the gas stream downstream of the separator. 

def calculate_mass_flows_after_separator(case, sensitivity_variables=None):
    """
    Calculate the mass flows of each component of the gas stream downstream of the separator for a given case and sensitivity values.

    Parameters:
    case (str): The case for which to calculate the mass flows.
    sensitivity_variables (dict): A dictionary containing sensitivity values for the parameters. The keys are the parameter names and the values are the sensitivity values.

    Returns:
    dict: A dictionary containing the mass flows of each component of the gas stream downstream of the separator in kg/day.
    """
    if sensitivity_variables:
        # Extract the sensitivity values
        oil_production = sensitivity_variables['oil_production']
    else:
        # Extract the default values
        oil_production = oil_production_default

    CH4_after_separator = (
        gas_composition_df.loc[case, 'CH4'] / 100 *
        oil_production *
        production_profile_df[f'{case} GOR, SCF/BBL'] / 1E6 *
        gas_densities.loc['CH4', 'Density tonne/MMSCF']
    )
    H2_after_separator = (
        gas_composition_df.loc[case, 'H2'] / 100 *
        oil_production *
        production_profile_df[f'{case} GOR, SCF/BBL'] / 1E6 *
        gas_densities.loc['H2', 'Density tonne/MMSCF']
    )
    N2_after_separator = ( 
        (gas_composition_df.loc[case, 'N2'] + gas_composition_df.loc[case, 'CO2'] + gas_composition_df.loc[case, 'Ar/oth inert'])  / 100 * #N2 stream is considered total of N2, CO2, and Ar/other inert gases.
        oil_production *
        production_profile_df[f'{case} GOR, SCF/BBL'] / 1E6 *
        gas_densities.loc['N2', 'Density tonne/MMSCF']
    )
    CO2_after_separator = (
        gas_composition_df.loc[case, 'CO2'] / 100 *
        oil_production *
        production_profile_df[f'{case} GOR, SCF/BBL'] / 1E6 *
        gas_densities.loc['CO2', 'Density tonne/MMSCF']
    )

    H2O_after_separator = (
        0.014 * (CH4_after_separator + H2_after_separator + N2_after_separator + CO2_after_separator) #Assume 1.4% water content in the gas stream. This is a modification of OPGEE 3.0a in the Brandt paper.
    )
    total_mass_flow_after_separator = CH4_after_separator + H2_after_separator + N2_after_separator + CO2_after_separator + H2O_after_separator
    
    return {
        'case': case,
        'CH4_after_separator': CH4_after_separator,
        'H2_after_separator': H2_after_separator,
        'N2_after_separator': N2_after_separator,
        'CO2_after_separator': CO2_after_separator,
        'H2O_after_separator': H2O_after_separator,
        'total_mass_flow_after_separator': total_mass_flow_after_separator
    }


# The calculations below require the determination of loss rates for the following:
# * Production Vent Completions
# * Production Vent Workovers
# * Liquids Unloadings
# 
# So, before proceeding to the calculate the emissions associated with these activities, we will first define functions to calculate the loss rates according to the method in Brandt/OPGEE.

# #### 2.1.1.1 Calculate component-level emissions factors
# 
# The OPGEE operating manual describes the calculation of emissions factors, broken down into "productivity tranches" (i.e. continuous range of observed gas flow rates is 'binned' into discrete sub-ranges). This calculation logic is replicated here:

# In[229]:


# First extract the average emissions factors for each equipment type and tranche from an excel file extracted from the OPGEE model:

# Load the Excel file
file_path = 'https://raw.githubusercontent.com/tblackfd/Thesis/main/MeanEmissionsFactorsByTranche.xlsx'

# Load data from both sheets
sheet_names = pd.ExcelFile(file_path).sheet_names
data_sheet1 = pd.read_excel(file_path, sheet_name=sheet_names[0])
data_sheet2 = pd.read_excel(file_path, sheet_name=sheet_names[1])

# Rename columns for better clarity and drop the redundant row
data_sheet1.columns = ['Equipment No', 'Equipment Type'] + [f'Tranche {i+1}' for i in range(data_sheet1.shape[1] - 2)]
data_sheet1 = data_sheet1.drop(0)

# Rename columns for sheet2
data_sheet2.columns = ['Detail'] + [f'Tranche {i+1}' for i in range(data_sheet2.shape[1] - 1)]

# Set index for better alignment
data_sheet1.set_index(['Equipment Type'], inplace=True)
data_sheet2.set_index('Detail', inplace=True)

emissions_factors_data_df = data_sheet1
productivity_tranche_data_df = data_sheet2

# Now define a function to calculate the emissions factors for each equipment type and tranche
def calculate_emissions_factors(equipment_type, case, sensitivity_variables=None, injection_well_flow_rate=None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
        length_of_field_life = sensitivity_variables.get('length_of_field_life', field_lifespan_default)
    else:
        oil_production = oil_production_default
        number_production_wells = number_production_wells_default
        GWP_H2 = GWP_H2_default
        length_of_field_life = field_lifespan_default

    if injection_well_flow_rate is not None:  # Special case for calculating emissions factors for injection wells
        loss_rates = []
        for flow_rate in injection_well_flow_rate:
            tranche_num = None
            for i in range(1, productivity_tranche_data_df.shape[1]):  # Look through each tranche in the DataFrame
                if flow_rate < productivity_tranche_data_df[f'Tranche {i}']['High Bound']:  # High Bound check
                    tranche_num = i
                    break
            if tranche_num is None:  # If the flow rate per well is greater than the highest value in the DataFrame
                tranche_num = productivity_tranche_data_df.shape[1]
            emissions_factor = emissions_factors_data_df.loc[equipment_type, f'Tranche {tranche_num}']
            loss_rate = emissions_factor
            loss_rates.append(loss_rate)

        # Convert the loss_rates list to a DataFrame for better readability
        loss_rates_df = pd.DataFrame(loss_rates, columns=['Loss Rate'])
        loss_rates_df.index = loss_rates_df.index + 1  # Start the index from 1
        loss_rates_df.index.name = 'Year'
    
    else: #Normal case for for all other equipment types
        total_production_rate = production_profile_df[f'{case} GOR, SCF/BBL'].values/1000 # MSCF/BBL
        productivity_per_well = oil_production * total_production_rate / number_production_wells # MSCF/well/day

    # Calculate the flow rate per well
    # total_production_rate = production_profile_df[f'{case} GOR, SCF/BBL'].values/1000 # MSCF/BBL
    # productivity_per_well = oil_production * total_production_rate / number_production_wells # MSCF/well/day

    # Calculate the 'mean gas rates' for each tranche in each year of operation
        mean_gas_rates = []
        for year in range(len(productivity_per_well)):
            tranche_gas_rates = []
            for tranche in range(1, 11):  # Assuming 10 tranches (1 to 10)
                normalized_gas_rate = productivity_tranche_data_df.loc['Normalised Gas Rate', f'Tranche {tranche}']
                tranche_gas_rate = productivity_per_well[year] * normalized_gas_rate
                tranche_gas_rates.append(tranche_gas_rate)
            mean_gas_rates.append(tranche_gas_rates)

        # Convert the mean gas rates to a DataFrame for better readability
        mean_gas_rates_df = pd.DataFrame(mean_gas_rates, columns=[f'Bin {i}' for i in range(1, 11)])
        mean_gas_rates_df.index = mean_gas_rates_df.index + 1  # Start the index from 1
        mean_gas_rates_df.index.name = 'Year'

        # For each of the gas flows associated with each of the "Bins" in each of the years, determine which tranche they fall into
        bin_assignments = []
        for year in range(mean_gas_rates_df.shape[0]):
            for bin in range(mean_gas_rates_df.shape[1]):
                flow = mean_gas_rates_df.iloc[year, bin]
                tranche_num = None
                for i in range(1, productivity_tranche_data_df.shape[1]):  # Look through each tranche in the DataFrame
                    if flow < productivity_tranche_data_df[f'Tranche {i}']['High Bound']:  # High Bound check
                        tranche_num = i
                        break
                if tranche_num is None:  # If the flow rate per well is greater than the highest value in the DataFrame
                    tranche_num = productivity_tranche_data_df.shape[1]
                bin_assignments.append({'Year': year + 1, 'Bin': f'Bin {bin + 1}', 'Tranche': tranche_num})

        # Convert the bin assignments to a DataFrame
        bin_assignments_df = pd.DataFrame(bin_assignments)

        #Now that we have the tranche for each Bin, we can look up the emissions factor for each row of the series in emissions_factors_data_df and add it as a new column to the bin_assignments_df DataFrame.
        emissions_factors = []
        for index, row in bin_assignments_df.iterrows():
            tranche = row['Tranche']
            emissions_factor = emissions_factors_data_df.loc[equipment_type, f'Tranche {tranche}']
            emissions_factors.append(emissions_factor)

        # Add the emissions factors to the DataFrame
        bin_assignments_df['Emissions Factor, kg CO2e/MSCF'] = emissions_factors
        
        # Finally, calculate the loss rates by calculating the sum of the product of the emissions factors by the Fraction of Total Gas in each tranche, taken from the productivity_tranche_data_df DataFrame.
        # This should result in a series of values, one for each year of the field life.

        loss_rates = []
        for year in range(mean_gas_rates_df.shape[0]):
            loss_rate = 0
            for tranche in range(1, 11):
                fraction_total_gas = productivity_tranche_data_df.loc['Frac Total Gas', f'Tranche {tranche}']
                # Retrieve the emission factor for the specific year and bin
                emission_factor = bin_assignments_df.loc[(bin_assignments_df['Year'] == year + 1) & (bin_assignments_df['Bin'] == f'Bin {tranche}'), 'Emissions Factor, kg CO2e/MSCF'].values[0]
                loss_rate += fraction_total_gas * emission_factor
            loss_rates.append(loss_rate)

        # Convert the loss_rates list to a DataFrame for better readability
        loss_rates_df = pd.DataFrame(loss_rates, columns=['Loss Rate'])
        loss_rates_df.index = loss_rates_df.index + 1  # Start the index from 1
        loss_rates_df.index.name = 'Year'

    # # Print the loss rates DataFrame
    # print(loss_rates_df)
      
    return {
        'case': case,
        'equipment_type': equipment_type,
        # 'productivity_per_well': productivity_per_well,
        # 'emissions_factors': emissions_factors,
        # 'mean_gas_rates_df': mean_gas_rates_df,
        # 'bin_assignments_df': bin_assignments_df,
        # 'mean_gas_rates': mean_gas_rates,
        # 'loss_rates': loss_rates,
        'loss_rates_df': loss_rates_df
    }

# Test the function
test = calculate_emissions_factors('Well', 'Baseline')['loss_rates_df']
test
#Print all 10 of the year 30 rows from the test dataframe
# print(test['bin_assignments_df'].loc[test['bin_assignments_df']['Year'] == 30])


# In[230]:


#Brandt's model shows that the loss rate at the well head increases as production decreases. These rates are extracted from Brandt's model (and calculated as per Brandt/OPGEE logic) as follows later below.

#Loss rates for this section are all based on flow rates downstream of the separator, so first calculate these rates. Create variables for all gases (CH4, H2, N2, CO2, H2O) downstream of the separator
#because these are required later, even if they are not considered for vented/fugitive emissions.

def calculate_total_production_vent_emissions(case, sensitivity_variables =None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
    else:
        oil_production = oil_production_default
        GWP_H2 = GWP_H2_default
        number_production_wells = number_production_wells_default
        field_lifespan = field_lifespan_default

    #Calculate the mass flows of each component of the gas stream downstream of the separator.
    mass_flows = calculate_mass_flows_after_separator(case, sensitivity_variables)
    CH4_after_separator = mass_flows['CH4_after_separator']
    H2_after_separator = mass_flows['H2_after_separator']
    N2_after_separator = mass_flows['N2_after_separator']
    CO2_after_separator = mass_flows['CO2_after_separator']
    H2O_after_separator = mass_flows['H2O_after_separator']

    #Calculate the mass flows of each component of the gas stream upstream of the separator. This is required to calculate the vented/fugitive emissions.
    mass_flows_upstream_separator = calculate_mass_flows_upstream_separator(case, sensitivity_variables)
    N2_upstream_separator = mass_flows_upstream_separator['N2_upstream_separator'].values #tonnes/day
    CH4_upstream_separator = mass_flows_upstream_separator['CH4_upstream_separator'].values #tonnes/day
    H2_upstream_separator = mass_flows_upstream_separator['H2_upstream_separator'].values #tonnes/day
    H2O_upstream_separator = mass_flows_upstream_separator['H2O_upstream_separator'].values #tonnes/day
    total_gas_mass_flow_upstream_separator = mass_flows_upstream_separator['total_gas_mass_flow_upstream_separator'] #tonnes/day
    
    #First consider venting associated with completion flowbacks. Calculate CH4 and H2 separately, then calculate their sum.
    # This requires replicating the calculations in Brandt's model for the loss rate associated with completions.
    # The OPGEE method is to use historical mean fugitive emissions of CH4 and CO2 from each activity (i.e. completion or workover) and then apportion this over the life of the field.
    # In the case of Geologic Hydrogen, it is assumed the total mass flow of fugitives is equal to the historic mean, however, the mass is adjusted to be comprised of the relevant gas composition.
    # The assumption is any gas reinjected into the subsurface will be into empty/saline aquifers, so the injection wells will not have any associated fugitive emissions. 
    # Further note that all wells are assumed vertical, with no fracturing, and no flaring of associated gases.

    completions_historic_mean_emissions_CO2 = 0.170402937817918 #tonnes/event
    completions_historic_mean_emissions_CH4 = 3.78671071910546 #tonnes/event
    completions_historic_mean_emissions_total = completions_historic_mean_emissions_CO2 + completions_historic_mean_emissions_CH4 #tonnes/event
    
    # Convert these values to MMSCF/event
    completions_historic_mean_emissions_CO2_MMSCF = completions_historic_mean_emissions_CO2 / gas_densities.loc['CO2', 'Density tonne/MMSCF'] #MMSCF/event
    completions_historic_mean_emissions_CH4_MMSCF = completions_historic_mean_emissions_CH4 / gas_densities.loc['CH4', 'Density tonne/MMSCF'] #MMSCF/event
    completions_historic_mean_emissions_total_MMSCF = completions_historic_mean_emissions_CO2_MMSCF + completions_historic_mean_emissions_CH4_MMSCF #MMSCF/event. Is this addition valid

    # Apportion these historic emissions according to the gas composition of the case at hand:
    completions_emissions_historic_apportioned_H2 = completions_historic_mean_emissions_total_MMSCF * gas_composition_df.loc[case, 'H2'] / 100 #MMSCF/event
    completions_emissions_historic_apportioned_N2 = completions_historic_mean_emissions_total_MMSCF * gas_composition_df.loc[case, 'N2'] / 100 #MMSCF/event
    completions_emissions_historic_apportioned_CO2 = completions_historic_mean_emissions_CO2_MMSCF * gas_composition_df.loc[case, 'CO2']  / 100 #MMSCF/event
    completions_fugitive_historic_apportioned_CH4 = completions_historic_mean_emissions_CH4_MMSCF * gas_composition_df.loc[case, 'CH4']  / 100 #MMSCF/event

    # Convert these volume flow values to mass flow values:
    completions_emissions_historic_apportioned_H2_mass = completions_emissions_historic_apportioned_H2 * gas_densities.loc['H2', 'Density tonne/MMSCF'] #tonnes/event
    completions_emissions_historic_apportioned_N2_mass = completions_emissions_historic_apportioned_N2 * gas_densities.loc['N2', 'Density tonne/MMSCF'] #tonnes/event
    completions_emissions_historic_apportioned_CO2_mass = completions_emissions_historic_apportioned_CO2 * gas_densities.loc['CO2', 'Density tonne/MMSCF'] #tonnes/event
    completions_emissions_historic_apportioned_CH4_mass = completions_fugitive_historic_apportioned_CH4 * gas_densities.loc['CH4', 'Density tonne/MMSCF'] #tonnes/event

    total_completion_emissions_H2 = completions_emissions_historic_apportioned_H2_mass * number_production_wells #tonnes H2
    total_completion_emissions_CH4 = completions_emissions_historic_apportioned_CH4_mass * number_production_wells #tonnes CH4
    total_completion_emissions_CO2 = completions_emissions_historic_apportioned_CO2_mass * number_production_wells #tonnes CO2

    total_molar_flow_upstream_separator = (
        (N2_upstream_separator / molecular_weights_gases.loc['N2', 'Molecular Weight (g/mol)'] +
         CH4_upstream_separator / molecular_weights_gases.loc['CH4', 'Molecular Weight (g/mol)'] +
         H2_upstream_separator / molecular_weights_gases.loc['H2', 'Molecular Weight (g/mol)'] +
         H2O_upstream_separator / molecular_weights_gases.loc['H2O', 'Molecular Weight (g/mol)']) * 1E6  # mol/day
    )

    mol_frac_N2 = N2_upstream_separator * 1E6 / molecular_weights_gases.loc['N2', 'Molecular Weight (g/mol)'] / total_molar_flow_upstream_separator
    mol_frac_CH4 = CH4_upstream_separator * 1E6 / molecular_weights_gases.loc['CH4', 'Molecular Weight (g/mol)'] / total_molar_flow_upstream_separator
    mol_frac_H2 = H2_upstream_separator * 1E6 / molecular_weights_gases.loc['H2', 'Molecular Weight (g/mol)'] / total_molar_flow_upstream_separator
    mol_frac_H2O = H2O_upstream_separator * 1E6 / molecular_weights_gases.loc['H2O', 'Molecular Weight (g/mol)'] / total_molar_flow_upstream_separator

    MW_gas_upstream_separator = (mol_frac_N2 * molecular_weights_gases.loc['N2', 'Molecular Weight (g/mol)'] +
                                    mol_frac_CH4 * molecular_weights_gases.loc['CH4', 'Molecular Weight (g/mol)'] +
                                    mol_frac_H2 * molecular_weights_gases.loc['H2', 'Molecular Weight (g/mol)'] +
                                    mol_frac_H2O * molecular_weights_gases.loc['H2O', 'Molecular Weight (g/mol)']) #g/mol
    
    field_cumulative_gass_production_mass = reservoir_df.loc[case, 'Raw Gas EUR, BCF'] * 1E9 * mol_per_SCF * MW_gas_upstream_separator / 1E6 #tonnes

    apportioned_completion_emissions_H2 = total_gas_mass_flow_upstream_separator / field_cumulative_gass_production_mass * total_completion_emissions_H2 #tonnes/day
    apportioned_completion_emissions_CH4 = total_gas_mass_flow_upstream_separator / field_cumulative_gass_production_mass * total_completion_emissions_CH4 #tonnes/day
    apportioned_completion_emissions_CO2 = total_gas_mass_flow_upstream_separator / field_cumulative_gass_production_mass * total_completion_emissions_CO2 #tonnes/day
    
    total_completion_flowback_emissions_per_day = apportioned_completion_emissions_H2 + apportioned_completion_emissions_CH4 + apportioned_completion_emissions_CO2  #tonnes/day
    
    production_vent_completions_loss_rate = total_completion_flowback_emissions_per_day * 1000 / (total_gas_mass_flow_upstream_separator* 1000) #fraction of total gas mass flow
    
    production_vent_completions_CH4 = production_vent_completions_loss_rate * GWP_CH4 * CH4_after_separator #Loss Rate * GWP * Gas Rate

    production_vent_completions_H2 = production_vent_completions_loss_rate * GWP_H2 * H2_after_separator #Loss Rate * GWP * Gas Rate

    production_vent_completions_emissions = production_vent_completions_CH4 + production_vent_completions_H2 #This gives a series of results. i.e. One per year of field life.

    
    #Now consider venting associated with workovers. Logic similar to above, but Brandt does not apportion the default OPGEE emissions as per the gas composition of the case. It is not clear if this is an appropriate assumption/decision.
    #First calculate the effective loss rate by assessing historical figures.
    #This requires determining the number of workovers:
    
    number_injection_wells = math.ceil(0.25*number_production_wells) # Brandt / OPGEE assumes that workovers are required on both production and injection wells, and that 25% of production wells are injection wells.
    total_number_wells = number_production_wells + number_injection_wells
    total_number_workovers = total_number_wells * number_well_workovers 

    workovers_historic_mean_emissions_CH4 = 0.0884749106401871 #tonnes/event

    total_workover_emissions_CH4 = workovers_historic_mean_emissions_CH4 * total_number_workovers #tonnes CH4

    total_workover_flowback_emissions_per_day = total_gas_mass_flow_upstream_separator / field_cumulative_gass_production_mass * total_workover_emissions_CH4 #tonnes/day
    
    production_vent_workover_loss_rate = total_workover_flowback_emissions_per_day * 1000 / (total_gas_mass_flow_upstream_separator* 1000) #fraction of total gas mass flow
    
    production_vent_workovers_CH4 = production_vent_workover_loss_rate * GWP_CH4 * CH4_after_separator

    production_vent_workovers_H2 = production_vent_workover_loss_rate * GWP_H2 * H2_after_separator

    production_vent_workovers_emissions = production_vent_workovers_CH4 + production_vent_workovers_H2

    #Now looking at the vents associated with Liquids Unloadings:

    #Liquids unloadings vary according to rate of production from the field, so baseline loss rates have been extracted from Brandt's model for each year of field life:

    # liquids_unloadings_percentages = np.array([
    #     0.022286, 0.035370, 0.045185, 0.045185, 0.047776, 0.069842, 0.075147, 0.075147, 0.075147,
    #     0.075147, 0.075147, 0.075147, 0.075147, 0.075147, 0.093191, 0.103238, 0.103238, 0.103238,
    #     0.103238, 0.114288, 0.114288, 0.114288, 0.114288, 0.114288, 0.114288, 0.114498, 0.114498,
    #     0.114498, 0.114498, 0.114498
    # ])
    # liquids_unloading_rates = liquids_unloadings_percentages / 100

    #Instead of taking results from Brandt's model, the following section replicates the underlying calculations.

    liquids_unloading_plunger_loss_rate = calculate_emissions_factors('LU-plunger', case, sensitivity_variables)['loss_rates_df']['Loss Rate']
    
    liquids_unloading_non_plunger_loss_rate = calculate_emissions_factors('LU-non plunger', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    liquids_unloading_rates = wells_LUp * liquids_unloading_plunger_loss_rate + wells_LUnp * liquids_unloading_non_plunger_loss_rate

    # Create the DataFrame
    df_liquids_unloadings = pd.DataFrame({
        'Year': range(1, 31),
        'Liquids Unloadings %': liquids_unloading_rates
    })

    production_vent_liquids_CH4 = df_liquids_unloadings['Liquids Unloadings %'].values * GWP_CH4 * CH4_after_separator 
    production_vent_liquids_H2 = df_liquids_unloadings['Liquids Unloadings %'].values * GWP_H2 * H2_after_separator

    production_vent_liquids_emissions = production_vent_liquids_CH4 + production_vent_liquids_H2

    total_production_vent_emissions = production_vent_completions_emissions + production_vent_workovers_emissions + production_vent_liquids_emissions #tCO2e/day

    return {
        'case': case,
        'total_production_vent_emissions': total_production_vent_emissions,
        'production_vent_completions_emissions': production_vent_completions_emissions,
        'production_vent_workovers_emissions': production_vent_workovers_emissions,
        'production_vent_liquids_emissions': production_vent_liquids_emissions,
        'CH4_after_separator': CH4_after_separator,
        'H2_after_separator': H2_after_separator,
        'N2_after_separator': N2_after_separator,
        'CO2_after_separator': CO2_after_separator,
        'H2O_after_separator': H2O_after_separator,
        'N2_upstream_separator': N2_upstream_separator,
        'production_vent_completions_loss_rate': production_vent_completions_loss_rate,
        'total_molar_flow_upstream_separator': total_molar_flow_upstream_separator,
        'production_vent_completions_CH4': production_vent_completions_CH4,
        'production_vent_workover_loss_rate': production_vent_workover_loss_rate,
    }
        
# # #Test the function:
# total_production_vent_emissions = calculate_total_production_vent_emissions('Baseline')
# print(f"Case: {total_production_vent_emissions['case']}, Total Production Vent Emissions: {total_production_vent_emissions['production_vent_workovers_emissions']}")

# # Print value of N2 after separator for the 'Baseline' case
# print(f"H2O after separator for the 'Baseline' case: {total_production_vent_emissions['H2O_after_separator']}")


# In[231]:


# Now we have determined the mass flows after the separator


# ### 2.1.2 Production & Extraction Fugitive
# 
# * Fugitives - Wellhead
#     * Calculated as fractional loss rate. This loss rate is calculated as the average of emissions of this component under different flowrates (divided into 10 "tranches"), weighted against the relative volume of gas produced in each flowrate tranche.
# * On-site meter
#     * Loss rate calculated in same manner as wellhead, above.
# * On-site dehydrator
#     * Loss rate calculated in same manner as wellhead, above.
# * On-site reciprocating compressor
#     * Loss rate calculated in same manner as wellhead, above.
# * On-site heater
#     * Loss rate calculated in same manner as wellhead, above.
# * On-site header
#     * 0.0000%
#     * Loss rate calculated in same manner as wellhead, above.
# * Chemical injection pumps
#     * Loss rate calculated in same manner as wellhead, above.
# * On-site tanks - leaks
#     * 0.000% - No tanks in design.

# #### 2.1.2.1 Use the component level emissions factors to calculate production fugitive emissions

# In[232]:


#Create a helper function to calculate fugitive emissions from the production phase of the field:
def calculate_total_production_fugitive_emissions(case, sensitivity_variables =None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
    else:
        oil_production = oil_production_default
        GWP_H2 = GWP_H2_default

    if case is None:
        raise ValueError("Case must be provided and cannot be None.")

    # First, retrieve the CH4 and H2 values after the separator from the vents calculation
    # Calculate the mass flows downstream of the separator
    mass_flows_after_separator = calculate_mass_flows_after_separator(case, sensitivity_variables)
    CH4_after_separator = mass_flows_after_separator['CH4_after_separator']
    H2_after_separator = mass_flows_after_separator['H2_after_separator']
    N2_after_separator = mass_flows_after_separator['N2_after_separator']
    CO2_after_separator = mass_flows_after_separator['CO2_after_separator']
    H2O_after_separator = mass_flows_after_separator['H2O_after_separator']
        
    # wellhead_fugitive_percentages= np.array([
    #     0.073042, 0.102432, 0.130786, 0.130786, 0.134990, 0.179778, 0.195050, 0.195050, 0.195050,
    #     0.195050, 0.195050, 0.195050, 0.195050, 0.195050, 0.220065, 0.244452, 0.244452, 0.244452,
    #     0.244452, 0.274038, 0.274038, 0.274038, 0.274038, 0.274038, 0.274038, 0.274519, 0.274519,
    #     0.274519, 0.274519, 0.274519
    # ])

    # wellhead_fugitive_loss_rates = wellhead_fugitive_percentages / 100 #Convert percentages to decimals, to be consistent with other calculations elsewhere in the model.

    # Use the function that calculates the emissions factors for the wellhead
    wellhead_fugitive_loss_rates = calculate_emissions_factors('Well', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    # Create the DataFrame
    df_wellhead_fugitive_loss_rates = pd.DataFrame({
        'Year': range(1, 31),
        'Wellhead Fugitive Loss Rates %': wellhead_fugitive_loss_rates
    })

    production_fugitive_wellhead_CH4 = wellhead_fugitive_loss_rates.values * GWP_CH4 * CH4_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_wellhead_H2 = wellhead_fugitive_loss_rates.values * GWP_H2 * H2_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_wellhead = production_fugitive_wellhead_CH4 + production_fugitive_wellhead_H2

    ### Now looking at the meter fugitive losses, which also change over the life of the field:

    # # Meter fugitive loss rates extracted from Brandt's model
    # meter_fugitive_percentages = np.array([
    #     0.045713, 0.063821, 0.085305, 0.085305, 0.090005, 0.117416, 0.127744, 0.127744, 0.127744,
    #     0.127744, 0.127744, 0.127744, 0.127744, 0.127744, 0.152734, 0.164370, 0.164370, 0.164370,
    #     0.164370, 0.197726, 0.197726, 0.197726, 0.197726, 0.197726, 0.197726, 0.198287, 0.198287,
    #     0.198287, 0.198287, 0.198287
    # ])

    # meter_fugitive_loss_rates = meter_fugitive_percentages / 100 #Convert percentages to decimals, to be consistent with other calculations elsewhere in the model.

    meter_fugitive_loss_rates = calculate_emissions_factors('Meter', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    # Create the DataFrame for meter fugitive losses
    df_meter_fugitive_losses = pd.DataFrame({
        'Year': range(1, 31),
        'Meter Fugitive Losses %': meter_fugitive_loss_rates
    })

    production_fugitive_meter_CH4 = df_meter_fugitive_losses['Meter Fugitive Losses %'].values * GWP_CH4 * CH4_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_meter_H2 = df_meter_fugitive_losses['Meter Fugitive Losses %'].values  * GWP_H2 * H2_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_meter = production_fugitive_meter_CH4 + production_fugitive_meter_H2

    # ### Dehydrator fugitive loss rates
    # dehydrator_fugitive_percentages = np.array([
    #     0.001652, 0.002315, 0.003324, 0.003324, 0.003659, 0.004127, 0.005150, 0.005150, 0.005150,
    #     0.005150, 0.005150, 0.005150, 0.005150, 0.005150, 0.005773, 0.006185, 0.006185, 0.006185,
    #     0.006185, 0.010305, 0.010305, 0.010305, 0.010305, 0.010305, 0.010305, 0.010315, 0.010315,
    #     0.010315, 0.010315, 0.010315
    # ])

    # dehydrator_fugitive_loss_rates = dehydrator_fugitive_percentages / 100 #Convert percentages to decimals, to be consistent with other calculations elsewhere in the model.

    dehydrator_fugitive_loss_rates = calculate_emissions_factors('Dehydrator', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    # Create the DataFrame for dehydrator fugitive losses
    df_dehydrator_fugitive_losses = pd.DataFrame({
        'Year': range(1, len(dehydrator_fugitive_loss_rates) + 1),
        'Dehydrator Fugitive Losses %': dehydrator_fugitive_loss_rates
    })

    # Calculate the CH4 and H2 components of fugitive emissions from dehydrators
    production_fugitive_dehydrator_CH4 = df_dehydrator_fugitive_losses['Dehydrator Fugitive Losses %'].values  * GWP_CH4 * CH4_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_dehydrator_H2 = df_dehydrator_fugitive_losses['Dehydrator Fugitive Losses %'].values * GWP_H2 * H2_after_separator #Loss Rate * GWP * gas rate
    # Sum the CH4 and H2 components to get the total fugitive emissions from dehydrators
    production_fugitive_dehydrator = production_fugitive_dehydrator_CH4 + production_fugitive_dehydrator_H2

    ### Compressor fugitive losses:

    # # Compressor fugitive loss rates
    # compressor_fugitive_percentages = np.array([
    #     0.006438, 0.009546, 0.012612, 0.012612, 0.012981, 0.019398, 0.020913, 0.020913, 0.020913,
    #     0.020913, 0.020913, 0.020913, 0.020913, 0.020913, 0.028692, 0.031455, 0.031455, 0.031455,
    #     0.031455, 0.033691, 0.033691, 0.033691, 0.033691, 0.033691, 0.033691, 0.033761, 0.033761,
    #     0.033761, 0.033761, 0.033761
    # ])

    # compressor_fugitive_loss_rates = compressor_fugitive_percentages / 100 #Convert percentages to decimals, to be consistent with other calculations elsewhere in the model.

    compressor_fugitive_loss_rates = calculate_emissions_factors('Recip Compressor', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    # Create the DataFrame for compressor fugitive losses
    df_compressor_fugitives = pd.DataFrame({
        'Year': range(1, len(compressor_fugitive_loss_rates) + 1),
        'Compressor Fugitive Losses %': compressor_fugitive_loss_rates
    })

    # Calculate the CH4 and H2 components of fugitive emissions from compressors
    production_fugitive_compressor_CH4 = df_compressor_fugitives['Compressor Fugitive Losses %'].values * GWP_CH4 * CH4_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_compressor_H2 = df_compressor_fugitives['Compressor Fugitive Losses %'].values * GWP_H2 * H2_after_separator #Loss Rate * GWP * gas rate
    # Sum the CH4 and H2 components to get the total fugitive emissions from compressors
    production_fugitive_compressor = production_fugitive_compressor_CH4 + production_fugitive_compressor_H2

    ### Heater fugitive losses:

    # # Heater fugitive loss rates
    # heater_fugitive_percentages = np.array([
    #     0.002672, 0.003710, 0.005876, 0.005876, 0.006623, 0.007484, 0.008829, 0.008829, 0.008829,
    #     0.008829, 0.008829, 0.008829, 0.008829, 0.008829, 0.010364, 0.011616, 0.011616, 0.011616,
    #     0.011616, 0.016787, 0.016787, 0.016787, 0.016787, 0.016787, 0.016787, 0.016800, 0.016800,
    #     0.016800, 0.016800, 0.016800
    # ])

    # heater_fugitive_loss_rates = heater_fugitive_percentages / 100 #Convert percentages to decimals, to be consistent with other calculations elsewhere in the model.

    heater_fugitive_loss_rates = calculate_emissions_factors('Heater', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    # Create the DataFrame for heater fugitive losses
    df_production_fugitive_heater = pd.DataFrame({
        'Year': range(1, 31),
        'Heater Fugitive Losses %': heater_fugitive_loss_rates
    })

    # Calculate the CH4 and H2 components of fugitive emissions from heaters
    production_fugitive_heater_CH4 = df_production_fugitive_heater['Heater Fugitive Losses %'].values * GWP_CH4 * CH4_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_heater_H2 = df_production_fugitive_heater['Heater Fugitive Losses %'].values * GWP_H2 * H2_after_separator #Loss Rate * GWP * gas rate
    # Sum the CH4 and H2 components to get the total fugitive emissions from heaters
    production_fugitive_heater = production_fugitive_heater_CH4 + production_fugitive_heater_H2

    ### Chemical Pump fugitive losses:

    # # Chemical pumps fugitive loss rates
    # chempumps_fugitive_loss_percentages = np.array([
    #     0.015471, 0.024089, 0.030308, 0.030308, 0.034018, 0.048136, 0.054173, 0.054173, 0.054173,
    #     0.054173, 0.054173, 0.054173, 0.054173, 0.054173, 0.068415, 0.080419, 0.080419, 0.080419,
    #     0.080419, 0.094304, 0.094304, 0.094304, 0.094304, 0.094304, 0.094304, 0.094402, 0.094402,
    #     0.094402, 0.094402, 0.094402
    # ])

    # chempumps_fugitive_loss_rates = chempumps_fugitive_loss_percentages / 100 #Convert percentages to decimals, to be consistent with other calculations elsewhere in the model.

    chempumps_fugitive_loss_rates = calculate_emissions_factors('Chemical Injection Pump', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    # Create the DataFrame for chemical pumps fugitive losses
    df_production_fugitive_chempumps = pd.DataFrame({
        'Year': range(1, 31),
        'Chemical Pumps Fugitive Losses %': chempumps_fugitive_loss_rates
    })

    # Calculate the CH4 and H2 components of fugitive emissions from chemical pumps
    production_fugitive_chempumps_CH4 = df_production_fugitive_chempumps['Chemical Pumps Fugitive Losses %'].values * GWP_CH4 * CH4_after_separator #Loss Rate * GWP * gas rate
    production_fugitive_chempumps_H2 = df_production_fugitive_chempumps['Chemical Pumps Fugitive Losses %'].values * GWP_H2 * H2_after_separator #Loss Rate * GWP * gas rate

    # Sum the CH4 and H2 components to get the total fugitive emissions from chemical pumps
    production_fugitive_chempumps = production_fugitive_chempumps_CH4 + production_fugitive_chempumps_H2

    total_production_fugitive_emissions = (
        production_fugitive_wellhead + production_fugitive_meter + production_fugitive_dehydrator +
        production_fugitive_compressor + production_fugitive_heater + production_fugitive_chempumps
    )
    #Add a heading to the output to indicate the case being considered so it can be returned by the function: 
    return {
        'case': case,
        'df_wellhead_fugitive_loss_rates': df_wellhead_fugitive_loss_rates['Wellhead Fugitive Loss Rates %'],
        'production_fugitive_wellhead': production_fugitive_wellhead,
        'production_fugitive_meter': production_fugitive_meter,
        'production_fugitive_dehydrator': production_fugitive_dehydrator,
        'production_fugitive_compressor': production_fugitive_compressor,
        'production_fugitive_heater': production_fugitive_heater,
        'production_fugitive_chempumps': production_fugitive_chempumps,
        'total_production_fugitive_emissions': total_production_fugitive_emissions,
        'CH4_after_separator': CH4_after_separator,
        'wellhead_fugitive_loss_rates': wellhead_fugitive_loss_rates
        # 'df_wellhead_fugitive_loss_rates['Wellhead Fugitive Loss Rates %']': df_wellhead_fugitive_loss_rates['Wellhead Fugitive Loss Rates %']
    }

# Example usage:
emissions_info = calculate_total_production_fugitive_emissions('Baseline')
# print(f"Case: {emissions_info['case']}, Total Fugitive Emissions: {emissions_info['wellhead_fugitive_loss_rates']}")
print(emissions_info['total_production_fugitive_emissions'])


# ## 2.2.1 Surface Processing Fugitives
# 
# Fugitive emissions via surface processing are assumed to only occur as part of 'gas gathering'. This is calculated similarly as in 2.1, above, with a fractional loss rate applied to the gas production rate at the relevant point in the process flow.

# In[233]:


# Gas gathering results are the biggest contributor to VFF emissions, so replicate the calculation of the emissions factors:

def calculate_gas_gathering_emissions_factors(case, sensitivity_variables =None):

    #First, call calculate_development_drilling_emissions to extract the mass flows and calculate the total gas production rate upstream of the separator in tonnes/hr
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
    else:
        oil_production = oil_production_default

    
    mass_flows_after_separator = calculate_mass_flows_after_separator(case, sensitivity_variables)
    N2_after_separator = mass_flows_after_separator['N2_after_separator'] #tonnes/day
    CH4_after_separator = mass_flows_after_separator['CH4_after_separator'] #tonnes/day
    H2_after_separator = mass_flows_after_separator['H2_after_separator'] #tonnes/day
    H2O_after_separator = mass_flows_after_separator['H2O_after_separator'] #tonnes/day
    total_mass_flow_after_separator = mass_flows_after_separator['total_mass_flow_after_separator'] #tonnes/day

    total_molar_flow_after_separator = (
        (N2_after_separator/ molecular_weights_gases.loc['N2', 'Molecular Weight (g/mol)'] +
         CH4_after_separator / molecular_weights_gases.loc['CH4', 'Molecular Weight (g/mol)'] +
         H2_after_separator / molecular_weights_gases.loc['H2', 'Molecular Weight (g/mol)'] +
         H2O_after_separator / molecular_weights_gases.loc['H2O', 'Molecular Weight (g/mol)']) * 1E6  # mol/day
    )

    total_gas_flow_after_separator = total_molar_flow_after_separator / mol_per_SCF / 1E6 #MMSCFD
    gathering_site_avg_throughput = 19.3 #MMSCFD per site. This is a default value from Brandt's model.
    expected_no_gathering_sites = total_gas_flow_after_separator / gathering_site_avg_throughput 
    
    gathering_system_throughput_per_site_per_hr = total_mass_flow_after_separator / expected_no_gathering_sites / 24 #tonnes/hr
    gathering_system_throughput_per_site_per_hr_log10 = np.log10(gathering_system_throughput_per_site_per_hr) #log10 of the total gas mass flow rate upstream of the separator in tonnes/hr

    gas_gathering_loss_rates = 10**((-1.8618)+gathering_system_throughput_per_site_per_hr_log10*(-0.59397)) #Constants for intercept and slope taken directly as defaults from Brandt's model.

    return {
        'case': case,
        'total_mass_flow_after_separator': total_mass_flow_after_separator,
        'total_molar_flow_after_separator': total_molar_flow_after_separator,
        'total_gas_flow_after_separator': total_gas_flow_after_separator, #MMSCFD
        'gathering_system_throughput_per_site_per_hr': gathering_system_throughput_per_site_per_hr,
        'gas_gathering_loss_rates': gas_gathering_loss_rates
    }

test = calculate_gas_gathering_emissions_factors('Baseline')
print(test['gas_gathering_loss_rates'])


# In[234]:


#Helper function for fugitive emissions from surface processing:

def calculate_gas_gathering_fugitives(case, sensitivity_variables =None):
    if sensitivity_variables:
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
    else:
        GWP_H2 = GWP_H2_default
        
    # Calculate the mass flows downstream of the separator
    mass_flows_after_separator = calculate_mass_flows_after_separator(case, sensitivity_variables)
    CH4_after_separator = mass_flows_after_separator['CH4_after_separator']
    H2_after_separator = mass_flows_after_separator['H2_after_separator']
    N2_after_separator = mass_flows_after_separator['N2_after_separator']
    CO2_after_separator = mass_flows_after_separator['CO2_after_separator']
    H2O_after_separator = mass_flows_after_separator['H2O_after_separator']

    # gas_gathering_fugitive_rate = 0.00495897945648821 #Taken from OPGEE model. This factor does not vary with production rate, so a series is not required and a single value is used for all years.

    gas_gathering_fugitive_rate = calculate_gas_gathering_emissions_factors(case,sensitivity_variables)['gas_gathering_loss_rates']

    gas_gathering_fugitive_CH4 = gas_gathering_fugitive_rate * GWP_CH4 * CH4_after_separator
    gas_gathering_fugitive_H2 = gas_gathering_fugitive_rate * GWP_H2 * H2_after_separator
    
    gas_gathering_fugitives = gas_gathering_fugitive_CH4 + gas_gathering_fugitive_H2

    #Need to also calculate the N2 losses (mass calculated only, as N2 has no GWP)
    gas_gathering_N2_losses = gas_gathering_fugitive_rate * N2_after_separator 

    N2_after_gas_gathering = N2_after_separator - gas_gathering_N2_losses

    H2O_after_gas_gathering = H2O_after_separator * (1 - gas_gathering_fugitive_rate)

    return {
        'case': case,
        'gas_gathering_fugitive_CH4': gas_gathering_fugitive_CH4,
        'gas_gathering_fugitive_H2': gas_gathering_fugitive_H2,
        'gas_gathering_fugitives': gas_gathering_fugitives,
        'N2_after_gas_gathering': N2_after_gas_gathering,
        'H2O_after_gas_gathering': H2O_after_gas_gathering
    }

# #Example usage:
# gas_gathering_fugitives_info = calculate_gas_gathering_fugitives('Baseline')
# print(f"Case: {gas_gathering_fugitives_info['case']}, Gas Gathering Fugitives: {gas_gathering_fugitives_info['H2O_after_gas_gathering']}")


# ## 2.2.2 Surface Processing Venting
# 
# Vented emissions through surface processing are assumed to only occur as part of gas dehydration (i.e. the glycol unit).
# 
# 

# In[235]:


#Helper function:
def calculate_gas_dehydration_vents(case, sensitivity_variables =None):
    if sensitivity_variables:
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
    else:
        GWP_H2 = GWP_H2_default
    
    # Calculate the mass flows downstream of the separator
    mass_flows_after_separator = calculate_mass_flows_after_separator(case, sensitivity_variables)
    N2_after_separator = mass_flows_after_separator['N2_after_separator'] #tonnes/day
    CH4_after_separator = mass_flows_after_separator['CH4_after_separator'] #tonnes/day
    H2_after_separator = mass_flows_after_separator['H2_after_separator'] #tonnes/day
    H2O_after_separator = mass_flows_after_separator['H2O_after_separator'] #tonnes/day
    total_mass_flow_after_separator = mass_flows_after_separator['total_mass_flow_after_separator'] #tonnes/day

    total_molar_flow_after_separator = (
        (N2_after_separator/ molecular_weights_gases.loc['N2', 'Molecular Weight (g/mol)'] +
         CH4_after_separator / molecular_weights_gases.loc['CH4', 'Molecular Weight (g/mol)'] +
         H2_after_separator / molecular_weights_gases.loc['H2', 'Molecular Weight (g/mol)'] +
         H2O_after_separator / molecular_weights_gases.loc['H2O', 'Molecular Weight (g/mol)']) * 1E6  # mol/day
    )

    total_gas_flow_after_separator = total_molar_flow_after_separator / mol_per_SCF / 1E6 #MMSCFD
    
    # Call the gas gathering fugitives function and store the results
    fugitives = calculate_gas_gathering_fugitives(case,sensitivity_variables)

    # Extract the CH4 and H2 after accounting for fugitive losses
    CH4_after_gas_gathering = CH4_after_separator - fugitives['gas_gathering_fugitive_CH4'] / GWP_CH4
    H2_after_gas_gathering = H2_after_separator - fugitives['gas_gathering_fugitive_H2'] / GWP_H2
    
    N2_after_gas_gathering = fugitives['N2_after_gas_gathering'] #tonne/day
    H2O_after_gas_gathering = fugitives['H2O_after_gas_gathering'] #tonne/day

    # Calculate the vents rates downstream of gas gathering (i.e., upstream of dehydration)
    # gas_dehydration_vent_rate = 0.000810912362269712  # Taken from OPGEE model. This value is a ratio/decimal, not a percentage, so no conversion is required.

    # Instead of taking the vent rate from Brandt, the following section replicates the underlying calculations.
    processing_plant_avg_throughput = 65500/517 #MMSCFD/site. US EIA in OPGEE.

    expected_no_processing_sites = total_gas_flow_after_separator / processing_plant_avg_throughput #Number of processing sites required to process the gas flow rate
    
    processing_plant_throughput_tonnes_per_hr = total_mass_flow_after_separator / 24 / expected_no_processing_sites #tonnes/hr. This line should probably really use the total gas flow after gas gathering, but Brandt uses the total mass flow after the separator, so this is what is used here.
    processing_plant_throughput_tonnes_log = np.log10(processing_plant_throughput_tonnes_per_hr) #log10 of the total gas mass flow rate upstream of the separator in tonnes/hr
    predicted_processing_system_loss_rate = 10**(-1.8618 + processing_plant_throughput_tonnes_log * (-0.59397))

    dehydration_process_unit_fraction = 0.5 #Fraction of gas processed in the dehydration unit. This is a default value from OPGEE.

    gas_dehydration_vent_rate = predicted_processing_system_loss_rate * dehydration_process_unit_fraction

    # Calculate CH4 and H2 emissions from dehydration venting
    gas_dehydration_vent_CH4 = gas_dehydration_vent_rate * GWP_CH4 * CH4_after_gas_gathering
    gas_dehydration_vent_H2 = gas_dehydration_vent_rate * GWP_H2 * H2_after_gas_gathering
    gas_dehydration_vents = gas_dehydration_vent_CH4 + gas_dehydration_vent_H2

    gas_dehydration_N2_losses = gas_dehydration_vent_rate * N2_after_gas_gathering

    #For later use, now calculate the gas flows remaining downstream of the dehydration system:

    CH4_after_dehy = CH4_after_gas_gathering - gas_dehydration_vent_CH4/GWP_CH4
    H2_after_dehy = H2_after_gas_gathering - gas_dehydration_vent_H2/GWP_H2
    N2_after_dehy = N2_after_gas_gathering - gas_dehydration_N2_losses

    # Calculating residual water content in the gas stream after dehydration is required, but more complicated than for other gases.
    # The following calculations are based on the Wagner and Pruss equation, which is used to calculate the vapor pressure of water at a given temperature.
    # This is then used to calculate the water content of the gas stream at the given temperature and pressure.

    gas_temp_F = 90 #F. Temperature of gas at inlet conditions
    gas_temp_K = (gas_temp_F - 32) * 5/9 + 273.15 #K. Temperature of gas at inlet conditions in Kelvin
    gas_pressure_psi = 500 #psi. Pressure of gas at inlet conditions
    gas_pressure = gas_pressure_psi * 0.00689476 #MPa. Pressure of gas at inlet conditions. Assumes that this is always 500 psi
    water_critical_temp = 647.096 #K
    water_critical_pressure = 22.064 #MPa
    tau = 1 - gas_temp_K / water_critical_temp
    Tc_on_T = water_critical_temp / gas_temp_K

    # water_mass_coefficients for Wagner and Pruss equation
    a1 =	-7.8595178
    a2 =	1.8440826
    a3 =	-11.7866497
    a4 =	22.6807411
    a5 =	-15.9618719
    a6 =	1.8012250

    ln_Pv_on_Pc = (a1*tau + a2*tau**(1.5) + a3*tau**(3) + a4*tau**(3.5) + a5*tau**(4) + a6*tau**(7.5)) * Tc_on_T
    Pv_on_Pc = np.exp(ln_Pv_on_Pc)
    Pv = Pv_on_Pc * water_critical_pressure #MPa. Vapor pressure of water at inlet conditions

    LogB = 6.69449-3083.87/(gas_temp_F+459.67) #Assumes inlet gas temperature is 90F
    B = 10**LogB 
     
    gas_water_content = 47484*Pv/gas_pressure + B #lb water/mmscf
    amount_gas_processed_daily = (CH4_after_gas_gathering / gas_densities.loc['CH4', 'Density tonne/MMSCF'] + H2_after_gas_gathering / gas_densities.loc['H2', 'Density tonne/MMSCF'] + N2_after_gas_gathering / gas_densities.loc['N2', 'Density tonne/MMSCF'] + H2O_after_gas_gathering / gas_densities.loc['H2O', 'Density tonne/MMSCF']) #lb water/mmscf #mmscf/day
    gas_multiplier = amount_gas_processed_daily / 1.0897 #multiplier for gas load in correlation equation for Aspen HYSYS
    mass_water_content = gas_water_content * amount_gas_processed_daily * 454.5 #g/day

    water_in_H2_concentration = 0.025 # kg H2O / kg H2 from Huang 2008
    water_in_H2_tonnes_per_day = H2_after_dehy * water_in_H2_concentration # tonnes/day
    water_in_N2_tonnes_per_day = 0.0069899000 * N2_after_dehy # tonnes/day. Per Brandt's application of Luks et al
    water_in_H2_N2_g_per_day = (water_in_H2_tonnes_per_day + water_in_N2_tonnes_per_day) * 1E6 # g/day
    water_in_H2_N2_MMSCFD = water_in_H2_N2_g_per_day / 22E6 # MMSCFD
    water_into_system_day_aspen = water_in_H2_N2_MMSCFD / gas_multiplier # MMSCFD

    x1 = gas_pressure_psi #psi
    x2 = gas_temp_F #F
    x3 = water_into_system_day_aspen #MMSCFD
    x4 = dehy_reflux_ratio 
    x5 = dehy_regen_temp #F
    value = amount_gas_processed_daily / 1.0897 #multiplier for gas load in correlation equation for Aspen HYSYS
    
    #Calculate the quadratic terms associated with x1 to x5. i.e. the square of each value and the cross terms.
    x1_sq = x1**2
    x2_sq = x2**2
    x3_sq = x3**2
    x4_sq = x4**2
    x5_sq = x5**2
    x1_x2 = x1 * x2
    x1_x3 = x1 * x3
    x1_x4 = x1 * x4
    x1_x5 = x1 * x5
    x2_x3 = x2 * x3
    x2_x4 = x2 * x4
    x2_x5 = x2 * x5
    x3_x4 = x3 * x4
    x3_x5 = x3 * x5
    x4_x5 = x4 * x5

    #Extract the water mass coefficients directly from the OPGEE model:
    water_mass_coefficients = {
    "Intercept": -0.0048379,
    "x1": -1.85780E-06,
    "x2": 1.76340E-05,
    "x3": 3.60070E+01,
    "x4": -4.88140E-05,
    "x5": 4.55080E-07,
    "x1:x2": 1.02290E-08,
    "x1:x3": -9.36920E-04,
    "x1:x4": 8.74220E-09,
    "x1:x5": 5.69770E-09,
    "x2:x3": 1.64790E-02,
    "x2:x4": 2.78650E-07,
    "x2:x5": -1.11930E-07,
    "x3:x4": -1.65480E-04,
    "x3:x5": -1.52550E-02,
    "x4:x5": 7.61480E-09,
    "x1^2": -1.04350E-10,
    "x2^2": -3.09780E-08,
    "x3^2": 4.18500E+00,
    "x4^2": 2.88120E-06,
    "x5^2": 9.13320E-09
    }

    # Calculate the sum product
    predicted_water_output = (
        water_mass_coefficients["Intercept"] +
        water_mass_coefficients["x1"] * x1 +
        water_mass_coefficients["x2"] * x2 +
        water_mass_coefficients["x3"] * x3 +
        water_mass_coefficients["x4"] * x4 +
        water_mass_coefficients["x5"] * x5 +
        water_mass_coefficients["x1:x2"] * x1_x2 +
        water_mass_coefficients["x1:x3"] * x1_x3 +
        water_mass_coefficients["x1:x4"] * x1_x4 +
        water_mass_coefficients["x1:x5"] * x1_x5 +
        water_mass_coefficients["x2:x3"] * x2_x3 +
        water_mass_coefficients["x2:x4"] * x2_x4 +
        water_mass_coefficients["x2:x5"] * x2_x5 +
        water_mass_coefficients["x3:x4"] * x3_x4 +
        water_mass_coefficients["x3:x5"] * x3_x5 +
        water_mass_coefficients["x4:x5"] * x4_x5 +
        water_mass_coefficients["x1^2"] * x1_sq +
        water_mass_coefficients["x2^2"] * x2_sq +
        water_mass_coefficients["x3^2"] * x3_sq +
        water_mass_coefficients["x4^2"] * x4_sq +
        water_mass_coefficients["x5^2"] * x5_sq
    ) #lb/MMSCF

    #Convert the output to tonnes/day
    predicted_water_output_tonnes_per_day = predicted_water_output *24*2.24/1000 #tonnes/day
    H2O_after_dehy = predicted_water_output_tonnes_per_day #tonnes/day


    return {
        'case': case,
        'gas_dehydration_vents': gas_dehydration_vents,
        'CH4_after_dehy': CH4_after_dehy,
        'H2_after_dehy': H2_after_dehy,
        'N2_after_dehy': N2_after_dehy,
        'H2O_after_dehy': H2O_after_dehy,
        'gas_dehydration_vent_rate': gas_dehydration_vent_rate
    }

# # Test usage
# gas_dehydration_vents_info = calculate_gas_dehydration_vents('Baseline')
# print(f"Case: {gas_dehydration_vents_info['case']}, Gas Dehydration Vents: {gas_dehydration_vents_info['gas_dehydration_vent_rate']}")


# ## 2.3 HC gas reinjection compressor
# 
# The Brandt paper assumes that non-H2 captured waste streams are compressed and re-injected into the subsurface. This factor estimates the amount of fugitive emissions through the compressor sub-system.
# 
# 

# In[236]:


def calculate_HC_gas_reinjection_compressor_fugitives(case, sensitivity_variables =None):
    if sensitivity_variables:
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        field_lifespan = field_lifespan_default
        GWP_H2 = GWP_H2_default
        number_production_wells = number_production_wells_default

    # Call calculate_gas_dehydration_vents to get the CH4 and H2 values after dehydration and determine the mass flows to the reinjection compressors
    gas_dehydration_vents_info = calculate_gas_dehydration_vents(case,sensitivity_variables)
    CH4_after_dehy = gas_dehydration_vents_info['CH4_after_dehy'] 
    CH4_to_reinjection = CH4_after_dehy #CH4 to reinjection is the same as CH4 after dehydration, as all of the CH4 stream is sent to reinjection.

    H2_after_dehy = gas_dehydration_vents_info['H2_after_dehy']
    H2_to_reinjection = H2_after_dehy * PSA_unit_slippage_rate #All of non-product gas stream from PSA unit is sent to reinjection, so this includes all of the H2 that is not properly separated by the PSA unit.
    
    N2_to_reinjection = gas_dehydration_vents_info['N2_after_dehy'] #All of the N2 stream is sent to reinjection, so this includes all of the N2 that is not properly separated by the PSA unit.
    H2O_to_reinjection = gas_dehydration_vents_info['H2O_after_dehy'] #All of the H2O stream is sent to reinjection, so this includes all of the H2O that is not properly separated by the PSA unit.

    total_gas_mass_flow_to_reinjection = CH4_to_reinjection + H2_to_reinjection + N2_to_reinjection + H2O_to_reinjection

    #The OPGEE model assumes a portion of the gas flowing to the reinjection compressors is used to power the compressors. Thus, first need to calculate the amount of fuel gas required to power the compressors.

    # In order to calculate the gas consumed when powering the compressors, first need to calculate the energy density of the associated gas stream:

    # Calculate the energy density of the gas stream in MJ/kg
    gas_energy_density_metric = (
        CH4_to_reinjection * LHV_density_gases_metric.loc['CH4', 'LHV (MJ/kg)'] +
        H2_to_reinjection * LHV_density_gases_metric.loc['H2', 'LHV (MJ/kg)']
    ) / (CH4_to_reinjection + H2_to_reinjection + N2_to_reinjection)  # MJ/kg

    # Calculate the energy density of the gas stream in btu/lb
    gas_energy_density_imperial = gas_energy_density_metric * btu_per_MJ / Pounds_per_kg  # Convert MJ/kg to btu/lb

    # Calculate the energy demand of the compressors in mmbtu/day. First need to calculate the required BHP of the compressors and the associated fuel use.

    # To calculate the BHP of the compressors, need to calculate the total adiabatic work of compression and factor in compressor inefficiencies.
    # Extract the reservoir pressure column for the selected case from the production_profiles DataFrame

    # Extract the reservoir pressure for the specified case
    reservoir_pressure = production_profile_df.get(case + ' Wellhead Pressure, PSI')

    # If the specified case column does not exist, default to the 'Baseline' case values
    if reservoir_pressure is None:
        reservoir_pressure = production_profile_df.get('Baseline Wellhead Pressure, PSI')

    compressor_discharge_pressure = reservoir_pressure + 500  # psia
    pressure_upstream_compressor = 25  # psia
    required_compression_ratio = compressor_discharge_pressure / pressure_upstream_compressor

    # Calculate the compression ratio per stage, assuming the ratio must be <5 per stage
    def calculate_compression_ratio_per_stage(required_compression_ratio):
        # Create an empty list to store the compression ratios per stage
        compression_ratios_per_stage = []

        # Iterate through each value in the Series
        for ratio in required_compression_ratio:
            if ratio < 5:
                compression_ratio_per_stage = ratio
            elif ratio**(1/2) < 5:
                compression_ratio_per_stage = ratio**(1/2)
            elif ratio**(1/3) < 5:
                compression_ratio_per_stage = ratio**(1/3)
            elif ratio**(1/4) < 5:
                compression_ratio_per_stage = ratio**(1/4)
            else:
                compression_ratio_per_stage = ratio**(1/5)
            compression_ratios_per_stage.append(compression_ratio_per_stage)
        
        # Convert the list back to a Series
        return pd.Series(compression_ratios_per_stage)

    # Example usage
    # required_compression_ratio = pd.Series([4, 8, 27, 64, 125])  # Example Series
    compression_ratio_per_stage = calculate_compression_ratio_per_stage(required_compression_ratio)
    
    # Based on the required compression ratio and associated compression ratio per stage, calculate the number of stages required
    def calculate_compression_stages(required_compression_ratio, compression_ratio_per_stage):
        if np.allclose(required_compression_ratio, compression_ratio_per_stage):
            return 1
        elif np.allclose(required_compression_ratio**(1/2), compression_ratio_per_stage):
            return 2
        elif np.allclose(required_compression_ratio**(1/3), compression_ratio_per_stage):
            return 3
        elif np.allclose(required_compression_ratio**(1/4), compression_ratio_per_stage):
            return 4
        else:
            return 5

    num_stages = calculate_compression_stages(required_compression_ratio, compression_ratio_per_stage)
    
    inlet_total_molar_flow = (
        (N2_to_reinjection / molecular_weights_gases.loc['N2', 'Molecular Weight (g/mol)'] +
         CH4_to_reinjection / molecular_weights_gases.loc['CH4', 'Molecular Weight (g/mol)'] +
         H2_to_reinjection / molecular_weights_gases.loc['H2', 'Molecular Weight (g/mol)'] +
         H2O_to_reinjection / molecular_weights_gases.loc['H2O', 'Molecular Weight (g/mol)']) * 1E6  # mol/day
    )

    total_gas_volume_flow_to_reinjection = inlet_total_molar_flow / mol_per_SCF / 1E6  # MMSCFD

    # Calculate the molar fractions of each gas in the gas stream
    mol_frac_N2 = N2_to_reinjection * 1E6 / molecular_weights_gases.loc['N2', 'Molecular Weight (g/mol)'] / inlet_total_molar_flow
    mol_frac_CH4 = CH4_to_reinjection * 1E6 / molecular_weights_gases.loc['CH4', 'Molecular Weight (g/mol)'] / inlet_total_molar_flow
    mol_frac_H2 = H2_to_reinjection * 1E6 / molecular_weights_gases.loc['H2', 'Molecular Weight (g/mol)'] / inlet_total_molar_flow
    mol_frac_H2O = H2O_to_reinjection * 1E6 / molecular_weights_gases.loc['H2O', 'Molecular Weight (g/mol)'] / inlet_total_molar_flow

    # Calculate the pseudo-critical temperature and pressure of the gas stream, as per the equation provided in the OPGEE model
    inlet_pseduocritical_temp = (
        (mol_frac_N2 * pseudo_crit_constants.loc['N2', 'Tc_Pc_Constant_K'] +
        mol_frac_CH4 * pseudo_crit_constants.loc['CH4', 'Tc_Pc_Constant_K'] +
        mol_frac_H2 * pseudo_crit_constants.loc['H2', 'Tc_Pc_Constant_K'] +
        mol_frac_H2O * pseudo_crit_constants.loc['H2O', 'Tc_Pc_Constant_K'])**2 /
        ((1/3) * (mol_frac_N2 * pseudo_crit_constants.loc['N2', 'Tc_Pc'] +
        mol_frac_CH4 * pseudo_crit_constants.loc['CH4', 'Tc_Pc'] +
        mol_frac_H2 * pseudo_crit_constants.loc['H2', 'Tc_Pc'] +
        mol_frac_H2O * pseudo_crit_constants.loc['H2O', 'Tc_Pc']) +
        (2/3) * (mol_frac_N2 * pseudo_crit_constants.loc['N2', 'sqrt_Tc_Pc'] +
        mol_frac_CH4 * pseudo_crit_constants.loc['CH4', 'sqrt_Tc_Pc'] +
        mol_frac_H2 * pseudo_crit_constants.loc['H2', 'sqrt_Tc_Pc'] +
        mol_frac_H2O * pseudo_crit_constants.loc['H2O', 'sqrt_Tc_Pc'])**2)
        )
    inlet_pseduocritical_temp_corr = inlet_pseduocritical_temp # Correction only required if the gas stream contains CO2 and/or H2S
    
    # Calculate the pseudo-critical pressure of the gas stream, as per the equation provided in the OPGEE model
    inlet_pseduocritical_pressure = ( #psia 
           (mol_frac_N2 * pseudo_crit_constants.loc['N2', 'Tc_Pc_Constant_K'] +
        mol_frac_CH4 * pseudo_crit_constants.loc['CH4', 'Tc_Pc_Constant_K'] +
        mol_frac_H2 * pseudo_crit_constants.loc['H2', 'Tc_Pc_Constant_K'] +
        mol_frac_H2O * pseudo_crit_constants.loc['H2O', 'Tc_Pc_Constant_K'])**2 /
        ((1/3) * (mol_frac_N2 * pseudo_crit_constants.loc['N2', 'Tc_Pc'] +
        mol_frac_CH4 * pseudo_crit_constants.loc['CH4', 'Tc_Pc'] +
        mol_frac_H2 * pseudo_crit_constants.loc['H2', 'Tc_Pc'] +
        mol_frac_H2O * pseudo_crit_constants.loc['H2O', 'Tc_Pc']) +
        (2/3) * (mol_frac_N2 * pseudo_crit_constants.loc['N2', 'sqrt_Tc_Pc'] +
        mol_frac_CH4 * pseudo_crit_constants.loc['CH4', 'sqrt_Tc_Pc'] +
        mol_frac_H2 * pseudo_crit_constants.loc['H2', 'sqrt_Tc_Pc'] +
        mol_frac_H2O * pseudo_crit_constants.loc['H2O', 'sqrt_Tc_Pc'])**2)**2
        )
    
    # Calculate the ratio of specific heats of the gas stream
    specific_heat_ratio = (
        (N2_to_reinjection*1000 * specific_heat_df.loc['N2', 'Specific heat C_p'] +
        CH4_to_reinjection*1000 * specific_heat_df.loc['CH4', 'Specific heat C_p'] +
        H2_to_reinjection*1000 * specific_heat_df.loc['H2', 'Specific heat C_p'] +
        H2O_to_reinjection*1000 * specific_heat_df.loc['H2O', 'Specific heat C_p']) /
        (N2_to_reinjection*1000 * specific_heat_df.loc['N2', 'Specific heat C_v'] +
        CH4_to_reinjection*1000 * specific_heat_df.loc['CH4', 'Specific heat C_v'] +
        H2_to_reinjection*1000 * specific_heat_df.loc['H2', 'Specific heat C_v'] +
        H2O_to_reinjection*1000 * specific_heat_df.loc['H2O', 'Specific heat C_v'])
    )

    first_stage_inlet_Z_factor = 1.00 #Assume ideal gas for now. Lookup table (large) for Z factor is required for more accurate results
    second_stage_inlet_Z_factor = 1.00 #Assume ideal gas for now. Lookup table (large) for Z factor is required for more accurate results
    third_stage_inlet_Z_factor = 1.00 #Assume ideal gas for now. Lookup table (large) for Z factor is required for more accurate results
    fourth_stage_inlet_Z_factor = 1.00 #Assume ideal gas for now. Lookup table (large) for Z factor is required for more accurate results
    fifth_stage_inlet_Z_factor = 1.00 #Assume ideal gas for now. Lookup table (large) for Z factor is required for more accurate results

    first_stage_inlet_temp_F = 90 #degF
    first_stage_inlet_temp_R = first_stage_inlet_temp_F + 460 #Convert to Rankine
    first_stage_inlet_pressure = pressure_upstream_compressor #psia
    
    # Calculate the outlet temperature of the first stage of the compressor
    first_stage_outlet_temp_F = ((((first_stage_inlet_temp_R) * (compression_ratio_per_stage ** ((first_stage_inlet_Z_factor * (specific_heat_ratio - 1)) / specific_heat_ratio)) - 460) - first_stage_inlet_temp_F) * 0.2) + first_stage_inlet_temp_F #degF
    first_stage_discharge_pressure = first_stage_inlet_pressure * compression_ratio_per_stage #psia
    
    first_stage_inlet_reduced_temp = first_stage_inlet_temp_R / inlet_pseduocritical_temp
    first_stage_inlet_reduced_pressure = first_stage_inlet_pressure / inlet_pseduocritical_pressure

    # Calculate the adiabatic work of the first stage of the compressor
    first_stage_adiabatic_work = ((specific_heat_ratio / (specific_heat_ratio - 1)) * 
                              (3.027 * 14.7 / (60 + 460)) * 
                              first_stage_inlet_temp_R * 
                              ((compression_ratio_per_stage ** (first_stage_inlet_Z_factor * (specific_heat_ratio - 1) / specific_heat_ratio)) - 1))# hp/MMSCFD
    
    second_stage_inlet_temp_F = first_stage_outlet_temp_F #degF
    second_stage_inlet_temp_R = second_stage_inlet_temp_F + 460 #Convert to Rankine
    second_stage_inlet_pressure = first_stage_discharge_pressure #psia

    # Calculate the outlet temperature of the second stage of the compressor
    second_stage_outlet_temp_F = ((((second_stage_inlet_temp_R) * (compression_ratio_per_stage ** ((second_stage_inlet_Z_factor * (specific_heat_ratio - 1)) / specific_heat_ratio)) - 460) - second_stage_inlet_temp_F) * 0.2) + second_stage_inlet_temp_F #degF
    second_stage_discharge_pressure = second_stage_inlet_pressure * compression_ratio_per_stage #psia

    second_stage_inlet_reduced_temp = second_stage_inlet_temp_R / inlet_pseduocritical_temp
    second_stage_inlet_reduced_pressure = second_stage_inlet_pressure / inlet_pseduocritical_pressure

    # Calculate the adiabatic work of the second stage of the compressor
    second_stage_adiabatic_work = ((specific_heat_ratio / (specific_heat_ratio - 1)) * 
                              (3.027 * 14.7 / (60 + 460)) * 
                              second_stage_inlet_temp_R * 
                              ((compression_ratio_per_stage ** (second_stage_inlet_Z_factor * (specific_heat_ratio - 1) / specific_heat_ratio)) - 1))# hp/MMSCFD
    
    third_stage_inlet_temp_F = second_stage_outlet_temp_F #degF
    third_stage_inlet_temp_R = third_stage_inlet_temp_F + 460 #Convert to Rankine
    third_stage_inlet_pressure = second_stage_discharge_pressure #psia

    # Calculate the outlet temperature of the third stage of the compressor
    third_stage_outlet_temp_F = ((((third_stage_inlet_temp_R) * (compression_ratio_per_stage ** ((third_stage_inlet_Z_factor * (specific_heat_ratio - 1)) / specific_heat_ratio)) - 460) - third_stage_inlet_temp_F) * 0.2) + third_stage_inlet_temp_F #degF
    third_stage_discharge_pressure = third_stage_inlet_pressure * compression_ratio_per_stage #psia

    third_stage_inlet_reduced_temp = third_stage_inlet_temp_R / inlet_pseduocritical_temp
    third_stage_inlet_reduced_pressure = third_stage_inlet_pressure / inlet_pseduocritical_pressure

    # Calculate the adiabatic work of the third stage of the compressor
    third_stage_adiabatic_work = ((specific_heat_ratio / (specific_heat_ratio - 1)) * 
                              (3.027 * 14.7 / (60 + 460)) * 
                              third_stage_inlet_temp_R * 
                              ((compression_ratio_per_stage ** (third_stage_inlet_Z_factor * (specific_heat_ratio - 1) / specific_heat_ratio)) - 1))# hp/MMSCFD
    
    fourth_stage_inlet_temp_F = third_stage_outlet_temp_F #degF
    fourth_stage_inlet_temp_R = fourth_stage_inlet_temp_F + 460 #Convert to Rankine
    fourth_stage_inlet_pressure = third_stage_discharge_pressure #psia

    # Calculate the outlet temperature of the fourth stage of the compressor
    fourth_stage_outlet_temp_F = ((((fourth_stage_inlet_temp_R) * (compression_ratio_per_stage ** ((fourth_stage_inlet_Z_factor * (specific_heat_ratio - 1)) / specific_heat_ratio)) - 460) - fourth_stage_inlet_temp_F) * 0.2) + fourth_stage_inlet_temp_F #degF
    fourth_stage_discharge_pressure = fourth_stage_inlet_pressure * compression_ratio_per_stage #psia

    fourth_stage_inlet_reduced_temp = fourth_stage_inlet_temp_R / inlet_pseduocritical_temp
    fourth_stage_inlet_reduced_pressure = fourth_stage_inlet_pressure / inlet_pseduocritical_pressure

    # Calculate the adiabatic work of the fourth stage of the compressor
    fourth_stage_adiabatic_work = ((specific_heat_ratio / (specific_heat_ratio - 1)) * 
                              (3.027 * 14.7 / (60 + 460)) * 
                              fourth_stage_inlet_temp_R * 
                              ((compression_ratio_per_stage ** (fourth_stage_inlet_Z_factor * (specific_heat_ratio - 1) / specific_heat_ratio)) - 1))# hp/MMSCFD
    
    fifth_stage_inlet_temp_F = fourth_stage_outlet_temp_F #degF
    fifth_stage_inlet_temp_R = fifth_stage_inlet_temp_F + 460 #Convert to Rankine
    fifth_stage_inlet_pressure = fourth_stage_discharge_pressure #psia

    # Calculate the outlet temperature of the fifth stage of the compressor
    fifth_stage_outlet_temp_F = ((((fifth_stage_inlet_temp_R) * (compression_ratio_per_stage ** ((fifth_stage_inlet_Z_factor * (specific_heat_ratio - 1)) / specific_heat_ratio)) - 460) - fifth_stage_inlet_temp_F) * 0.2) + fifth_stage_inlet_temp_F #degF
    fifth_stage_discharge_pressure = fifth_stage_inlet_pressure * compression_ratio_per_stage #psia

    fifth_stage_inlet_reduced_temp = fifth_stage_inlet_temp_R / inlet_pseduocritical_temp
    fifth_stage_inlet_reduced_pressure = fifth_stage_inlet_pressure / inlet_pseduocritical_pressure

    # Calculate the adiabatic work of the fifth stage of the compressor
    fifth_stage_adiabatic_work = ((specific_heat_ratio / (specific_heat_ratio - 1)) * 
                              (3.027 * 14.7 / (60 + 460)) * 
                              fifth_stage_inlet_temp_R * 
                              ((compression_ratio_per_stage ** (fifth_stage_inlet_Z_factor * (specific_heat_ratio - 1) / specific_heat_ratio)) - 1))# hp/MMSCFD
    
    
    # Calculate the total adiabatic work of the compressor, based on the number of stages. i.e. Excluding stages > num_stages.
    if num_stages == 1:
        total_compressor_adiabatic_work = first_stage_adiabatic_work
    elif num_stages == 2:
        total_compressor_adiabatic_work = first_stage_adiabatic_work + second_stage_adiabatic_work
    elif num_stages == 3:
        total_compressor_adiabatic_work = first_stage_adiabatic_work + second_stage_adiabatic_work + third_stage_adiabatic_work
    elif num_stages == 4:
        total_compressor_adiabatic_work = first_stage_adiabatic_work + second_stage_adiabatic_work + third_stage_adiabatic_work + fourth_stage_adiabatic_work
    elif num_stages == 5:
        total_compressor_adiabatic_work = first_stage_adiabatic_work + second_stage_adiabatic_work + third_stage_adiabatic_work + fourth_stage_adiabatic_work + fifth_stage_adiabatic_work
    else:
        print('Error: Number of stages is not within the expected range of 1-5')
        return
    
    # Now calculate the output horsepower of the compressor
    total_MMSCFD = inlet_total_molar_flow / 1E6 / mol_per_SCF #MMSCFD
    compressor_output_hp = total_compressor_adiabatic_work * total_MMSCFD #hp
   
    compressor_bhp = compressor_output_hp / eta_compressor #bhp

    # Cycle through each of the values of compressor_bhp to determine which bhp value in the natural gas engine efficiency dataframe is closest and record this value in the closest_bhp variable
    closest_bhp = []
    for bhp in compressor_bhp:
        closest_bhp.append(ng_engine_efficiency_data_df.index.to_series().sub(bhp).abs().idxmin())

    # Ensure closest_bhp is a Series
    closest_bhp = pd.Series(closest_bhp)

    # Return the fuel use associated with each closest bhp value
    compressor_fuel_efficiency = ng_engine_efficiency_data_df.loc[closest_bhp, 'Efficiency btu LHV/bhp-hr'].values  # Convert to NumPy array for element-wise multiplication

    # Calculate the energy required by the compressors in mmbtu/day
    compressor_total_fuel_use = compressor_bhp * compressor_fuel_efficiency * 24 / 1E6  # mmbtu/day

    energy_flow_rate = total_gas_mass_flow_to_reinjection * 1000 * gas_energy_density_metric / mmbtu_to_MJ # mmBtu/d

    # Now calculate the fraction of the gas stream that is consumed by the compressors as fuel gas
    fraction_gas_consumed = compressor_total_fuel_use / energy_flow_rate

    # Calculate the fuel gas consumption of each component of the gas stream
    CH4_fuel_gas_consumption = fraction_gas_consumed * CH4_to_reinjection
    H2_fuel_gas_consumption = fraction_gas_consumed * H2_to_reinjection
    N2_fuel_gas_consumption = fraction_gas_consumed * N2_to_reinjection
    H2O_fuel_gas_consumption = fraction_gas_consumed * H2O_to_reinjection

    # # For fugitive losses, take the loss rates for the baseline case from the OPGEE model:

    # HC_gas_reinjection_compressor_loss_rates = np.array([
    #     0.000163391, 0.000163391, 0.000163391, 0.000163391, 0.000163391, 0.000163391, 0.000163391, 0.000163391,
    #     0.000163391, 0.000163391, 0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102,
    #     0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102,
    #     0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102, 0.000365102
    # ])

    # Rather than taking the loss rates for the Baseline case from the OPGEE model, calculate them here instead to consider the particulars of the selected case

    number_injection_wells = np.ceil(number_production_wells*0.25) #Assume 25% of production wells are used for reinjection
    injection_rate_per_well = total_gas_volume_flow_to_reinjection / number_injection_wells * 1000 #MSCFD/well
    
    HC_gas_reinjection_compressor_loss_rates = calculate_emissions_factors('Recip Compressor', case, sensitivity_variables,injection_well_flow_rate=injection_rate_per_well)['loss_rates_df']['Loss Rate'] # 

    df_HC_gas_reinjection_compressor_loss_rates = pd.DataFrame({
        'Year': range(1, len(HC_gas_reinjection_compressor_loss_rates) + 1),
        'HC Gas Reinjection Compressor Loss Rates': HC_gas_reinjection_compressor_loss_rates
    })

    # Calculate the CH4 and H2 components of fugitive emissions from HC gas reinjection compressors
    HC_gas_reinjection_compressor_fugitives_CH4 = df_HC_gas_reinjection_compressor_loss_rates['HC Gas Reinjection Compressor Loss Rates'].values * GWP_CH4 * CH4_to_reinjection 
    HC_gas_reinjection_compressor_fugitives_H2 = df_HC_gas_reinjection_compressor_loss_rates['HC Gas Reinjection Compressor Loss Rates'].values * GWP_H2 * H2_to_reinjection #Note the Brandt model does not include H2 emissions from this source. It is unclear why this is the case.

    # Calculate the mass flows to the reservoir after accounting for fugitive losses
    CH4_to_reservoir = CH4_to_reinjection - df_HC_gas_reinjection_compressor_loss_rates['HC Gas Reinjection Compressor Loss Rates'].values * CH4_to_reinjection - CH4_fuel_gas_consumption
    H2_to_reservoir = H2_to_reinjection - df_HC_gas_reinjection_compressor_loss_rates['HC Gas Reinjection Compressor Loss Rates'].values * H2_to_reinjection - H2_fuel_gas_consumption
    N2_to_reservoir = N2_to_reinjection - df_HC_gas_reinjection_compressor_loss_rates['HC Gas Reinjection Compressor Loss Rates'].values * N2_to_reinjection - N2_fuel_gas_consumption 
    H2O_to_reservoir = H2O_to_reinjection - df_HC_gas_reinjection_compressor_loss_rates['HC Gas Reinjection Compressor Loss Rates'].values * H2O_to_reinjection - H2O_fuel_gas_consumption
    total_mass_flow_to_reservoir = CH4_to_reservoir + H2_to_reservoir + N2_to_reservoir + H2O_to_reservoir

    # Sum the CH4 and H2 components to get the total fugitive emissions from HC gas reinjection compressors
    HC_gas_reinjection_compressor_fugitives = HC_gas_reinjection_compressor_fugitives_CH4 + HC_gas_reinjection_compressor_fugitives_H2

    # Calculate the emissions associated with combustion of the fuel gas. Use an emissions factor to get CO2e from the fuel gas consumption.
    # First need to convert the CH4 used in fuel gas from mass to mmbtu
    CH4_fuel_gas_consumption_mmbtu = CH4_fuel_gas_consumption * 1000 * LHV_density_gases_metric.loc['CH4', 'LHV (MJ/kg)'] / mmbtu_to_MJ #mmbtu/day
    HC_gas_reinjection_compressor_combustion_emissions = CH4_fuel_gas_consumption_mmbtu * reciprocating_compressor_ng_emissions_factor / 1E6 #tonnes CO2e/day
    # HC_gas_reinjection_compressor_combustion_emissions = 0 # Toggle to check the impact of these emissions on the total emissions

    return {
        'case': case,
        'CH4_to_reinjection': CH4_to_reinjection,
        'H2_to_reinjection': H2_to_reinjection,
        'N2_to_reinjection': N2_to_reinjection,
        'H2O_to_reinjection': H2O_to_reinjection,
        'HC_gas_reinjection_compressor_fugitives_CH4': HC_gas_reinjection_compressor_fugitives_CH4,
        'HC_gas_reinjection_compressor_fugitives_H2': HC_gas_reinjection_compressor_fugitives_H2,
        'HC_gas_reinjection_compressor_fugitives': HC_gas_reinjection_compressor_fugitives,
        'CH4_to_reservoir': CH4_to_reservoir,
        'H2_to_reservoir': H2_to_reservoir,
        'N2_to_reservoir': N2_to_reservoir,
        'H2O_to_reservoir': H2O_to_reservoir,
        'total_mass_flow_to_reservoir': total_mass_flow_to_reservoir,
        'gas_energy_density_metric': gas_energy_density_metric, #MJ/kg
        'gas_energy_density_imperial': gas_energy_density_imperial,
        'inlet_total_molar_flow': inlet_total_molar_flow,
        'inlet_pseduocritical_temp': inlet_pseduocritical_temp,
        'mol_frac_N2': mol_frac_N2,
        'inlet_pseudocritical_pressure': inlet_pseduocritical_pressure,
        'specific_heat_ratio': specific_heat_ratio,
        'compression_ratio_per_stage': compression_ratio_per_stage,
        'first_stage_outlet_temp_F': first_stage_outlet_temp_F,
        'first_stage_discharge_pressure': first_stage_discharge_pressure,
        'first_stage_adiabatic_work': first_stage_adiabatic_work,
        'total_compressor_adiabatic_work': total_compressor_adiabatic_work,
        'num_stages': num_stages,
        'compressor_bhp': compressor_bhp,
        'closest_bhp': closest_bhp,
        'compressor_fuel_efficiency': compressor_fuel_efficiency,
        'compressor_total_fuel_use': compressor_total_fuel_use,
        'fraction_gas_consumed': fraction_gas_consumed,
        'gas_energy_density_imperial': gas_energy_density_imperial,
        'CH4_fuel_gas_consumption': CH4_fuel_gas_consumption,
        'H2_fuel_gas_consumption': H2_fuel_gas_consumption,
        'N2_fuel_gas_consumption': N2_fuel_gas_consumption,
        'H2O_fuel_gas_consumption': H2O_fuel_gas_consumption,
        'HC_gas_reinjection_compressor_combustion_emissions': HC_gas_reinjection_compressor_combustion_emissions,
        'HC_gas_reinjection_compressor_loss_rates': HC_gas_reinjection_compressor_loss_rates,
        'injection_rate_per_well': injection_rate_per_well,
        'HC_gas_reinjection_compressor_loss_rates': df_HC_gas_reinjection_compressor_loss_rates,
        'total_gas_volume_flow_to_reinjection': total_gas_volume_flow_to_reinjection,
        # 'check_mean_gas_rates_df': check_mean_gas_rates_df,
        # 'check_bin_assignments_df': check_bin_assignments_df,
        # 'check_mean_gas_rates': check_mean_gas_rates
    }


#Test Usage:
HC_gas_reinjection_compressor_fugitives_info = calculate_HC_gas_reinjection_compressor_fugitives('Baseline')
# print(f"Case: {HC_gas_reinjection_compressor_fugitives_info['case']}, HC Gas Reinjection Compressor Fugitives: {HC_gas_reinjection_compressor_fugitives_info['CH4_to_reinjection']}") #Checked and confirmed aligned with Brandt's model
# print(f"Case: {HC_gas_reinjection_compressor_fugitives_info['case']}, HC Gas Reinjection Compressor Fugitives: {HC_gas_reinjection_compressor_fugitives_info['H2_to_reinjection']}") #Checked and confirmed aligned with Brandt's model
# print(f"Case: {HC_gas_reinjection_compressor_fugitives_info['case']}, HC Gas Reinjection Compressor Fugitives: {HC_gas_reinjection_compressor_fugitives_info['HC_gas_reinjection_compressor_combustion_emissions']}")
# print(f"Case: {HC_gas_reinjection_compressor_fugitives_info['case']}, HC Gas Reinjection Compressor Fugitives: {HC_gas_reinjection_compressor_fugitives_info['HC_gas_reinjection_compressor_fugitives_H2']}")
print(f"Case: {HC_gas_reinjection_compressor_fugitives_info['case']}, HC Gas Reinjection Compressor Fugitives: {HC_gas_reinjection_compressor_fugitives_info['HC_gas_reinjection_compressor_fugitives']}")

# # Print the types of the outputs to ensure they are consistent with the expected types
# print(type(HC_gas_reinjection_compressor_fugitives_info['HC_gas_reinjection_compressor_fugitives']))


# ## 2.4 HC gas injection wells
# 
# Paper assumes certain amount of fugitive emissions downstream of the reinjection compressors, at the injection wells. The amount of waste gas that is re-injected is the full amount of waste gas, less the amount of gas that is combusted to power the compressor(s).

# In[237]:


def calculate_HC_gas_reinjection_well_fugitives(case, sensitivity_variables=None):
    if sensitivity_variables:
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
    else:
        field_lifespan = field_lifespan_default
        GWP_H2 = GWP_H2_default
        number_production_wells = number_production_wells_default
        field_lifespan = field_lifespan_default

    # HC_gas_reinjection_well_loss_rates = np.array([
    #     0.001944255, 0.001944255, 0.001944255, 0.001944255, 0.001944255, 0.001944255, 0.001944255, 0.001944255,
    #     0.001944255, 0.001944255, 0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451,
    #     0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451,
    #     0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451, 0.005462451
    # ])

    # Calculate the loss rates, rather than relying on the Baseline case rates taken directly from OPGEE:
    # First, call calculate_HC_gas_reinjection_compressor_fugitives to get the values of flows to the reinjection wells
    HC_gas_reinjection_compressor_fugitives_info = calculate_HC_gas_reinjection_compressor_fugitives(case, sensitivity_variables)
    CH4_to_reservoir = HC_gas_reinjection_compressor_fugitives_info['CH4_to_reservoir']
    H2_to_reservoir = HC_gas_reinjection_compressor_fugitives_info['H2_to_reservoir']
    N2_to_reservoir = HC_gas_reinjection_compressor_fugitives_info['N2_to_reservoir']
    H2O_to_reservoir = HC_gas_reinjection_compressor_fugitives_info['H2O_to_reservoir']
    total_mass_flow_to_reservoir = HC_gas_reinjection_compressor_fugitives_info['total_mass_flow_to_reservoir']
    total_gas_volume_flow_to_reinjection = HC_gas_reinjection_compressor_fugitives_info['total_gas_volume_flow_to_reinjection']
    
    # Determinte the amount of gas injected per injection well
    number_injection_wells = np.ceil(number_production_wells*0.25) #Assume 25% of production wells are used for reinjection
    injection_rate_per_well = total_gas_volume_flow_to_reinjection / number_injection_wells * 1000 #MSCFD/well
    
    HC_gas_reinjection_well_loss_rates = calculate_emissions_factors('Well', case, sensitivity_variables,injection_well_flow_rate=injection_rate_per_well)['loss_rates_df']['Loss Rate'] #
    

    # Create the DataFrame
    df_HC_gas_reinjection_well_loss_rates = pd.DataFrame({
        'Year': range(1, len(HC_gas_reinjection_well_loss_rates) + 1),
        'HC Gas Reinjection Well Loss Rates': HC_gas_reinjection_well_loss_rates
    })

    # Calculate the CH4 and H2 components of fugitive emissions from HC gas reinjection wells
    HC_gas_reinjection_well_fugitives_CH4 = df_HC_gas_reinjection_well_loss_rates['HC Gas Reinjection Well Loss Rates'].values * GWP_CH4 * CH4_to_reservoir 
    HC_gas_reinjection_well_fugitives_H2 = df_HC_gas_reinjection_well_loss_rates['HC Gas Reinjection Well Loss Rates'].values * GWP_H2 * H2_to_reservoir

    # Sum the CH4 and H2 components to get the total fugitive emissions from HC gas reinjection wells
    HC_gas_reinjection_well_fugitives = HC_gas_reinjection_well_fugitives_CH4 + HC_gas_reinjection_well_fugitives_H2

    return {
        'case': case,
        'HC_gas_reinjection_well_fugitives': HC_gas_reinjection_well_fugitives,

    }


# # Test usage
# HC_gas_reinjection_well_fugitives_info = calculate_HC_gas_reinjection_well_fugitives('Baseline')
# print(f"Case: {HC_gas_reinjection_well_fugitives_info['case']}, HC Gas Reinjection Well Fugitives: {HC_gas_reinjection_well_fugitives_info['HC_gas_reinjection_well_fugitives']}") 


# ## 2.5 Separation

# In[238]:


#Definine a helper function for fugitive loss rates during separation:
def calculate_production_separation_fugitives(case, sensitivity_variables =None):
    if sensitivity_variables:
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
    else:
        GWP_H2 = GWP_H2_default
        field_lifespan = field_lifespan_default

    # Calculate the mass flows downstream of the separator
    mass_flows_after_separator = calculate_mass_flows_after_separator(case, sensitivity_variables)
    CH4_after_separator = mass_flows_after_separator['CH4_after_separator']
    H2_after_separator = mass_flows_after_separator['H2_after_separator']
    N2_after_separator = mass_flows_after_separator['N2_after_separator']
    CO2_after_separator = mass_flows_after_separator['CO2_after_separator']
    H2O_after_separator = mass_flows_after_separator['H2O_after_separator']

    # # Separation fugitive loss rates
    # separation_fugitive_loss_rates = [
    #     0.00061497, 0.000923162, 0.001252111, 0.001252111, 0.001281899,
    #     0.001694345, 0.001873018, 0.001873018, 0.001873018, 0.001873018,
    #     0.001873018, 0.001873018, 0.001873018, 0.001873018, 0.002178708,
    #     0.002287947, 0.002287947, 0.002287947, 0.002287947, 0.002862778,
    #     0.002862778, 0.002862778, 0.002862778, 0.002862778, 0.002862778,
    #     0.002867841, 0.002867841, 0.002867841, 0.002867841, 0.002867841
    # ]

    # Use the function to calculate the separation fugitive loss rates
    separation_fugitive_loss_rates = calculate_emissions_factors('Separator', case, sensitivity_variables)['loss_rates_df']['Loss Rate']

    # Create the DataFrame for separation fugitive losses
    df_separation_fugitive = pd.DataFrame({
        'Year': range(1, 31),
        'Separation Fugitive Losses %': separation_fugitive_loss_rates
    })

    # Calculate CH4 and H2 fugitive emissions and their total
    production_separation_fugitive_CH4 = df_separation_fugitive['Separation Fugitive Losses %'].values * GWP_CH4 * CH4_after_separator #Loss rates are saved as decimals, not percentage, so no need to divide by 100
    production_separation_fugitive_H2 = df_separation_fugitive['Separation Fugitive Losses %'].values  * GWP_H2 * H2_after_separator #Loss rates are saved as decimals, not percentage, so no need to divide by 100
    production_separation_fugitives = production_separation_fugitive_CH4 + production_separation_fugitive_H2

    #Now calculate losses for N2,CO2, and H2O which are not considered in the above calculations:
    N2_separation_losses = df_separation_fugitive['Separation Fugitive Losses %'].values * N2_after_separator 
    CO2_separation_losses = df_separation_fugitive['Separation Fugitive Losses %'].values * CO2_after_separator 
    H2O_separation_losses = df_separation_fugitive['Separation Fugitive Losses %'].values * H2O_after_separator

    return {
        'case': case,
        'production_separation_fugitive_CH4': production_separation_fugitive_CH4,
        'production_separation_fugitive_H2': production_separation_fugitive_H2,
        'production_separation_fugitives': production_separation_fugitives,
        'N2_separation_losses': N2_separation_losses,
        'CO2_separation_losses': CO2_separation_losses,
        'H2O_separation_losses': H2O_separation_losses    
    }

# #Test Usage:
# production_separation_fugitives_info = calculate_production_separation_fugitives('Baseline')
# print(f"Case: {production_separation_fugitives_info['case']}, Production Separation Fugitives: {production_separation_fugitives_info['production_separation_fugitives']}")

# #Print the value of N2 separation losses for the 'Baseline' case:
# production_separation_fugitives_info = calculate_production_separation_fugitives('Baseline')
# print(f"N2 separation losses for the 'Baseline' case: {production_separation_fugitives_info['N2_separation_losses']}")


# In[239]:


# Now we have determined the mass flows after the separator and the fugitive losses during separation, we can back-calculate the mass flows at the wellhead.

def calculate_mass_flows_upstream_separator(case, sensitivity_variables=None):
    """
    Calculate the mass flows of each component of the gas stream upstream of the separator for a given case and sensitivity values.

    Parameters:
    case (str): The case for which to calculate the mass flows.
    sensitivity_variables (dict): A dictionary containing sensitivity values for the parameters. The keys are the parameter names and the values are the sensitivity values.

    Returns:
    dict: A dictionary containing the mass flows of each component of the gas stream upstream of the separator in kg/day.
    """
    if sensitivity_variables:
        oil_production = sensitivity_variables['oil_production']
        GWP_H2 = sensitivity_variables['GWP_H2']
    else:
        oil_production = oil_production_default
        GWP_H2 = GWP_H2_default

    # Calculate the mass flows downstream of the separator
    mass_flows_after_separator = calculate_mass_flows_after_separator(case, sensitivity_variables)
    CH4_after_separator = mass_flows_after_separator['CH4_after_separator']
    H2_after_separator = mass_flows_after_separator['H2_after_separator']
    N2_after_separator = mass_flows_after_separator['N2_after_separator']
    CO2_after_separator = mass_flows_after_separator['CO2_after_separator']
    H2O_after_separator = mass_flows_after_separator['H2O_after_separator']

    production_separation_fugitives_info = calculate_production_separation_fugitives(case,sensitivity_variables)
    production_separation_fugitive_CH4 = production_separation_fugitives_info['production_separation_fugitive_CH4']
    production_separation_fugitive_H2 = production_separation_fugitives_info['production_separation_fugitive_H2']
    N2_separation_losses = production_separation_fugitives_info['N2_separation_losses']
    CO2_separation_losses = production_separation_fugitives_info['CO2_separation_losses']
    H2O_separation_losses = production_separation_fugitives_info['H2O_separation_losses']

    # Calculate the mass flows of each component of the gas stream upstream of the separator
    CH4_upstream_separator = CH4_after_separator + production_separation_fugitive_CH4/GWP_CH4
    H2_upstream_separator = H2_after_separator + production_separation_fugitive_H2/GWP_H2
    N2_upstream_separator = N2_after_separator + N2_separation_losses
    CO2_upstream_separator = CO2_after_separator + CO2_separation_losses
    H2O_upstream_separator = H2O_after_separator + H2O_separation_losses
    total_gas_mass_flow_upstream_separator = CH4_upstream_separator + H2_upstream_separator + N2_upstream_separator + CO2_upstream_separator + H2O_upstream_separator

    return {
        'case': case,
        'CH4_upstream_separator': CH4_upstream_separator,
        'H2_upstream_separator': H2_upstream_separator,
        'N2_upstream_separator': N2_upstream_separator,
        'CO2_upstream_separator': CO2_upstream_separator,
        'H2O_upstream_separator': H2O_upstream_separator,
        'total_gas_mass_flow_upstream_separator': total_gas_mass_flow_upstream_separator,
        'N2_separation_losses': N2_separation_losses,
    }
    # #Similarly call the production separation fugitives function to get the CH4 and H2 flow rates needed for the calculation below:
    
# # Test Usage:
# mass_flows_upstream_separator_info = calculate_mass_flows_upstream_separator('Baseline')
# print(mass_flows_upstream_separator_info['N2_separation_losses'])
    
    # lifetime_field_energy_production = calculate_exploration_emissions(case,sensitivity_variables)['lifetime_field_energy_production']

    # development_drilling_fuel_consumption = drilling_fuel_per_foot_vertical * field_depth * total_number_wells

    # #Next, calculate the energy content of this fuel:
    # development_drilling_energy_consumption = development_drilling_fuel_consumption * diesel_energy_density / 1E6 #mmbtu LHV

    # #Next, calculate the wellhead energy production of the field. This is comprised of energy of gas at the wellhead, plus the energy content of oil at the wellhead:
    # #Need to calculate the mass of gas at the wellhead. This is derived by adding the losses during separation to the flows out of the separation unit 


# ## 2.6 Total operational VFF emissions
# This is the sum total of the calculations in Sections 2.1 and 2.2, above.
# 

# In[240]:


def calculate_total_operational_VFF_emissions(case, sensitivity_variables=None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
    else:
        oil_production = oil_production_default
        GWP_H2 = GWP_H2_default

    # Calculate vent emissions
    vent_emissions = calculate_total_production_vent_emissions(case, sensitivity_variables)
    production_vent_completions_emissions = vent_emissions['production_vent_completions_emissions']
    production_vent_workovers_emissions = vent_emissions['production_vent_workovers_emissions']
    production_vent_liquids_emissions = vent_emissions['production_vent_liquids_emissions']

    # Calculate fugitive emissions from the production phase
    fugitive_emissions = calculate_total_production_fugitive_emissions(case, sensitivity_variables)
    production_fugitive_wellhead = fugitive_emissions['production_fugitive_wellhead']
    production_fugitive_meter = fugitive_emissions['production_fugitive_meter']
    production_fugitive_dehydrator = fugitive_emissions['production_fugitive_dehydrator']
    production_fugitive_compressor = fugitive_emissions['production_fugitive_compressor']
    production_fugitive_heater = fugitive_emissions['production_fugitive_heater']
    production_fugitive_chempumps = fugitive_emissions['production_fugitive_chempumps']

    # Calculate gas gathering and dehydration vents emissions
    gas_gathering_fugitives = calculate_gas_gathering_fugitives(case,sensitivity_variables)['gas_gathering_fugitives']
    gas_dehydration_vents = calculate_gas_dehydration_vents(case,sensitivity_variables)['gas_dehydration_vents']

    # These functions need to be defined or modified to return numerical values
    HC_gas_reinjection_compressor_fugitives = calculate_HC_gas_reinjection_compressor_fugitives(case,sensitivity_variables)['HC_gas_reinjection_compressor_fugitives']
    HC_gas_reinjection_well_fugitives = calculate_HC_gas_reinjection_well_fugitives(case,sensitivity_variables)['HC_gas_reinjection_well_fugitives']
    production_separation_fugitives = calculate_production_separation_fugitives(case,sensitivity_variables)['production_separation_fugitives']

    # Summing all components
    total_operational_VFF_emissions = (
        production_vent_completions_emissions +
        production_vent_workovers_emissions +
        production_vent_liquids_emissions +
        production_fugitive_wellhead +
        production_fugitive_meter +
        production_fugitive_dehydrator +
        production_fugitive_compressor +
        production_fugitive_heater +
        production_fugitive_chempumps +
        gas_gathering_fugitives +
        gas_dehydration_vents +
        HC_gas_reinjection_compressor_fugitives +
        HC_gas_reinjection_well_fugitives +
        production_separation_fugitives
    ) * 1000  # Convert from tCO2e/day to kgCO2e/day

    # Calculate the relative (percentage) contributions of each component of total_operational_VFF_emissions
    total_operational_VFF_emissions_component_percentages = {
        'production_vent_completions_emissions': (production_vent_completions_emissions * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_vent_workovers_emissions': (production_vent_workovers_emissions * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_vent_liquids_emissions': (production_vent_liquids_emissions * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_fugitive_wellhead': (production_fugitive_wellhead * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_fugitive_meter': (production_fugitive_meter * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_fugitive_dehydrator': (production_fugitive_dehydrator * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_fugitive_compressor': (production_fugitive_compressor * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_fugitive_heater': (production_fugitive_heater * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_fugitive_chempumps': (production_fugitive_chempumps * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'gas_gathering_fugitives': (gas_gathering_fugitives * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'gas_dehydration_vents': (gas_dehydration_vents * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'HC_gas_reinjection_compressor_fugitives': (HC_gas_reinjection_compressor_fugitives * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'HC_gas_reinjection_well_fugitives': (HC_gas_reinjection_well_fugitives * 1000).sum() / total_operational_VFF_emissions.sum() * 100,
        'production_separation_fugitives': (production_separation_fugitives * 1000).sum() / total_operational_VFF_emissions.sum() * 100
    }

    # Calculate the total lifetime operational VFF emissions by summing emissions over each year
    total_lifetime_operational_VFF_emissions = (total_operational_VFF_emissions * 365).sum()  # kgCO2e/lifetime

    # Calculate the relative (percentage) contributions of each component of total lifetime operational VFF emissions
    total_lifetime_operational_VFF_emissions_component_percentages = {
        'lifetime_production_vent_completions_emissions': ((production_vent_completions_emissions * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions) * 100,
        'lifetime_production_vent_workovers_emissions': (production_vent_workovers_emissions * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_vent_liquids_emissions': (production_vent_liquids_emissions * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_fugitive_wellhead': (production_fugitive_wellhead * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_fugitive_meter': (production_fugitive_meter * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_fugitive_dehydrator': (production_fugitive_dehydrator * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_fugitive_compressor': (production_fugitive_compressor * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_fugitive_heater': (production_fugitive_heater * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_fugitive_chempumps': (production_fugitive_chempumps * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_gas_gathering_fugitives': (gas_gathering_fugitives * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_gas_dehydration_vents': (gas_dehydration_vents * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_HC_gas_reinjection_compressor_fugitives': (HC_gas_reinjection_compressor_fugitives * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_HC_gas_reinjection_well_fugitives': (HC_gas_reinjection_well_fugitives * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100,
        'lifetime_production_separation_fugitives': (production_separation_fugitives * 365 * 1000).sum() / total_lifetime_operational_VFF_emissions * 100
    }

    return {
        'case': case,
        'total_operational_VFF_emissions': total_operational_VFF_emissions,  # kgCO2e/day
        'total_operational_VFF_emissions_component_percentages': total_operational_VFF_emissions_component_percentages,  # Percentage contributions of each component
        'production_vent_completions_emissions': production_vent_completions_emissions,
        'production_vent_workovers_emissions': production_vent_workovers_emissions,
        'production_vent_liquids_emissions': production_vent_liquids_emissions,
        'production_fugitive_wellhead': production_fugitive_wellhead,
        'production_fugitive_meter': production_fugitive_meter,
        'production_fugitive_dehydrator': production_fugitive_dehydrator,
        'production_fugitive_compressor': production_fugitive_compressor,
        'production_fugitive_heater': production_fugitive_heater,
        'production_fugitive_chempumps': production_fugitive_chempumps,
        'gas_gathering_fugitives': gas_gathering_fugitives,
        'gas_dehydration_vents': gas_dehydration_vents,
        'HC_gas_reinjection_compressor_fugitives': HC_gas_reinjection_compressor_fugitives,
        'HC_gas_reinjection_well_fugitives': HC_gas_reinjection_well_fugitives,
        'production_separation_fugitives': production_separation_fugitives,
        'total_lifetime_operational_VFF_emissions': total_lifetime_operational_VFF_emissions,  # kgCO2e/lifetime
        'total_lifetime_operational_VFF_emissions_component_percentages': total_lifetime_operational_VFF_emissions_component_percentages  # Percentage contributions of each component
    }

# Example usage
percentages = calculate_total_operational_VFF_emissions('Baseline')['total_lifetime_operational_VFF_emissions_component_percentages']
total = []
for component, percentage in percentages.items():
    print(f'{component}: {percentage:.2f}%')
    total = total + [percentage]
    
print(sum(total))

#Plot the total lifetime operational VFF emissions components for the baseline case:

def plot_total_lifetime_operational_VFF_emissions_component_percentages(case):
    # Calculate the total lifetime operational VFF emissions components
    total_lifetime_operational_VFF_emissions_component_percentages = calculate_total_operational_VFF_emissions(case)['total_lifetime_operational_VFF_emissions_component_percentages']

    # Create a DataFrame for the components
    df_total_lifetime_operational_VFF_emissions_component_percentages = pd.DataFrame({
        'Component': list(total_lifetime_operational_VFF_emissions_component_percentages.keys()),
        'Percentage': list(total_lifetime_operational_VFF_emissions_component_percentages.values())
    })

    # Plot the components
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Percentage', y='Component', data=df_total_lifetime_operational_VFF_emissions_component_percentages, palette='viridis')
    ax.set_title('Total Lifetime Operational VFF Emissions Components', fontsize=16)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Component', fontsize=12)
    plt.show()

# # Example usage
plot_total_lifetime_operational_VFF_emissions_component_percentages('Baseline')


# In[241]:


# # Create two categories for the fugitive emissions: varying with production and constant with production
# 1. Fugitive emissions calculated with factors that vary with varying production are:
# Liquids unloadings
# Wellhead fugitive losses
# Meter fugitive losses
# Dehydrator fugitive losses
# Compressor fugitive losses
# Heater fugitive losses
# Chemical pump fugitive losses
# Separation fugitive losses

# 2. Fugitive emissions calculated with factors that remain constant with varying production are:
# Fugitives from completions
# Fugitives from workovers
# Surface processing venting
# Gas gathering fugitives

# Create a helper function to calculate the total fugitive emissions varying with production:
# This function will calculate the total fugitive emissions varying with production by summing the emissions from the following components:
# Liquids unloadings, wellhead fugitive losses, meter fugitive losses, dehydrator fugitive losses, compressor fugitive losses, heater fugitive losses, chemical pump fugitive losses, and separation fugitive losses
# This function will use 'total_lifetime_operational_VFF_emissions': total_lifetime_operational_VFF_emissions, and 'total_lifetime_operational_VFF_emissions_component_percentages': total_lifetime_operational_VFF_emissions_component_percentages

def calculate_total_fugitive_emissions_varying_with_production(case,sensitivity_variables):
    # Calculate the total lifetime operational VFF emissions and its components
    total_operational_VFF_emissions_info = calculate_total_operational_VFF_emissions(case,sensitivity_variables)
    total_lifetime_operational_VFF_emissions = total_operational_VFF_emissions_info['total_lifetime_operational_VFF_emissions']
    total_lifetime_operational_VFF_emissions_component_percentages = total_operational_VFF_emissions_info['total_lifetime_operational_VFF_emissions_component_percentages']

    # Extract the percentage contribution of each relevant component of the total lifetime operational VFF emissions
    lifetime_production_vent_completions_emissions = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_vent_completions_emissions']
    lifetime_production_vent_workovers_emissions = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_vent_workovers_emissions']
    lifetime_production_vent_liquids_emissions = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_vent_liquids_emissions']
    lifetime_production_fugitive_wellhead = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_fugitive_wellhead']
    lifetime_production_fugitive_meter = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_fugitive_meter']
    lifetime_production_fugitive_dehydrator = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_fugitive_dehydrator']
    lifetime_production_fugitive_compressor = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_fugitive_compressor']
    lifetime_production_fugitive_heater = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_fugitive_heater']
    lifetime_production_fugitive_chempumps = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_fugitive_chempumps']
    lifetime_production_separation_fugitives = total_lifetime_operational_VFF_emissions_component_percentages['lifetime_production_separation_fugitives']

    # Sum the fugitive emissions varying with production
    total_fugitive_emissions_varying_with_production = (
        lifetime_production_vent_liquids_emissions +
        lifetime_production_fugitive_wellhead +
        lifetime_production_fugitive_meter +
        lifetime_production_fugitive_dehydrator +
        lifetime_production_fugitive_compressor +
        lifetime_production_fugitive_heater +
        lifetime_production_fugitive_chempumps +
        lifetime_production_separation_fugitives
    ) * total_lifetime_operational_VFF_emissions / 100 # Convert from percentage to decimal

    # Calculate this as a percentage of the total lifetime operational VFF emissions
    percent_total_fugitive_emissions_varying_with_production = total_fugitive_emissions_varying_with_production / total_lifetime_operational_VFF_emissions * 100

    return {
        'case': case,
        'total_fugitive_emissions_varying_with_production': total_fugitive_emissions_varying_with_production,
        'percent_total_fugitive_emissions_varying_with_production': percent_total_fugitive_emissions_varying_with_production
    }

# # # Example usage
# total_fugitive_emissions_varying_with_production_info = calculate_total_fugitive_emissions_varying_with_production('Baseline')
# print(f"Total Fugitive Emissions Varying with Production for Baseline: {total_fugitive_emissions_varying_with_production_info['total_fugitive_emissions_varying_with_production']} kgCO2e/lifetime")
# print(f"Percentage of Total VFF Emissions Varying with Production for Baseline: {total_fugitive_emissions_varying_with_production_info['percent_total_fugitive_emissions_varying_with_production']} %")


# In[242]:


# Create a helper function to calculate the total fugitive emissions constant with production:
# This function will calculate the total fugitive emissions constant with production by summing the emissions from the following components:
# Fugitives from completions, fugitives from workovers, surface processing venting, and gas gathering fugitives
# The function will return the total fugitive emissions constant with production in kgCO2e/day
# The function will also return the percentage contribution of each component to the total fugitive emissions constant with production

def calculate_total_fugitive_emissions_constant_with_production(case,sensitivity_variables):

    # Calculate the total production vent emissions
    total_production_vent_emissions_info = calculate_total_production_vent_emissions(case,sensitivity_variables)
    production_vent_completions_emissions = total_production_vent_emissions_info['production_vent_completions_emissions']
    production_vent_workovers_emissions = total_production_vent_emissions_info['production_vent_workovers_emissions']
    production_vent_liquids_emissions = total_production_vent_emissions_info['production_vent_liquids_emissions']

    # Calculate the gas gathering fugitives
    gas_gathering_fugitives = calculate_gas_gathering_fugitives(case,sensitivity_variables)['gas_gathering_fugitives']

    # Sum the emissions from the components
    total_fugitive_emissions_constant_with_production = (
        production_vent_completions_emissions +
        production_vent_workovers_emissions +
        production_vent_liquids_emissions +
        gas_gathering_fugitives
    ) * 1000  # Convert from tCO2e/day to kgCO2e/day

    # Calculate the relative (percentage) contributions of each component of total_fugitive_emissions_constant_with_production:
    total_fugitive_emissions_constant_with_production_components = {
        'production_vent_completions_emissions': production_vent_completions_emissions / total_fugitive_emissions_constant_with_production * 100,
        'production_vent_workovers_emissions': production_vent_workovers_emissions / total_fugitive_emissions_constant_with_production * 100,
        'production_vent_liquids_emissions': production_vent_liquids_emissions / total_fugitive_emissions_constant_with_production * 100,
        'gas_gathering_fugitives': gas_gathering_fugitives / total_fugitive_emissions_constant_with_production * 100
    }

    # Calculate the relative contribution of the total fugitive emissions constant with production to the total operational VFF emissions
    total_operational_VFF_emissions = calculate_total_operational_VFF_emissions(case,sensitivity_variables)['total_operational_VFF_emissions']
    total_fugitive_emissions_constant_with_production_percentage = total_fugitive_emissions_constant_with_production / total_operational_VFF_emissions * 100
    total_lifetime_operational_VFF_emissions = calculate_total_operational_VFF_emissions(case,sensitivity_variables)['total_lifetime_operational_VFF_emissions']
    total_lifetime_fugitive_emissions_constant_with_production = (total_fugitive_emissions_constant_with_production*365).sum()
    percentage_lifetime_fugitive_emissions_constant_with_production = total_lifetime_fugitive_emissions_constant_with_production / total_lifetime_operational_VFF_emissions * 100

    return {
        'case': case,
        'total_fugitive_emissions_constant_with_production': total_fugitive_emissions_constant_with_production,
        'total_fugitive_emissions_constant_with_production_components': total_fugitive_emissions_constant_with_production_components,
        'total_fugitive_emissions_constant_with_production_percentage': total_fugitive_emissions_constant_with_production_percentage,
        'total_lifetime_fugitive_emissions_constant_with_production': total_lifetime_fugitive_emissions_constant_with_production,
        'percentage_lifetime_fugitive_emissions_constant_with_production': percentage_lifetime_fugitive_emissions_constant_with_production
    }

# # Example usage
# case = 'Baseline'
# total_fugitive_emissions_constant_with_production_info = calculate_total_fugitive_emissions_constant_with_production(case)
# total_fugitive_emissions_constant_with_production = total_fugitive_emissions_constant_with_production_info['total_fugitive_emissions_constant_with_production']
# # print(f"Total Fugitive Emissions Constant with Production for {case}: {total_fugitive_emissions_constant_with_production} kgCO2e/day")
# print(f"The percentage contribution of the total fugitive emissions constant with production to the total operational VFF emissions for {case} is {total_fugitive_emissions_constant_with_production_info['total_fugitive_emissions_constant_with_production_percentage']}%")

#Check that the sum of the total fugitive emissions varying with production and the total fugitive emissions constant with production is equal to the total operational VFF emissions:

# total_lifetime_operational_VFF_emissions = calculate_total_operational_VFF_emissions('Baseline')['total_lifetime_operational_VFF_emissions']
# total_lifetime_fugitive_emissions_varying_with_production = calculate_total_fugitive_emissions_varying_with_production('Baseline')['total_lifetime_fugitive_emissions_varying_with_production']
# total_lifetime_fugitive_emissions_constant_with_production = calculate_total_fugitive_emissions_constant_with_production('Baseline')['total_lifetime_fugitive_emissions_constant_with_production']

# total_lifetime_fugitive_emissions = total_lifetime_fugitive_emissions_varying_with_production + total_lifetime_fugitive_emissions_constant_with_production

# assert total_lifetime_operational_VFF_emissions == total_lifetime_fugitive_emissions, "The sum of the total fugitive emissions varying with production and the total fugitive emissions constant with production should be equal to the total operational VFF emissions"

# # Example usage
# total_fugitive_emissions_constant_with_production_info = calculate_total_fugitive_emissions_constant_with_production('Baseline')
# # total_fugitive_emissions_varying_with_production_percentage = total_fugitive_emissions_varying_with_production_info['total_fugitive_emissions_varying_with_production_percentage']
# # print(f"Total Fugitive Emissions Varying with Production for {'Baseline'}: {total_fugitive_emissions_varying_with_production_percentage} kgCO2e/day")
# print(f"The percentage contribution of the total fugitive emissions constant with production to the total operational VFF emissions for {'Baseline'} is {total_fugitive_emissions_constant_with_production_info['percentage_lifetime_fugitive_emissions_constant_with_production'] }%")


# # 3. Drilling Energy-Use & VFF
# 
# Brandt's OPGEE analysis combines estimates of emissions from exploration activities together with drilling & development activities. The source of these emissions is assumed to be limited to the running of diesel engines required to complete these activities.

# ## 3.1 Exploration
# 
# Brandt's paper assumes that the only emissions associated with exploration is combustion in diesel engines. He estimates the energy consumption from survey vehicles (assumed to be heavy duty trucks for onshore assets), the expected emissions from drilling activities, and then calculates a ratio of expected total energy consumption from these two sources to the expected total energy produced from the the H2 in the field ("Fractional Energy Consumption").
# 
# Separately, the daily production rate of the asset (in energy terms) is calculated (i.e. H2 leaving the PSA unit). Daily energy consumed burning diesel is thus the Fractional Energy Consumption ratio multiplied by the daily production rate. Daily energy consumption is converted to daily emissions by multiplying by an 'emissions factor' for heavy duty trucks.
# 
# Emissions factor is taken from GREET 1_2016, sheet 'EF', Table 2.2 and 2.3, "Emission Factors of Fuel Combustion: Feedstock and Fuel Transportation from Product Origin to Product Destination back to Poduct Origin (grams per mmbtu of fuel burned)".

# In[243]:


#Note the functions/calculations below use constants/assumptions defined previously. 

#Now define a function to calculate the emissions associated with exploration that vary with the case:
def calculate_exploration_emissions(case, sensitivity_variables=None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
        field_depth = sensitivity_variables.get('field_depth', depths[case])
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        oil_production = oil_production_default
        field_lifespan = field_lifespan_default
        field_depth = depths[case]
        GWP_H2 = GWP_H2_default
        number_production_wells = number_production_wells_default

    #Calculate the 
        
    #Call the gas dehydration vents function to get the H2 flow rate after dehydration:
    gas_dehydration_vents_info = calculate_gas_dehydration_vents(case,sensitivity_variables)
    H2_after_dehy = gas_dehydration_vents_info['H2_after_dehy']

    total_number_exploration_wells = number_dry_wells + number_exploration_wells

    drilling_diesel_consumption = drilling_fuel_per_foot_vertical * field_depth * total_number_exploration_wells #Brandt paper assumes all wells are vertical, so can ignore calcs relating to horizontal drilling.
    drilling_energy_consumption_by_field = drilling_diesel_consumption * diesel_energy_density/1E6 #mmbtu LHV

    energy_intensity_per_well = drilling_energy_consumption_by_field / total_number_exploration_wells

    survey_vehicle_energy_consumption = heavy_duty_truck_diesel_intensity * weight_land_survey * distance_survey / 1E6
    drilling_energy_consumption = (number_dry_wells + number_exploration_wells) * energy_intensity_per_well

    daily_field_H2_exported = H2_after_dehy * (1 - PSA_unit_slippage_rate) #tonnes/day. It is assumed that 10% of H2 entering the PSA unit is lost.

    daily_energy_from_produced_oil = 0.475092430879946 #mmBTU/day. This is the energy content of the oil produced, as calculated in the OPGEE model.

    daily_field_energy_exported = daily_field_H2_exported * LHV_H2 + daily_energy_from_produced_oil  #mmbtu/day. Energy exported is just the H2 that leaves the processing facility, i.e. Mass * heating value.

    lifetime_field_energy_production = daily_field_energy_exported * 365 * field_lifespan #mmbtu. Total energy produced over the life of the field.

    #Now calculate the fraction of energy spent while exploring to the total saleable energy exported from the asset:
    fractional_energy_consumption = (survey_vehicle_energy_consumption + drilling_energy_consumption) / lifetime_field_energy_production

    #The rate of GHG emissions per unit of energy produced is thus:
    exploration_GHG_emission_rate = fractional_energy_consumption * emissions_factor_trucks #gCO2 eq./mmbtu crude

    #Next, the equivalent daily energy use is the fractional rate of energy consumption multiplied by the daily energy production:
    exploration_daily_energy_use = fractional_energy_consumption * daily_field_energy_exported

    #Finally, the emissions attributed to exploration is the daily energy use, multiplied by the relevant emissions factor:
    exploration_emissions = exploration_daily_energy_use * emissions_factor_diesel_exploration / 1E6 #tCO2eq/d

    return {
        'case': case,
        'exploration_emissions': exploration_emissions,
        'exploration_daily_energy_use': exploration_daily_energy_use,
        'exploration_GHG_emission_rate': exploration_GHG_emission_rate,
        'fractional_energy_consumption': fractional_energy_consumption,
        'daily_field_energy_exported': daily_field_energy_exported,
        'lifetime_field_energy_production': lifetime_field_energy_production,
        'daily_field_H2_exported': daily_field_H2_exported
    }
#     # Test Usage:
    
# exploration_emissions_info = calculate_exploration_emissions('High CH4')
# print(f"Case: {exploration_emissions_info['case']}, Exploration Emissions: {exploration_emissions_info['exploration_emissions']}")


# ## 3.2 Drilling & Development
# 
# This section takes a very similar approach to Section 3.1. It estimates the energy required (in the form of diesel) to develop the field (i.e. under the default assumptions, drill 50x production wells plus 13 injection wells) and then uses an emissions factor to convert this energy consumption into GHG emissions. 

# In[244]:


#Now define a function to calculate the emissions associated with development that vary with the case:
def calculate_development_drilling_emissions(case,sensitivity_variables=None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
    else:
        oil_production = oil_production_default
        number_production_wells = number_production_wells_default
        GWP_H2 = GWP_H2_default

    #Calculate the number of wells in the field based on the sensitifity variable:
    number_injection_wells = math.ceil(number_production_wells * 0.25) #Assume 25% of wells are injection wells
    total_number_wells = number_production_wells + number_injection_wells

    #First calculate the total development energy for the entire field. There is no fracturing, so the only energy use is in drilling the wells.
    #First step of this calculation is to estimate the drilling fuel consumption:
    field_depth = depths[case]

    # Retrieve the mass flows downstream of the separator
    mass_flows_after_separator = calculate_mass_flows_after_separator(case, sensitivity_variables)
    CH4_after_separator = mass_flows_after_separator['CH4_after_separator']
    H2_after_separator = mass_flows_after_separator['H2_after_separator']
    N2_after_separator = mass_flows_after_separator['N2_after_separator']
    CO2_after_separator = mass_flows_after_separator['CO2_after_separator']
    H2O_after_separator = mass_flows_after_separator['H2O_after_separator']

    # Retrieve the mass flows upstream of the separator:
    mass_flows_upstream_separator = calculate_mass_flows_upstream_separator(case, sensitivity_variables) #tonnes/day
    CH4_upstream_separator = mass_flows_upstream_separator['CH4_upstream_separator']
    H2_upstream_separator = mass_flows_upstream_separator['H2_upstream_separator']
    N2_upstream_separator = mass_flows_upstream_separator['N2_upstream_separator']
    CO2_upstream_separator = mass_flows_upstream_separator['CO2_upstream_separator']
    H2O_upstream_separator = mass_flows_upstream_separator['H2O_upstream_separator']
    total_gas_mass_flow_upstream_separator = mass_flows_upstream_separator['total_gas_mass_flow_upstream_separator']

    
    #Similarly call the production separation fugitives function to get the CH4 and H2 flow rates needed for the calculation below:
    production_separation_fugitives_info = calculate_production_separation_fugitives(case,sensitivity_variables)
    production_separation_fugitive_CH4 = production_separation_fugitives_info['production_separation_fugitive_CH4']
    production_separation_fugitive_H2 = production_separation_fugitives_info['production_separation_fugitive_H2']
    N2_separation_losses = production_separation_fugitives_info['N2_separation_losses']
    CO2_separation_losses = production_separation_fugitives_info['CO2_separation_losses']
    H2O_separation_losses = production_separation_fugitives_info['H2O_separation_losses']
    
    lifetime_field_energy_production = calculate_exploration_emissions(case,sensitivity_variables)['lifetime_field_energy_production']

    development_drilling_fuel_consumption = drilling_fuel_per_foot_vertical * field_depth * total_number_wells

    #Next, calculate the energy content of this fuel:
    development_drilling_energy_consumption = development_drilling_fuel_consumption * diesel_energy_density / 1E6 #mmbtu LHV

    #Next, calculate the wellhead energy production of the field. This is comprised of energy of gas at the wellhead, plus the energy content of oil at the wellhead:

    #Now determine the energy density of the gas produced. This is a weighted average of the energy densities of the individual gases, weighted by their mass fractions.
    gas_energy_density = (CH4_upstream_separator * LHV_density_gases_metric.loc['CH4','LHV (MJ/kg)'] + H2_upstream_separator * LHV_density_gases_metric.loc['H2','LHV (MJ/kg)'] + N2_upstream_separator * LHV_density_gases_metric.loc['N2','LHV (MJ/kg)'] + CO2_upstream_separator * LHV_density_gases_metric.loc['CO2','LHV (MJ/kg)']) / total_gas_mass_flow_upstream_separator
    wellhead_gas_energy = total_gas_mass_flow_upstream_separator * 1000 * gas_energy_density / mmbtu_to_MJ #mmbtu/day. This is the energy content of the gas produced, as calculated in the OPGEE model.
    wellhead_oil_energy = 0.517752042726451 #mmbtu/day. This is the energy content of the oil produced, as calculated in the OPGEE model.
    wellhead_total_energy = wellhead_gas_energy + wellhead_oil_energy

    #Now calculate the energy use as a fraction of the total energy produced and convert to emissions:
    development_drilling_energy = development_drilling_energy_consumption * wellhead_total_energy / lifetime_field_energy_production #mmbtu/day
    development_drilling_emissions = development_drilling_energy * emissions_factor_diesel_drilling / 1E6 #tCO2eq/d

    return {
        'case': case,
        'development_drilling_emissions': development_drilling_emissions, #tCO2eq/d
        'development_drilling_energy': development_drilling_energy,
        'wellhead_total_energy': wellhead_total_energy,
        'wellhead_oil_energy': wellhead_oil_energy,
        'wellhead_gas_energy': wellhead_gas_energy,
        'gas_energy_density': gas_energy_density,
        'total_gas_mass_flow_upstream_separator': total_gas_mass_flow_upstream_separator,
        'CO2_upstream_separator': CO2_upstream_separator,
        'N2_upstream_separator': N2_upstream_separator,
        'H2_upstream_separator': H2_upstream_separator,
        'CH4_upstream_separator': CH4_upstream_separator,
        'H2O_upstream_separator': H2O_upstream_separator,
        'development_drilling_energy_consumption': development_drilling_energy_consumption,
        'development_drilling_fuel_consumption': development_drilling_fuel_consumption
    }

# # Test Usage:
# development_drilling_emissions_info = calculate_development_drilling_emissions('Baseline')
# print(f"Case: {development_drilling_emissions_info['case']}, Development Drilling Emissions: {development_drilling_emissions_info['development_drilling_emissions']}")

#Code to return total_gas_mass_flow_upstream_separator for all cases. This is used in the Summary plotting section below.
total_gas_mass_flow_upstream_separator = {}
for case in cases:
    total_gas_mass_flow_upstream_separator[case] = calculate_development_drilling_emissions(case)['total_gas_mass_flow_upstream_separator']
# print(total_gas_mass_flow_upstream_separator)



# # 4. Embodied Emissions
# 
# Embodied emissions are those emmisions associated with the production of the physical equipment that is installed to enable production of the geologic H2. Given the assumption of no fracturing of wells, the relevant material categories are as follows:
# 
# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQUAAAA9CAIAAAArquuyAAAAAXNSR0IArs4c6QAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAABBaADAAQAAAABAAAAPQAAAAAiYXttAAAPnElEQVR4Ae2de/BWUxfHdVGJlEIiurim6KJEBo2UktuMxtQwjcl1MDT+EF2mXHMZGZWmNLqISG5jKqaSEmOYSsqlonIdSgxvFCnv+6k1rWe/e++zf+f39HvqnJ59/nhmPeusvfba6+y1r9+zT7VPP/30gHhFD0QP7PJATX5bt26dX2/8/vvvdevWPfDAA/NbhGh5djxQXU3ZsmVL9erVq/3/NW3aNBXwErNnzz7qqKO8t1xmpYTd5C6nf//+Rx55ZOm6uL/++uupp576448/3KyTOG+//fYRRxyRdDfyM+6Bnf2DXP/ddb3wwgtnnHHGbt4BFdb1zp07z5gxQ+XDRKWEw6q4u23btueee27BggXt27evULg4gc2bN992222XXnrpIYccklIDXvz3339TCkexzHmAxlUiQVrB9957T/7q7+LFi3v06HHXXXc1bdq0V69ec+fO7dSpU5MmTYYMGYLM+++/D1OEhw8ffuKJJyI2ePBg6gRMi2MKE0Unn3xyvXr1evbsuX79eoTJ6LLLLrv77rvRcNpppy1atChJrfDJF28iiarevXvfe++9rVq14pZXc6AIos2b19lnn00WqP3xxx8xj3imgejTp8+GDRtE3mXOmzevYcOGps5I58gDB1jxQHV8Zvc1efJkSjJr1izqBBVu6tSphAH0yF0XBPWYuwwPEHv33XcPP/zwhQsXTpo0idaUiuJyVHj58uU1atS48cYbkbnwwgs7duxI/EhG1HIIOGeddZZXLUy56BmwYcqUKW+88QbECSecMHr06IDmpCLs1ucpgmiePn36unXrDjrooOuvv57moFu3bmLbDz/84DJjPKg/80jY8dCiRYs2u6927dpRJGon8wpGDtD9+vU788wzIXbs2EFVoHJoFSdamNfOmTOHYcySJUu+++47l6PCAwcO7NChg/hrxYoV1Oa1a9dyl2kx82P4r7/+urSyrhJJxe/WrVtJuHr1aqogxBdffAEzSXOgCKrQzWvjxo1o/uabbx566KFjjz1WOj2Zrnz//fdeZowH9Wce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)
# 
# These will be considered in the following sections.
# 
# In general, the Brandt model takes the approach of calculating the total embodied emissions associated with the field, then linking this to the daily oil production rate (which is typically assumed to be constant over field life). This effectively spreads the lifecycle emissions evenly over the life of the field. This is not representative of the environmental impacts of these emissions, most of which will be released during offsite manufacturing of equipment (i.e. prior to the start of field life), with a portion associated with field abandonment being released close to the end of field life. The Brandt approach should be reviewed against a standard method for accounting for embodied emissions over time, if such a standard exists. This method means that embodied emissions are independent of flow rate and pressure (which decrease over field life), but equipment must be sized for the most demanding design case (i.e. typicall early field life with high pressure and flow).

# ## 4.1 Emissions from steel
# 
# Steel is used in several parts of the assumed development:
# * Wellbore Construction (both for production and injection wells)
# * Production & Surface Processing Facilities
# * Ancilliary Structures (i.e. Tanks)
# * Export Pipelines
# * Gathering System Piping

# ### 4.1.1 Wellbore Construction Steel Emissions
# 
# OPGEE model parameterises well construction. In the absense of more specific design details, it defaults to 'moderate' assumptions, as shown below:
# 
# ![image.png](attachment:image.png)
# 
# High-level casing design of the moderate well is as per image below. OPGEE quotes the source of this design as: Nguyen, J.P. Drilling (1996) Oil and Gas Field Development Techniques, p. 37 for Parentis oil field, modified to recognize OPGEE default.
# 
# ![image-2.png](attachment:image-2.png)
# 
# The OPGEE model then uses assumed masses of steel per unit length of casing and production tubing to ultimately calculate a total mass of steel per well.
# 
# For the "Shallow" sensitivity case (1500ft depth), assume top of well design is the same but that bottom of well is only at 1500 ft.
# 
# For the "Deep" sensitivity case (12,000ft depth), use a different well design from the same reference text book cited in the OPGEE model.
# 
# ![image-3.png](attachment:image-3.png) ![image-4.png](attachment:image-4.png)

# In[ ]:


#First calculate the steel requirements for the production wells (50x production wells is default assumption):

# Create dataframes for casing and tubing mass data taken from the OPGEE model, which indicates "Diameters from Mitchell and Miska (2011), Figure 7.18" and "Data from Mitchell, ed. (2006) Petroleum Engineering Handbook, Volume II: Drilling Engineering, Table 7.10"ArithmeticError"
casing_mass_lookup_data = {
    'Outside Diameter (in)': [
        4.500, 5.000, 5.500, 6.625, 7.000, 7.625, 8.625, 9.625, 
        10.750, 11.750, 13.375, 16.000, 18.625, 20.000, 30.000
    ],
    'Steel Casing Mass per Unit Length (lb/ft)': [
        11.28, 14.38, 17.08, 13.25, 27.50, 28.80, 36.14, 40.23,
        50.12, 48.20, 60.70, 77.60, 87.50, 94.00, 168.06
    ]
}
casing_mass_lookup = pd.DataFrame(casing_mass_lookup_data)
casing_mass_lookup.set_index('Outside Diameter (in)', inplace=True)

tubing_mass_lookup_data = {
    'Nominal Diameter (in)': [0.750, 1.000, 1.250, 1.500, 2.375, 2.875, 3.500, 4.000, 4.500],
    'Tubing Mass per Unit Length (lb/ft)': [1.14, 1.70, 2.30, 2.75, 4.90, 7.50, 10.20, 9.50, 12.60]
}
tubing_mass_lookup = pd.DataFrame(tubing_mass_lookup_data)
tubing_mass_lookup.set_index('Nominal Diameter (in)', inplace=True)

# Function to calculate mass of casing or tubing
def calculate_casing_or_tubing_mass(diameter, length, lookup_df):
    try:
        # Access the mass per unit length directly using the index and extracting the scalar value
        mass_per_unit_length = lookup_df.loc[diameter, lookup_df.columns[0]]
        # Calculate the total mass
        total_mass = mass_per_unit_length * length
        return total_mass
    except KeyError:
        raise ValueError("Error. Diameter is not included in the lookup table")

# Function to calculate steel mass for given case and sensitivity variables. The design of a well varies stepwise with depth, so these cases can be considered as sensitivity cases but cannot easily be included in a Monte Carlo uncertainty simulation.
def calculate_well_steel_mass_MC(case, sensitivity_variables=None):
    if case == 'Shallow': #This well design is a modified version of the baseline design detailed in the OPGEE model and taken from "J.P. Drilling (1996) Oil and Gas Field Development Techniques, p. 37 for Parentis oil field, modified to recognize OPGEE default."
        conductor_diameter = 20  # in
        conductor_length = 50  # ft
        surface_casing_diameter = 11.75  # in
        surface_casing_length = 1000  # ft
        production_casing_diameter = 7  # in
        production_casing_length = 1500  # ft
        production_tubing_diameter = 2.375  # in
        production_tubing_length = 1500  # ft
    elif case == 'Deep':
        conductor_diameter = 30  # in. This well design is a modified version of the Meillon-Pont d'As (gas) well design, which is also shown in "J.P. Drilling (1996) Oil and Gas Field Development Techniques, p. 37"
        conductor_length = 50  # ft
        surface_casing_diameter = 20  # in
        surface_casing_length = 5000  # ft
        intermediate_casing_diameter = 13.375  # in
        intermediate_casing_length = 3300  # ft
        production_casing_diameter = 7  # in
        production_casing_length = 12000  # ft
        production_tubing_diameter = 2.375  # in
        production_tubing_length = 12000  # ft
    else: #This is the default case. i.e. For "Baseline" and all other cases. It is taken directly from the OPGEE model, which states "Moderate well design is is from Nguyen, J.P. Drilling (1996) Oil and Gas Field Development Techniques, p. 37 for Parentis oil field, modified to recognize OPGEE default."
        conductor_diameter = 20 #in 
        conductor_length = 50 #ft
        surface_casing_diameter = 11.75 #in
        surface_casing_length = 1000 #ft
        production_casing_diameter = 7 #in
        production_casing_length = 6000 #ft
        production_tubing_diameter = 2.375 #in
        production_tubing_length = 6000 #ft

    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default

    # Now calculate the numbert of Injection Wells inferred by the number of production wells:
    number_injection_wells = math.ceil(0.25*number_production_wells) #Assumption is the number of injection wells is 25% of the number of production wells. Rounding up, as you can't drill a fraction of a well.
    # Now calculate the total number of wells in this sensitivity scenario:
    total_number_wells = number_production_wells + number_injection_wells 

    # Calculate masses based on the given dimensions
    conductor_mass = calculate_casing_or_tubing_mass(conductor_diameter, conductor_length, casing_mass_lookup)
    surface_casing_mass = calculate_casing_or_tubing_mass(surface_casing_diameter, surface_casing_length, casing_mass_lookup)
    production_casing_mass = calculate_casing_or_tubing_mass(production_casing_diameter, production_casing_length, casing_mass_lookup)
    production_tubing_mass = calculate_casing_or_tubing_mass(production_tubing_diameter, production_tubing_length, tubing_mass_lookup)

    if case == 'Deep':
        intermediate_casing_mass = calculate_casing_or_tubing_mass(intermediate_casing_diameter, intermediate_casing_length, casing_mass_lookup)
    else:
        intermediate_casing_mass = 0 # Because the Deep case is the only case with intermediate casing.

    # Sum up total mass for each production well and then calculate total mass for all production wells
    production_well_steel_mass_individual = (
        conductor_mass + surface_casing_mass + intermediate_casing_mass + production_casing_mass + production_tubing_mass
    )
    production_well_steel_mass_total = production_well_steel_mass_individual * number_production_wells

    # Assuming injection wells are the same as production wells, calculate the total steel mass for all wells
    total_steel_mass_all_wells = production_well_steel_mass_individual * total_number_wells  # This is the value if all wells are identical, regardless of production or injection.
    injection_well_steel_mass_brandt = production_well_steel_mass_total * 0.25 / 1.25  # This is the manner Brandt calculates injection well steel mass. The assumed number of injection wells differs here to elsewhere in the model (i.e. here is 20% but elsewhere is 25%)
    total_steel_mass_all_wells_brandt = production_well_steel_mass_total + injection_well_steel_mass_brandt

    return {
        'case': case,
        'production_well_steel_mass_total': production_well_steel_mass_total,
        'total_steel_mass_all_wells': total_steel_mass_all_wells,
        'total_steel_mass_all_wells_brandt': total_steel_mass_all_wells_brandt
    }

# # Example usage:
# sensitivity_variables = {
#     'number_production_wells': 50,
#     # 'total_number_wells': 100
# }
# result_shallow = calculate_well_steel_mass_MC('Shallow', sensitivity_variables)
# result_deep = calculate_well_steel_mass_MC('Deep', sensitivity_variables)

# print(result_shallow)
# print(result_deep)
# a = result_deep['production_well_steel_mass_total']
# b = result_shallow['production_well_steel_mass_total']
# print(a/b)


# ### 4.1.2 Production & Surface Processing Facilities Steel Emissions
# 
# This section estimates the quantity of steel, and its associated emissions, required for the following surface processing equipment items/sub-systems:
# * Surface tubing
# * Separators
# * Gas sweetening equipment
# * Gas dehydration equipment
# * Gas compression equipment
# * Gas separation via Pressure Swing Adsorption (PSA)

# #### 4.1.2.1 Surface Tubing
# 
# Here, Brandt takes the OPGEE default values for the surface tubing, which are stated below. It assumes std weight tubing, based on a lookup table from  "Oilfield data handbook", Apex Distribution Inc.
# Brandt assumes that tubing is only required for the production wells, not the injection wells. This is a curious assumption, as injection wells would also require tubing. Calculations for both just production wells and all wells are included below.

# In[ ]:


# Calculate the steel required for surface tubing, as a function of case and sensitivity variables

def calculate_surface_tubing_steel_mass(case, sensitivity_variables=None):
    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default

    # Calculate the total number of wells in this sensitivity scenario
    number_injection_wells = math.ceil(0.25 * number_production_wells)  # Assumption is the number of injection wells is 25% of the number of production wells. Rounding up, as you can't drill a fraction of a well.
    total_number_wells = number_production_wells + number_injection_wells

    # Define the surface tubing dimensions and properties
    surface_tubing_avg_diameter = 2.00  # in
    surface_tubing_strength_class = 'std'
    surface_tubing_weight_per_ft = 3.653  # lb/ft
    surface_tubing_avg_length_per_well = 250  # ft/well

    # Calculate the total mass of surface tubing for all wells
    surface_tubing_total_mass_all_wells = surface_tubing_weight_per_ft * surface_tubing_avg_length_per_well * total_number_wells  # lb
    surface_tubing_total_mass_brandt = surface_tubing_weight_per_ft * surface_tubing_avg_length_per_well * number_production_wells  # lb

    return {
        'case': case,
        'surface_tubing_total_mass_all_wells': surface_tubing_total_mass_all_wells,
        'surface_tubing_total_mass_brandt': surface_tubing_total_mass_brandt
    }

# Example usage:
sensitivity_variables = {
    'number_production_wells': 50,
    # 'total_number_wells': 100
}
result_surface_tubing = calculate_surface_tubing_steel_mass('Baseline', sensitivity_variables)
print(result_surface_tubing)


# surface_tubing_avg_diameter = 2.00 #in
# surface_tubing_strength_class = 'std'
# surface_tubing_weight_per_ft = 3.653 #lb/ft
# surface_tubing_avg_length_per_well = 250 #ft/well
# surface_tubing_total_mass_all_wells = surface_tubing_weight_per_ft * surface_tubing_avg_length_per_well * total_number_wells #lb
# surface_tubing_total_mass_brandt = surface_tubing_weight_per_ft * surface_tubing_avg_length_per_well * number_production_wells #lb

# print("Mass of surface tubing assuming all wells:", surface_tubing_total_mass_all_wells, "lb")
# print("Mass of surface tubing per Brandt:", surface_tubing_total_mass_brandt, "lb")



# #### 4.1.2.2 Separator(s)
# 
# The OPGEE model includes a simple look-up table for separator sizing and throughput based on inlet pressure (From http://www.surfaceequip.com/two-three-phase-vertical-horizontal-separators-gas-scrubbers.html, accessed April 21, 2014) and assumes that it is cheapest to buy a smaller number of larger separators.
# 
# ![image.png](attachment:image.png)
# 
# Brandt assumes the smallest separator is installed, despite the pressure at start-of-field-life exceeding the stated working pressure of the 'default' separator. In reality, with such a small volume of produced liquid (relative to gas), it may be that a horizontal separator is not effective in achieving two-phase (i.e. liquid vs gas) separation, let alone three-phase separation (i.e. gas, water & oil).
# 
# For the time being, I will retain Brandt's assumption of a single separator of the smallest size provided in the OPGEE default table (noting the provided reference is now defunct).

# In[ ]:


separator_nominal_weight = 1692 #lb. As noted above, this is taken directly from the OPGEE model.
separator_multiplier_additional_material = 1.5
separator_total_weight = separator_nominal_weight * separator_multiplier_additional_material #lb

print(separator_total_weight)


# #### 4.1.2.3 Gas sweetening equipment
# 
# Gas sweetening (aka acid gas removal, AGR) equipment is that which strips 'sour' CO2 and H2S from the product gas stream. 
# 
# Design operates assuming that pressure entering the sub-system is controlled to operating pressure of 500psi. 
# 
# Brandt's analysis adopts the default design assumptions for an AGR unit, as shown below:
# 
# ![image.png](attachment:image.png)

# In[ ]:


#First define a function to calculate absorber inner diameter based on the logic in OPGEE, which cites data from Manning and Thompson (1991), derived from Khan and Manning (1985).

def calculate_absorber_inner_diameter(pressure, gas_flow_rate):
    if pressure <= 200:
        return 12.259 * gas_flow_rate ** 0.4932
    elif 200 < pressure <= 500:
        return 12.259 * gas_flow_rate ** 0.4932
    elif 500 < pressure <= 1000:
        return 8.6115 * gas_flow_rate ** 0.479
    elif 1000 < pressure <= 1500:
        return 8.6115 * gas_flow_rate ** 0.479
    else:
        raise ValueError("Pressure cannot be greater than 1500 psig")
    
#Now call the function to return the absorber inner diameter for this case:

absorber_operating_pressure = 500 #psig. The OPGEE model assumes pressure is controlled to this point.
absorber_height = 30 #ft. Assumed height of absorber. This is a default assumption in the OPGEE model.
absorber_shell_thickness = 0.5 #in. Assumed thickness of absorber shell. This is a default assumption in the OPGEE model.
absorber_shell_steel_density = 20.4 #lb/ft^2 for 0.5 in thick steel. Assumed density of steel for absorber shell. This is a default assumption in the OPGEE model.

#It is not clear if Brandt assumes maximum or minimum gas flow rate for this calculation, so both are calculated here.
max_gas_flow_rate = max(production_profile_df['Baseline Raw Gas Rate, MSCFD']) / 1000 #MMSCFD
min_gas_flow_rate = min(production_profile_df['Baseline Raw Gas Rate, MSCFD']) / 1000 #MMSCFD
print("Maximum Gas Flow Rate:", max_gas_flow_rate)
print("Minimum Gas Flow Rate:", min_gas_flow_rate)

absorber_inner_diameter_max_gas_flow = calculate_absorber_inner_diameter(absorber_operating_pressure, max_gas_flow_rate)
absorber_inner_diameter_min_gas_flow = calculate_absorber_inner_diameter(absorber_operating_pressure, min_gas_flow_rate)

print("Absorber Inner Diameter based on max gas flow rate:", absorber_inner_diameter_max_gas_flow, "in")
print("Absorber Inner Diameter based on min gas flow rate:", absorber_inner_diameter_min_gas_flow, "in")

absorber_shell_area = math.pi * absorber_inner_diameter_min_gas_flow/12 * absorber_height + 2*math.pi * (absorber_inner_diameter_min_gas_flow/12/2)**2#ft^2. This is the surface area of the absorber shell, assuming a cylindrical shape with flat ends. Would be more appropriate/realistic to assume elliptical or hemispherical ends.
absorber_shell_mass = absorber_shell_area * absorber_shell_steel_density #lb. This is the mass of steel required to construct the absorber shell.
absorber_aux_mass = absorber_shell_mass #lb. OPGEE assumes that mass of trays, aux piping etc. is equal to the mass of the absorber shell.
absorber_mass = absorber_shell_mass + absorber_aux_mass #lb. This is the total mass of the absorber unit.
desorber_mass = absorber_shell_mass + absorber_aux_mass #lb. This is the total mass of the desorber unit, assumed to be identical to the absorber unit.
ancilliary_materials_factor_absorber = 2 #Assumed factor for ancilliary materials. Default assumption in OPGEE model. i.e. Additional steel mass is twice the mass of that associated with absorber/desorbers.
gas_sweetening_equip_total_mass = (absorber_mass + desorber_mass) * ancilliary_materials_factor_absorber #lb. This is the total mass of the gas sweetening equipment.
print(gas_sweetening_equip_total_mass)


# #### 4.1.2.4 Gas dehydration equipment
# 
# OPGEE model assumes that glycol (TEG) dehydration is used to remove water from the product gas stream. OPGEE defaults are used, as per image below:
# 
# ![image.png](attachment:image.png)

# In[ ]:


contactor_operating_pressure = absorber_operating_pressure # 500 psig. The OPGEE model assumes pressure is controlled to this point.

#Define a the lookup table used to size the contactor. Notes as to the source of this data are in the OPGEE model.

data = {
    'Pressure (psig)': [400]*13 + [600]*13 + [800]*13 + [1000]*13 + [1200]*13,
    'Throughput (mmscf/d)': [0.0, 3.1, 4.6, 6.8, 14.3, 16.8, 23.4, 25.5, 34.4, 43.6, 53.0, 61.1, 77.4,
                             0.0, 3.7, 5.8, 8.2, 17.5, 20.6, 28.7, 31.2, 42.1, 53.5, 65.3, 79.5, 93.8,
                             0.0, 4.1, 6.6, 9.5, 20.6, 24.0, 33.5, 36.4, 49.2, 62.2, 76.0, 93.0, 110.1,
                             0.0, 4.6, 7.2, 10.4, 23.2, 26.9, 37.7, 41.0, 55.4, 70.1, 86.1, 98.6, 124.9,
                             0.0, 5.0, 7.9, 11.3, 25.5, 29.9, 42.5, 45.8, 61.0, 77.7, 96.0, 108.4, 137.6],
    'OD (in.)': [16.0, 20.0, 24.0, 30.0, 36.0, 42.0, 42.8, 48.8, 54.9, 61.0, 67.1, 73.2, 'NA',
                 16.0, 20.0, 24.0, 30.0, 36.0, 42.0, 43.0, 49.2, 55.3, 61.4, 67.5, 73.6, 'NA',
                 16.0, 20.0, 24.0, 30.0, 36.0, 42.0, 43.3, 49.5, 55.6, 61.8, 67.9, 74.1, 'NA',
                 16.0, 20.0, 24.0, 30.0, 36.0, 42.0, 43.6, 49.8, 56.0, 62.2, 68.4, 74.6, 'NA',
                 16.0, 20.0, 24.0, 30.0, 36.0, 42.0, 43.8, 50.1, 56.3, 62.6, 68.8, 75.0, 'NA'],
    'ID (in.)': [15.6, 19.5, 23.5, 29.4, 35.3, 41.2, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 'NA',
                 15.5, 19.4, 23.3, 29.2, 35.1, 41.0, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 'NA',
                 15.4, 19.3, 23.2, 29.0, 34.8, 40.7, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 'NA',
                 15.3, 19.1, 23.0, 28.8, 34.6, 40.4, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 'NA',
                 15.2, 19.0, 22.9, 28.6, 34.4, 40.2, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 'NA'],
    'Height (ft.)': [15, 15, 15, 20, 20, 20, 25, 25, 25, 30, 30, 30, 'NA',
                     15, 15, 15, 20, 20, 20, 25, 25, 25, 30, 30, 30, 'NA',
                     15, 15, 15, 20, 20, 20, 25, 25, 25, 30, 30, 30, 'NA',
                     15, 15, 15, 20, 20, 20, 25, 25, 25, 30, 30, 30, 'NA',
                     15, 15, 15, 20, 20, 20, 25, 25, 25, 30, 30, 30, 'NA'],
    'Estimated thickness': [0.4, 0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 0.9, 1.0, 1.1, 'NA',
                            0.5, 0.5, 0.6, 0.7, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 'NA',
                            0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.3, 1.4, 1.6, 1.7, 1.9, 2.1, 'NA',
                            0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 'NA',
                            0.8, 0.9, 1.1, 1.3, 1.6, 1.8, 1.8, 2.0, 2.3, 2.5, 2.7, 3.0, 'NA'],
    'Rounded thickness': [0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 'NA',
                          0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 'NA',
                          0.6, 0.7, 0.8, 1.0, 1.2, 1.3, 1.3, 1.5, 1.6, 1.8, 1.9, 2.1, 'NA',
                          0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 'NA',
                          0.8, 1.0, 1.1, 1.4, 1.6, 1.8, 1.8, 2.1, 2.3, 2.6, 2.8, 3.0, 'NA']
}

# Create the DataFrame
contactor_sizing_lookup = pd.DataFrame(data)



# In[ ]:


def find_contactor_sizing(gas_flow_rate, operating_pressure):
    """
    Retrieve the contactor sizing information based on gas flow rate and the closest available pressure that is
    greater than or equal to the specified operating pressure.

    Parameters:
    gas_flow_rate (float): The gas flow rate in mmscf/d.
    operating_pressure (int): The operating pressure in psig.

    Returns:
    tuple: Returns a tuple containing the ID, OD, height, and rounded thickness, or a message if no match is found.
    """
    # Filter DataFrame to find the minimum pressure that is greater than or equal to the specified operating pressure
    available_pressures = contactor_sizing_lookup['Pressure (psig)'].unique()
    # Find pressures greater than or equal to the operating_pressure and take the minimum of those
    possible_pressures = available_pressures[available_pressures >= operating_pressure]
    if possible_pressures.size == 0:
        return "No available pressure equal to or greater than the specified pressure."

    closest_pressure = possible_pressures.min()

    # Filter the DataFrame for the closest pressure found
    pressure_match = contactor_sizing_lookup[contactor_sizing_lookup['Pressure (psig)'] == closest_pressure]

    # Find the row with the closest throughput that is greater than or equal to the given gas flow rate
    throughput_match = pressure_match[pressure_match['Throughput (mmscf/d)'] >= gas_flow_rate].sort_values(by='Throughput (mmscf/d)')
    if throughput_match.empty:
        return "No throughput available that matches or exceeds the specified flow rate."

    # Take the closest match based on throughput
    closest_match = throughput_match.iloc[0]

    # Extract the relevant columns
    id_diameter = closest_match['ID (in.)']
    od_diameter = closest_match['OD (in.)']
    height = closest_match['Height (ft.)']
    rounded_thickness = closest_match['Rounded thickness']

    return (id_diameter, od_diameter, height, rounded_thickness)

# Example usage:
contactor_sizing_results = find_contactor_sizing(min_gas_flow_rate, contactor_operating_pressure)
print("ID, OD, Height, Rounded Thickness:", contactor_sizing_results)

contactor_ID = contactor_sizing_results[0]
contactor_OD = contactor_sizing_results[1]
contactor_height = contactor_sizing_results[2]
contactor_thickness = contactor_sizing_results[3]

contactor_shell_volume = math.pi * ((contactor_OD/2)**2 - (contactor_ID/2)**2) * contactor_height * 12 + 2 * math.pi * (contactor_OD/2)**2 * contactor_thickness #in^3. This is the volume of the contactor shell, assuming a cylindrical shape with flat ends. 
# Would be more appropriate/realistic to assume elliptical or hemispherical ends. Note the formula in the OPGEE model for this is incorrect, as it does not include pi in the area of cylinder calculation, nor does it account for any end coverings.
print("Shell Volume:", contactor_shell_volume)
contactor_shell_mass = contactor_shell_volume * steel_density #lb. This is the mass of steel required to construct the contactor shell.

contactor_aux_mass = contactor_shell_mass #lb. OPGEE assumes that mass of trays, aux piping etc. is equal to the mass of the contactor shell. 
ancilliary_materials_factor_contactor = 2 #Assumed factor for ancilliary materials. Default assumption in OPGEE model. i.e. Additional steel mass is twice the mass of that associated with contactor.

dehydration_equip_total_mass = (contactor_shell_mass + contactor_aux_mass) * ancilliary_materials_factor_contactor #lb. This is the total mass of the dehydration equipment.
print('Total mass of dehydration equipment:', dehydration_equip_total_mass,'lb')


# #### 4.1.2.5 Gas reinjection compressor
# 
# Brandt assumes that all waste gas is re-injected into the subsurface.
# 
# "From relationship for centrifugal compressors from MS thesis of Y. Sun 2015.  Relationship is M = 2887 + 0.7820*Qg, where M is mass of compressor in kg, and Qg is gas flow rate in m3/hr.  Conversion factor of 0.85 is used to convert from mscf/d to m3/hr."

# In[ ]:


gas_injection_volume = 0.666295945746444 #mscf/d. This is the volume of gas injected. OPGEE calculates this based on assumed reservoir/injection pressure, process pressure, and uses a
#model of adiabatic compression. Adding these calculations is a future improvement to this model.
gas_injection_compressor_mass = (2887 + 0.7820/0.85 * gas_injection_volume) / 0.454 #lb. This is the mass of the gas injection compressor, as calculated in the OPGEE model. 
# Conversion factor of 0.85 is used to convert from mscf/d to m3/hr. 0.454 converts from kg to lb.
gas_injection_aux_mass_factor = 2 #Assumed factor for ancilliary materials. Default assumption in OPGEE model. i.e. Additional steel mass is twice the mass of that associated with gas injection compressor.
gas_injection_total_mass = gas_injection_compressor_mass * gas_injection_aux_mass_factor #lb
print('Total mass of gas injection equipment:', gas_injection_total_mass,'lb')


# #### 4.1.2.6 Gas separation by Pressure Swing Adsorption (PSA)
# 
# According to Brandt, PSA is considered standard technology for gas separation in hydrogen production. PSA is not included in the default OPGEE model, so Brandt just assumes a multiple of separator mass.

# In[ ]:


PSA_unit_mass = separator_total_weight * 5 #lb. This is the mass of the PSA unit, as calculated in the OPGEE model.
print('Total mass of PSA unit:', PSA_unit_mass,'lb')


# ### 4.1.3 Ancilliary Structures Steel Emissions
# 
# The OPGEE defaults for "ancilliary structures and construction" include only steel tanks for oil and produced water storage. Produced water storage is excluded in the hydrogen study. The default assumptions call for 3000 bbl of total oil storage capacity, provided by 4off tanks.

# In[ ]:


mass_steel_tanks = 63079.0964136334 #lb. The OPGEE model does not link this calculation to any inputs, so this is taken directly from the model and will not change between scenarios.


# ### 4.1.4 Export Pipelines Steel Emissions
# 
# Brandt's analysis states "Because we do not know the type of pipeline network that crude will be transported over, we compute the steel intensity of crude transport for the entire US pipeline system". It calculates "bbl oil transported per lb of steel" for the total US system then divides the total volume of assumed oil production from the hydrogen field by this ratio to estimate the transport infrastructure attributable to this development.

# In[ ]:


mass_us_pipelines = 74484864000 #lb. This is not linked to any inputs so is taken directly from the OPGEE model.
mass_us_pipelines_ancilliary = 0.5 * mass_us_pipelines #lb. This is the mass of ancilliary equipment associated with the pipeline system, as calculated in the OPGEE model.
total_us_pipeline_system_mass = mass_us_pipelines + mass_us_pipelines_ancilliary #lb. This is the total mass of the US pipeline system, as calculated in the OPGEE model.
total_crude_transported_by_us_pipelines = 109500 #MMbbl/pipe_lifetime. This is the total crude oil transported by the US pipeline system over its lifetime, as calculated in the OPGEE model.
crude_transported_per_steel_mass = total_crude_transported_by_us_pipelines / (total_us_pipeline_system_mass/1E6) #MMbbl/lb. This is the total crude oil transported per lb of steel in the US pipeline system, as calculated in the OPGEE model.

print('Crude transported per lb of steel in US pipeline system:', crude_transported_per_steel_mass, 'MMbbl/lb')

# Define a function that calculates the export pipeline steel mass based on varying oil production assumption rates. Also takes case into account, but assumption of case does not affect assumption of oil production rate.
def calculate_export_pipeline_steel_mass(case, sensitivity_variables=None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
    else:
        oil_production = oil_production_default
        field_lifespan = field_lifespan_default
 
    total_oil_production = oil_production * 365 * field_lifespan  # bbl. This is the total oil production over the lifetime of the field.

    # Calculate export pipeline steel mass
    export_pipeline_steel_mass = total_oil_production / crude_transported_per_steel_mass  # lb

    return {
        'case': case,
        'total_oil_production': total_oil_production,
        'export_pipeline_steel_mass': export_pipeline_steel_mass
    }

# # Example usage
# sensitivity_variables = {
#     'oil_production': 0.1,  # bbl/day
# }

# result = calculate_export_pipeline_steel_mass('Baseline', sensitivity_variables)
# print(result)


# ### 4.1.5 Gathering System Piping Steel Emissions
# 
# The OPGEE model calculates the mass of steel required for the average US natural gas well gathering line piping and multiplies this by the number of wells under consideration.

# In[ ]:


# Define a function that calculates the total mass of steel required for the gathering system, based on the number of wells and the mass of steel required per well, by case and sensitivity variables.

def calculate_gathering_system_steel_mass(case, sensitivity_variables=None):
    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default

    # Calculate the total number of wells in this sensitivity scenario
    number_injection_wells = math.ceil(0.25 * number_production_wells)  # Assumption is the number of injection wells is 25% of the number of production wells. Rounding up, as you can't drill a fraction of a well.
    total_number_wells = number_production_wells + number_injection_wells

    gathering_system_steel_per_well = 13779.116 #lb/well. This figure is calculated in OPGEE but is not dependent on any inputs, so is taken directly from the model.
    total_gathering_system_steel_mass = gathering_system_steel_per_well * total_number_wells #lb. This is the total mass of steel required for the gathering system, as calculated in the OPGEE model.

    return {
        'case': case,
        'total_gathering_system_steel_mass': total_gathering_system_steel_mass #This differs from Brandt's calculation because the OPGEE model does not include gas re-injection wells in the calculation.
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 100,
# }

# result = calculate_gathering_system_steel_mass('Baseline', sensitivity_variables)
# print(result)
# print('Total mass of steel required for gathering system:', total_gathering_system_steel_mass, 'lb') 


# ### 4.1.6 Total Emissions from Steel
# 
# Summing the mass of steel and multiplying by assumed carbon intensity

# In[ ]:


def calculate_total_steel_mass(case, sensitivity_variables):
    
    # Calculate total steel mass for wells
    # well_steel_mass = calculate_well_steel_mass_MC(case, sensitivity_variables)
    export_pipeline_steel_mass = calculate_export_pipeline_steel_mass(case, sensitivity_variables)['export_pipeline_steel_mass']
    total_steel_mass_all_wells = calculate_well_steel_mass_MC(case,sensitivity_variables)['total_steel_mass_all_wells']
    surface_tubing_total_mass_all_wells = calculate_surface_tubing_steel_mass(case, sensitivity_variables)['surface_tubing_total_mass_all_wells']
    total_gathering_system_steel_mass = calculate_gathering_system_steel_mass(case, sensitivity_variables)['total_gathering_system_steel_mass']

    # Calculate total steel mass
    total_steel_mass = (total_steel_mass_all_wells +
                        surface_tubing_total_mass_all_wells +
                        separator_total_weight +
                        gas_sweetening_equip_total_mass +
                        dehydration_equip_total_mass +
                        gas_injection_total_mass +
                        PSA_unit_mass +
                        mass_steel_tanks +
                        export_pipeline_steel_mass +
                        total_gathering_system_steel_mass)
    return {
        'case': case,
        'total_steel_mass': total_steel_mass,
    }

def calculate_total_steel_emissions_MC(case, sensitivity_variables):
   
    total_emissions_steel = calculate_total_steel_mass(case, sensitivity_variables)['total_steel_mass'] * steel_emissions_intensity  # gCO2
    brandt_emissions = 1.96291E+10  # Reference emissions from Brandt
    percent_difference = (total_emissions_steel - brandt_emissions) / brandt_emissions * 100

    return {
        'case': case,
        'total_emissions_steel': total_emissions_steel,
        'percent_difference_from_brandt': percent_difference
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50,
#     'oil_production': 0.10,  # bbl/day  

# }

# mass_baseline = calculate_total_steel_mass('Baseline', sensitivity_variables)
# mass_shallow = calculate_total_steel_mass('Shallow', sensitivity_variables)
# mass_deep = calculate_total_steel_mass('Deep', sensitivity_variables)

# emissions_baseline = calculate_total_steel_emissions_MC('Baseline', sensitivity_variables)
# emissions_shallow = calculate_total_steel_emissions_MC('Shallow', sensitivity_variables)
# emissions_deep = calculate_total_steel_emissions_MC('Deep', sensitivity_variables)

# print(mass_baseline)
# print(mass_shallow)
# print(mass_deep)

# print(emissions_baseline)
# print(emissions_shallow)
# print(emissions_deep)


# ## 4.2 Emissions from cement
# 
# Cement is used in several parts of the assumed development/facilities, including:
# 
# * Production & Injection Wells
# * Wellbore Plugs (at Field Abandonment)

# ### 4.2.1 Production & Injection Well Cement Emissions
# 
# Section 4.1.1 outlines the design of the "moderate" complexity well assumed by Brandt. This section calculates the volume of cement required for these well, such that associated emissions can be inferred.

# In[ ]:


#As with the calculation of the amount of steel associated with wells, the amount of cement is also dependent on the well design, so only 3 scenarios are considered as sensitivity cases.

def calculate_cement_volume_mass(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables:
        sensitivity_variables = {}
    else:
        sensitivity_variables = {}

    # Define parameters for each case
    parameters = {
        'Baseline': {
            'conductor_hole_diameter': 26,
            'surface_casing_hole_diameter': 14.75,
            'production_casing_hole_diameter': 8.75,
            'production_tubing_hole_diameter': 6.0,
            'conductor_diameter': 20,
            'conductor_length': 50,
            'surface_casing_diameter': 11.75,
            'surface_casing_length': 1000,
            'production_casing_diameter': 7,
            'production_casing_length': 6000,
            'production_tubing_diameter': 2.375,
            'production_tubing_length': 6000
        },
        'Shallow': {
            'conductor_hole_diameter': 26,
            'surface_casing_hole_diameter': 14.75,
            'production_casing_hole_diameter': 8.75,
            'production_tubing_hole_diameter': 6.0,
            'conductor_diameter': 20,
            'conductor_length': 50,
            'surface_casing_diameter': 11.75,
            'surface_casing_length': 1000,
            'production_casing_diameter': 7,
            'production_casing_length': 1500,
            'production_tubing_diameter': 2.375,
            'production_tubing_length': 1500
        },
        'Deep': {
            'conductor_hole_diameter': 36,
            'surface_casing_hole_diameter': 26,
            'intermediate_casing_hole_diameter': 17.5,
            'production_casing_hole_diameter': 8.75,
            'production_tubing_hole_diameter': 6.0,
            'conductor_diameter': 30,
            'conductor_length': 50,
            'surface_casing_diameter': 20,
            'surface_casing_length': 5000,
            'intermediate_casing_diameter': 13.375,
            'intermediate_casing_length': 3300,
            'production_casing_diameter': 7,
            'production_casing_length': 12000,
            'production_tubing_diameter': 2.375,
            'production_tubing_length': 12000
        }
    }

    # Select parameters based on the case
    case_parameters = parameters.get(case, parameters['Baseline'])

    # Override parameters with sensitivity variables if provided
    for key in case_parameters.keys():
        if key in sensitivity_variables:
            case_parameters[key] = sensitivity_variables[key]

    bentonite_density = 14.4775342369801 * 7.48  # lb/ft^3. This is the density of bentonite slurry, as calculated in the OPGEE model. Conversion factor of 7.48 is used to convert from lb/gal to lb/ft^3.
    cement_excess_factor = 1.75  # Assumed factor to account for drilling enlargement and cement infiltration. Default assumption in OPGEE model.

    # Calculate void volumes
    conductor_void_volume = math.pi * ((case_parameters['conductor_hole_diameter'] / 2 / 12) ** 2 - (case_parameters['conductor_diameter'] / 2 / 12) ** 2) * case_parameters['conductor_length'] * cement_excess_factor  # ft^3
    surface_casing_void_volume = math.pi * ((case_parameters['surface_casing_hole_diameter'] / 2 / 12) ** 2 - (case_parameters['surface_casing_diameter'] / 2 / 12) ** 2) * case_parameters['surface_casing_length'] * cement_excess_factor  # ft^3
    production_casing_void_volume = math.pi * ((case_parameters['production_casing_hole_diameter'] / 2 / 12) ** 2 - (case_parameters['production_casing_diameter'] / 2 / 12) ** 2) * min(case_parameters['production_casing_length'], 600) * cement_excess_factor  # ft^3

    if case == 'Deep':
        intermediate_casing_void_volume = math.pi * ((case_parameters['intermediate_casing_hole_diameter'] / 2 / 12) ** 2 - (case_parameters['intermediate_casing_diameter'] / 2 / 12) ** 2) * min(case_parameters['intermediate_casing_length'], 600) * cement_excess_factor  # ft^3
    else:
        intermediate_casing_void_volume = 0


    #Calculate the total number of wells based on the sensitivity variable
    number_production_wells = sensitivity_variables.get('number_production_wells',50) #Retreive the number of production wells from the sensitivity variables, otherwise default to 50.
    number_injection_wells = math.ceil(number_production_wells * 0.25)
    total_number_wells = number_production_wells + number_injection_wells

    # Calculate total volumes
    cement_volume_per_well = conductor_void_volume + surface_casing_void_volume + intermediate_casing_void_volume + production_casing_void_volume  # ft^3
    total_well_cement_volume = cement_volume_per_well * total_number_wells  # ft^3
    total_well_cement_mass = total_well_cement_volume * bentonite_density  # lb

    return {
        'case': case,
        'conductor_void_volume': conductor_void_volume,
        'surface_casing_void_volume': surface_casing_void_volume,
        'intermediate_casing_void_volume': intermediate_casing_void_volume,
        'production_casing_void_volume': production_casing_void_volume,
        'cement_volume_per_well': cement_volume_per_well,
        'total_well_cement_volume': total_well_cement_volume,
        'total_well_cement_mass': total_well_cement_mass
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50  # Example sensitivity variable
# }

# shallow_cement_volumes = calculate_cement_volume_mass('Shallow', sensitivity_variables)
# baseline_cement_volumes = calculate_cement_volume_mass('Baseline', sensitivity_variables)
# deep_cement_volumes = calculate_cement_volume_mass('Deep', sensitivity_variables)

# print(shallow_cement_volumes)
# print(baseline_cement_volumes)
# print(deep_cement_volumes)




# ### 4.2.2 Wellbore Plug Cement Emissions
# 
# The calculation considers the volume of cement required to safely 'plug' and abandon the wells at the end of field life. 

# In[ ]:


# Define a function that calculates the wellbore plug mass and volume based on varying sensitivity variables:

def calculate_wellbore_plug_mass_and_volume(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default

    # Define bentonite density
    bentonite_density = 14.4775342369801 * 7.48  # lb/ft^3. Conversion factor of 7.48 is used to convert from lb/gal to lb/ft^3.

    # Define parameters
    wellbore_plug_length = 321  # ft. Default length of a wellbore plug in the OPGEE model.
    production_casing_diameter = 7  # in. We assume in this version of the model that all cases (regardless of shallow, baseline or deep) have the same production casing diameter. Brandt does not include the assumptions of different casing diameters in his model.

    # Calculate wellbore plug volume and mass
    wellbore_plug_diameter = production_casing_diameter  # in. Default diameter of a wellbore plug in the OPGEE model.
    wellbore_plug_volume = math.pi * (wellbore_plug_diameter / 2 / 12) ** 2 * wellbore_plug_length  # ft^3
    wellbore_plug_mass = wellbore_plug_volume * bentonite_density  # lb

    # Get number of production wells from sensitivity variables or use default
    number_production_wells = sensitivity_variables.get('number_production_wells', 50)  # Default is 50 production wells
    number_injection_wells = number_production_wells * 0.25 / 1.25  # Calculation based on Brandt's assumption
    total_number_wells = number_production_wells + number_injection_wells

    # Calculate total wellbore plug mass and volume
    total_wellbore_plug_mass = wellbore_plug_mass * total_number_wells  # lb
    total_wellbore_plug_volume = wellbore_plug_volume * total_number_wells  # ft^3

    return {
        'case': case,
        'wellbore_plug_volume': wellbore_plug_volume,
        'wellbore_plug_mass': wellbore_plug_mass,
        'total_wellbore_plug_mass': total_wellbore_plug_mass,
        'total_wellbore_plug_volume': total_wellbore_plug_volume
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50  # Example sensitivity variable
# }

# wellbore_plug_results = calculate_wellbore_plug_mass_and_volume('Deep', sensitivity_variables)
# print(wellbore_plug_results)


# ### 4.2.3 Total Emissions from Cement
# 
# Summing the mass of cement and multiplying by emissions intensity:

# In[ ]:


# Now define a function to calculate total cement emissions based on case and varying sensitivity variables:

def calculate_total_cement_emissions(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default

    # Calculate cement volumes and masses
    cement_volumes = calculate_cement_volume_mass(case, sensitivity_variables)
    wellbore_plug_results = calculate_wellbore_plug_mass_and_volume(case, sensitivity_variables)

    # Calculate total cement emissions
    total_cement_emissions = (cement_volumes['total_well_cement_volume'] + wellbore_plug_results['total_wellbore_plug_volume']) * cement_emissions_intensity  # gCO2

    return {
        'case': case,
        'total_cement_emissions': total_cement_emissions
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50  # Example sensitivity variable
# }

# cement_emissions_baseline = calculate_total_cement_emissions('Baseline', sensitivity_variables)
# cement_emissions_shallow = calculate_total_cement_emissions('Shallow', sensitivity_variables)
# cement_emissions_deep = calculate_total_cement_emissions('Deep', sensitivity_variables)

# print(cement_emissions_baseline)
# print(cement_emissions_shallow)
# print(cement_emissions_deep)

# cement_emissions_difference = (total_cement_emissions - 2304599426.49053) / 2304599426.49053 * 100    
# print(f'Cement emissions: Percent difference from Brandt: {cement_emissions_difference:.2f}%')


# ## 4.3 Emissions from drilling mud
# 
# Calculates the embodied emissions associated with the drilling mud that is required to drill the production and injection wells. Note Brandt assumes hydrogen wells will fit the "medium" classification of all relevant categories.

# In[ ]:


max_volume_mud_required_multiple = 1 #This is a multiple of the full wellbore volume. Note in OPGEE reads "While wellbore will be partially filled with drillstring, we assume that the maximum mud volume required is equal to total wellbore volume due to mud infiltration and mud tank volumes"
mud_density = 14.0203703703704 #lb/gal of drilling fluid. Assumes "medium mud type". OPGEE calculates this figure based on the density of water, bentonite, and barite, and the volume of each in the drilling fluid.
bentonite_mud_density = 2.00925925925926 #lb/gal of bentonite. OPGEE calculates this figure based on the density of bentonite and the volume of bentonite in the drilling fluid.
bentonite_emissions_intensity = 31.471592226739 / 2.204 #gCO2/lb. This is the emissions intensity of bentonite production, as calculated in the OPGEE model. Conversion factor of 2.204 is used to convert from kg to lb.
barite_mud_density = 5.83333333333333 #lb/gal of barite. OPGEE calculates this figure based on the density of barite and the volume of barite in the drilling fluid.
barite_emissions_intensity = 282.458033501317 / 2.204 #gCO2/lb. This is the emissions intensity of barite production, as calculated in the OPGEE model. Conversion factor of 2.204 is used to convert from kg to lb.

cubic_feet_per_gallon = 0.133681 #ft^3/gal. This is the conversion factor from gallons to cubic feet
gallons_per_cubic_foot = 1 / cubic_feet_per_gallon #gal/ft^3. This is the conversion factor from cubic feet to gallons

# Define a function to calculate the total emissions associated with drilling mud use based on case and varying sensitivity variables:
# Drilling mud calculation uses hole diameter, as the drilling mud sits above the drill bit as it drills, thus the mud must fill the full void.
def calculate_drilling_mud_emissions(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default

    #Calculate the number of each type of well, based on the sensitivity variable:
    number_injection_wells = math.ceil(number_production_wells * 0.25)
    total_number_wells = number_production_wells + number_injection_wells

    # Define parameters
    parameters = {
        'Baseline': {
            'conductor_hole_diameter': 26,
            'surface_casing_hole_diameter': 14.75,
            'production_casing_hole_diameter': 8.75,
            'production_casing_length': 6000,
            'conductor_length': 50,
            'surface_casing_length': 1000,
            'production_casing_length': 6000
        },
        'Shallow': {
            'conductor_hole_diameter': 26,
            'surface_casing_hole_diameter': 14.75,
            'production_casing_hole_diameter': 8.75,
            'production_casing_length': 1500,
            'conductor_length': 50,
            'surface_casing_length': 1000,
            'production_casing_length': 1500
        },
        'Deep': {
            'conductor_hole_diameter': 36,
            'surface_casing_hole_diameter': 26,
            'intermediate_casing_hole_diameter': 17.5,
            'production_casing_hole_diameter': 8.75,
            'production_casing_length': 12000,
            'conductor_length': 50,
            'surface_casing_length': 500,
            'intermediate_casing_length': 3300,
            'production_casing_length': 12000
        }
    }

    # Select parameters based on the case
    case_parameters = parameters.get(case, parameters['Baseline'])

    # Override parameters with sensitivity variables if provided
    for key in case_parameters.keys():
        if key in sensitivity_variables:
            case_parameters[key] = sensitivity_variables[key]

    # Calculate wellbore volumes
    if case == 'Deep':
        wellbore_volume = math.pi * ((case_parameters['conductor_hole_diameter'] / 2 / 12) ** 2 * case_parameters['conductor_length'] +
                                     (case_parameters['surface_casing_hole_diameter'] / 2 / 12) ** 2 * case_parameters['surface_casing_length'] +
                                     (case_parameters['intermediate_casing_hole_diameter'] / 2 / 12) ** 2 * case_parameters['intermediate_casing_length'] +
                                     (case_parameters['production_casing_hole_diameter'] / 2 / 12) ** 2 * case_parameters['production_casing_length'])
    else:
        wellbore_volume = math.pi * ((case_parameters['conductor_hole_diameter'] / 2 / 12) ** 2 * case_parameters['conductor_length'] +
                                 (case_parameters['surface_casing_hole_diameter'] / 2 / 12) ** 2 * case_parameters['surface_casing_length'] +
                                 (case_parameters['production_casing_hole_diameter'] / 2 / 12) ** 2 * case_parameters['production_casing_length'])  # ft^3

    # Calculate total wellbore volume
    total_volume_all_wellbores = wellbore_volume * total_number_wells  # ft^3

    # Calculate total drilling mud mass
    total_drilling_mud_mass = gallons_per_cubic_foot * total_volume_all_wellbores * mud_density  # lb

    # Calculate bentonite and barite masses
    bentonite_mass = gallons_per_cubic_foot * total_volume_all_wellbores * bentonite_mud_density  # lb
    barite_mass = gallons_per_cubic_foot * total_volume_all_wellbores * barite_mud_density  # lb

    # Calculate emissions associated with bentonite and barite production
    bentonite_emissions = bentonite_mass * bentonite_emissions_intensity  # gCO2
    barite_emissions = barite_mass * barite_emissions_intensity  # gCO2

    # Calculate total drilling mud emissions
    total_drilling_mud_emissions = bentonite_emissions + barite_emissions  # gCO2

    return {
        'case': case,
        'total_drilling_mud_mass': total_drilling_mud_mass,
        'total_drilling_mud_emissions': total_drilling_mud_emissions
    }    

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50  # Example sensitivity variable
# }

# drilling_mud_emissions_baseline = calculate_drilling_mud_emissions('Baseline', sensitivity_variables)
# drilling_mud_emissions_shallow = calculate_drilling_mud_emissions('Shallow', sensitivity_variables)
# drilling_mud_emissions_deep = calculate_drilling_mud_emissions('Deep', sensitivity_variables)

# print(drilling_mud_emissions_baseline)
# print(drilling_mud_emissions_shallow)
# print(drilling_mud_emissions_deep)


# ## 4.4 Emissions from transporting materials
# 
# This section accounts for the fact that the materials used in developing the field will first need to be transported to the field. It does this on a mass basis, considering assumed transport distances and transport modalities (i.e. truck vs rail)

# In[ ]:


#Assumed shipment distances for each material category:

shipment_distance_steel = 1000 #miles. 
shipment_distance_cement = 100 #miles.
shipment_distance_drilling_mud = 1000 #miles.

shipment_mode_steel = 'rail' 
shipment_mode_cement = 'truck'
shipment_mode_drilling_mud = 'rail'

trucking_energy_intensity = 969 #btu LHV/ton mi 
rail_energy_intensity = 370 #btu LHV/ton mi

trucking_emissions_intensity = 78651.2982557601 #on-site GHG g/mmbtu of fuel burned - LHV
rail_emissions_intensity = 78989.5272089378 #on-site GHG g/mmbtu of fuel burned - LHV

# Now create functions to calculate energy and emissions associated with shipping each material category based on case and varying sensitivity variables:

def calculate_shipment_energy_steel(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default
        
    # Calculate total steel mass
    total_steel_mass = calculate_total_steel_mass(case, sensitivity_variables)['total_steel_mass']  # lb

    # Calculate energy required to ship steel
    shipment_energy_steel = total_steel_mass / 2000 * shipment_distance_steel * (rail_energy_intensity if shipment_mode_steel == 'rail' else trucking_energy_intensity) / 1E6  # mmbtu

    return {
        'case': case,
        'shipment_energy_steel': shipment_energy_steel
    }

def calculate_shipment_energy_cement(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default

    # Calculate total cement mass
    total_well_cement_mass = calculate_cement_volume_mass(case, sensitivity_variables)['total_well_cement_mass']  # lb

    # Calculate energy required to ship cement
    shipment_energy_cement = total_well_cement_mass / 2000 * shipment_distance_cement * (rail_energy_intensity if shipment_mode_cement == 'rail' else trucking_energy_intensity) / 1E6  # mmbtu

    return {
        'case': case,
        'shipment_energy_cement': shipment_energy_cement
    }

def calculate_shipment_energy_drilling_mud(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
    else:
        number_production_wells = number_production_wells_default
        
    # Calculate total drilling mud mass
    total_drilling_mud_mass = calculate_drilling_mud_emissions(case, sensitivity_variables)['total_drilling_mud_mass']  # lb

    # Calculate energy required to ship drilling mud
    shipment_energy_drilling_mud = total_drilling_mud_mass / 2000 * shipment_distance_drilling_mud * (rail_energy_intensity if shipment_mode_drilling_mud == 'rail' else trucking_energy_intensity) / 1E6  # mmbtu

    return {
        'case': case,
        'shipment_energy_drilling_mud': shipment_energy_drilling_mud
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50  # Example sensitivity variable
# }
# test_case = 'Shallow'
# shipment_energy_steel_baseline = calculate_shipment_energy_steel(test_case, sensitivity_variables)
# shipment_energy_cement_baseline = calculate_shipment_energy_cement(test_case, sensitivity_variables)
# shipment_energy_drilling_mud_baseline = calculate_shipment_energy_drilling_mud(test_case, sensitivity_variables)

# print(shipment_energy_steel_baseline)
# print(shipment_energy_cement_baseline)
# print(shipment_energy_drilling_mud_baseline)


#Now calculate the emissions associated with shipping each material category:

def calculate_shipment_emissions_steel(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    # Calculate energy required to ship steel
    shipment_energy_steel = calculate_shipment_energy_steel(case, sensitivity_variables)['shipment_energy_steel']  # mmbtu

    # Calculate emissions associated with shipping steel
    shipment_emissions_steel = shipment_energy_steel * (rail_emissions_intensity if shipment_mode_steel == 'rail' else trucking_emissions_intensity)  # gCO2

    return {
        'case': case,
        'shipment_emissions_steel': shipment_emissions_steel
    }

def calculate_shipment_emissions_cement(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    # Calculate energy required to ship cement
    shipment_energy_cement = calculate_shipment_energy_cement(case, sensitivity_variables)['shipment_energy_cement']  # mmbtu

    # Calculate emissions associated with shipping cement
    shipment_emissions_cement = shipment_energy_cement * (rail_emissions_intensity if shipment_mode_cement == 'rail' else trucking_emissions_intensity)  # gCO2

    return {
        'case': case,
        'shipment_emissions_cement': shipment_emissions_cement
    }

def calculate_shipment_emissions_drilling_mud(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables is None:
        sensitivity_variables = {}

    # Calculate energy required to ship drilling mud
    shipment_energy_drilling_mud = calculate_shipment_energy_drilling_mud(case, sensitivity_variables)['shipment_energy_drilling_mud']  # mmbtu

    # Calculate emissions associated with shipping drilling mud
    shipment_emissions_drilling_mud = shipment_energy_drilling_mud * (rail_emissions_intensity if shipment_mode_drilling_mud == 'rail' else trucking_emissions_intensity)  # gCO2

    return {
        'case': case,
        'shipment_emissions_drilling_mud': shipment_emissions_drilling_mud
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50, # Example sensitivity variable
#     'oil_production': 0.10  # Example sensitivity variable
# }

# shipment_emissions_steel_baseline = calculate_shipment_emissions_steel(test_case, sensitivity_variables)
# shipment_emissions_cement_baseline = calculate_shipment_emissions_cement(test_case, sensitivity_variables)
# shipment_emissions_drilling_mud_baseline = calculate_shipment_emissions_drilling_mud(test_case, sensitivity_variables)

# print(shipment_emissions_steel_baseline)
# print(shipment_emissions_cement_baseline)
# print(shipment_emissions_drilling_mud_baseline)


# ## 4.5 Total Embodied Emissions & Equivalent Daily Rate

# In[ ]:


# Define functions to calculate total emobided emissions and daily rate of embodied emissions based on case and varying sensitivity variables:

def calculate_total_embodied_emissions(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables:
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
    else:
        field_lifespan = field_lifespan_default

    # Calculate total emissions associated with steel
    total_steel_emissions = calculate_total_steel_emissions_MC(case, sensitivity_variables)['total_emissions_steel']  # gCO2

    # Calculate total emissions associated with cement
    total_cement_emissions = calculate_total_cement_emissions(case, sensitivity_variables)['total_cement_emissions']  # gCO2

    # Calculate total emissions associated with drilling mud
    total_drilling_mud_emissions = calculate_drilling_mud_emissions(case, sensitivity_variables)['total_drilling_mud_emissions']  # gCO2

    # Calculate total embodied emissions
    total_embodied_emissions = (total_steel_emissions + total_cement_emissions + total_drilling_mud_emissions) / 1000  # kgCO2

    return {
        'case': case,
        'total_embodied_emissions': total_embodied_emissions
    }

def calculate_embodied_emissions_daily_rate(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables:
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
    else:
        field_lifespan = field_lifespan_default

    # Calculate total embodied emissions
    total_embodied_emissions = calculate_total_embodied_emissions(case, sensitivity_variables)['total_embodied_emissions']  # kgCO2

    # Calculate daily rate of embodied emissions
    embodied_emissions_daily_rate = total_embodied_emissions / (365 * field_lifespan)  # kgCO2/day

    return {
        'case': case,
        'embodied_emissions_daily_rate': embodied_emissions_daily_rate
    }

# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50,  # Example sensitivity variable
#     'oil_production': 0.1  # Example sensitivity variable
# }

# total_embodied_emissions_baseline = calculate_total_embodied_emissions('Baseline', sensitivity_variables)
# embodied_emissions_daily_rate_baseline = calculate_embodied_emissions_daily_rate('Baseline', sensitivity_variables)

# total_embodied_emissions_shallow = calculate_total_embodied_emissions('Shallow', sensitivity_variables)
# embodied_emissions_daily_rate_shallow = calculate_embodied_emissions_daily_rate('Shallow', sensitivity_variables)

# total_embodied_emissions_deep = calculate_total_embodied_emissions('Deep', sensitivity_variables)
# embodied_emissions_daily_rate_deep = calculate_embodied_emissions_daily_rate('Deep', sensitivity_variables)

# print(total_embodied_emissions_baseline)
# print(embodied_emissions_daily_rate_baseline)
# print(total_embodied_emissions_shallow)
# print(embodied_emissions_daily_rate_shallow)
# print(total_embodied_emissions_deep)
# print(embodied_emissions_daily_rate_deep)


# # 5. "Other" Offsite Emissions
# 
# This category includes GHG emissions from:
# 
# * Diesel supply/export
# * Electricity supply/export

# ## 5.1 Emissions from Diesel Supply/Export
# 
# Diesel consumption and associated emissions during exploration and drilling were already calculated in Section 3. This section appears to account for "upstream" emissions associated with the fuel. i.e. "The indirect energy consumption and GHG emissions of imported fuel" (OPGEE Manual). OPGEE and GREET call these the "Fuel Cycle" emissions, also known as the "well-to-tank" emissions (whereas combustion emissions are known as "tank-to-wheel" emissions).

# In[ ]:


diesel_total_fuel_cycle_emission_intensity = 19559.2732502507 #gCO2eq/mmbtu. This is the total fuel cycle emissions intensity of diesel, as calculated in GREET1_2016 and referenced in the OPGEE model.
#This apparently accounts for the emissions associated with producing and transporting the fuel to the site, excluding the emissions from combustion (which were calculated in Section 3).

def calculate_total_diesel_emissions(case,sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default) #Retreive the number of production wells from the sensitivity variables, otherwise use the default value.
        oil_production = sensitivity_variables.get('oil_production', oil_production_default) #Retreive the oil production from the sensitivity variables, otherwise use the default value.
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default) #Retreive the field lifespan from the sensitivity variables, otherwise use the default value.
    else:
        number_production_wells = number_production_wells_default
        oil_production = oil_production_default
        field_lifespan = field_lifespan_default
        

    exploration_daily_energy_use = calculate_exploration_emissions(case,sensitivity_variables)['exploration_daily_energy_use'] #mmbtu. This is the daily energy use associated with exploration activities.
    development_drilling_energy = calculate_development_drilling_emissions(case,sensitivity_variables)['development_drilling_energy'] #mmbtu. This is the energy use associated with development drilling activities.

    total_diesel_energy_consumption = exploration_daily_energy_use + development_drilling_energy #mmbtu. This is the total energy consumption associated with diesel use.
    # total_diesel_emissions_Section3 = exploration_emissions + development_drilling_emissions #This is the total emissions associated with diesel combustion, as calculated above in Section 3..
    total_diesel_emissions = total_diesel_energy_consumption * diesel_total_fuel_cycle_emission_intensity / 1E6 #tCO2e/d. This is the total emissions associated with diesel use, using the "fuel cycle" emissions intensity of diesel discussed above.

    return {
        'case': case,
        'total_diesel_emissions': total_diesel_emissions, #tCO2e/d
    }


# # Example usage
# sensitivity_variables = {
#     'number_production_wells': 50,  # Example sensitivity variable
#     'oil_production': 0.1,  # Example sensitivity variable
#     'field_lifespan': 30
# }

# total_diesel_emissions_baseline = calculate_total_diesel_emissions('Baseline', sensitivity_variables)
# total_diesel_emissions_shallow = calculate_total_diesel_emissions('Shallow', sensitivity_variables)
# total_diesel_emissions_deep = calculate_total_diesel_emissions('Deep', sensitivity_variables)

# print(total_diesel_emissions_baseline)
# print(total_diesel_emissions_shallow)
# print(total_diesel_emissions_deep)


# As shown below, the "fuel cycle" emissions represent ~25% of the combustion emissions. 
# print(f'Total emissions associated with diesel use: {total_diesel_emissions:.2e} gCO2')
# print(total_diesel_emissions_Section5/total_diesel_emissions_Section3*100)
# print('Total diesel energy consumption:', total_diesel_energy_consumption, 'mmbtu')

# #Test Usage:
# calculate_total_diesel_emissions('Baseline')


# In[ ]:


# Function to calculate total diesel emissions
def calculate_total_diesel_emissions(case, sensitivity_variables=None):
    # Set default sensitivity variables if none provided
    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
    else:
        number_production_wells = number_production_wells_default
        oil_production = oil_production_default
        field_lifespan = field_lifespan_default
        GWP_H2 = GWP_H2_default
        
    exploration_daily_energy_use = calculate_exploration_emissions(case, sensitivity_variables)['exploration_daily_energy_use']
    development_drilling_energy = calculate_development_drilling_emissions(case, sensitivity_variables)['development_drilling_energy']

    total_diesel_energy_consumption = exploration_daily_energy_use + development_drilling_energy
    total_diesel_emissions = total_diesel_energy_consumption * diesel_total_fuel_cycle_emission_intensity / 1E6

    return {
        'case': case,
        'total_diesel_emissions': total_diesel_emissions,
    }

## Example usage
# sensitivity_variables = {
#     'number_production_wells': 50,  # Example sensitivity variable
#     'oil_production': 0.1,  # Example sensitivity variable
#     'field_lifespan': 30
# }

# total_diesel_emissions_baseline = calculate_total_diesel_emissions('Baseline', sensitivity_variables)
# total_diesel_emissions_shallow = calculate_total_diesel_emissions('Shallow', sensitivity_variables)
# total_diesel_emissions_deep = calculate_total_diesel_emissions('Deep', sensitivity_variables)

# print(total_diesel_emissions_baseline)
# print(total_diesel_emissions_shallow)
# print(total_diesel_emissions_deep)


# ## 5.2 Emissions from Electricity Supply/Export
# 
# The OPGEE model assigns emissions to estimates of the electricity necessary to operate the processing equipment. In the baseline case, it is assumed that the only equipment consuming/requiring electricity is the Dehydrator and the Produced Water Treatment system. Note that is separately assumed that the compressors in the process (reinjection compressor and separation boosting compressor(s)) are powered by 'fuel gas' (in this case, a portion of the hydrogen product gas stream).

# ### 5.2.1 Dehydration Unit Electricity Consumption
# 
# This calculation is based on a HYSYS process simulation to estimate energy/electricity consumption of a dehydration unit. It is assumed here that this energy consumption does not change between various sensitivity cases.

# In[ ]:


#The components in the dehydration unit that are assumed to require electricity are pump(s) and air-cooling fan(s).
Eta_reboiler_dehydrator = 1.25 #btu consumed per btu delivered. This is the inverse of the assumed energy efficiency of the reboiler in the dehydration unit. Default assumption in OPGEE model.
reboiler_heat_duty = 23.2584787632862 #kW. This is the heat duty of the reboiler in the dehydration unit, as calculated in the OPGEE model using the results of a statstical model from Aspen HYSYS. 
#Replicating the HYSYS model results in this notebook is an opportunity for future improvement. The minimal impact of this calculation on overall emissions results means this is not a high priority.

btu_per_kWh = 3412.14163 #btu/kWh. This is the conversion factor from kWh to btu.

predicted_reboiler_daily_fuel_use= reboiler_heat_duty * 1000 * 3600* 24 / mmbtu_to_MJ / 1000000 * Eta_reboiler_dehydrator #mmbtu LHV/day. This is the 'fuel gas' used in the reboiler. The baseline case uses produced H2 as the fuel gas, so there are no GHG emissions associated with this combustion.

predicted_dehydration_pump_duty = 0.349399190420817 #kW. This is the duty of the pump in the dehydration unit, as calculated in the OPGEE model using the results of a statstical model from Aspen HYSYS.
#Replicating the HYSYS model results in this notebook is an opportunity for future improvement. The minimal impact of this calculation on overall emissions results means this is not a high priority.

predicted_dehydration_pump_electricity_use = 24 * predicted_dehydration_pump_duty #kWh/day. 

predicted_condenser_thermal_load = 4.98931450698191 #kW. This is the predicted thermal load of the condenser in the dehydration unit, as calculated in the OPGEE model using the results of a statstical model from Aspen HYSYS.
#Replicating the HYSYS model results in this notebook is an opportunity for future improvement. The minimal impact of this calculation on overall emissions results means this is not a high priority.

condenser_thermal_load = predicted_condenser_thermal_load * 1000 * 3600 / mmbtu_to_MJ

delta_temperature_aircooler_dehydrator = 40 #F. This is the temperature difference across the air cooler in the dehydration unit. Default assumption in OPGEE model.

blower_air_quantity = condenser_thermal_load/(0.24*delta_temperature_aircooler_dehydrator) 

blower_CFM = blower_air_quantity/(1*60*0.0749) #Cubic feet per minute. From GPSA Handbook, Ch. 10, assuming standard conditions of sea level, 70F ambient temp.

delta_pressure_aircooler_dehydrator = 0.60 #in. H20. This is default assumption from Secondary Inputs page of OPGEE.

cooling_fan_delivered_horsepower = blower_CFM * delta_pressure_aircooler_dehydrator / (6256 * 0.7) #bhp. GPSA Handbook. "Fan laws" and "Fan efficiency" sections.

cooling_fan_motor_horsepower =  cooling_fan_delivered_horsepower /  0.92 #bhp. GPSA Handbook. "Speed reducer efficienty of 0.92" 

cooling_fan_energy_intensity = (2967 * cooling_fan_motor_horsepower**(-0.018))/btu_per_kWh #=IFERROR((Drivers!K123*M143^Drivers!K124)/btu_per_kWh,0)

dehydration_cooling_fan_electricity_use = cooling_fan_motor_horsepower * 24 * cooling_fan_energy_intensity #kWh/day

total_electricity_use_dehydration = predicted_dehydration_pump_electricity_use + dehydration_cooling_fan_electricity_use #kWh/day

# print('Total electricity use for dehydration:', total_electricity_use_dehydration, 'kWh/day')
# print('Total "fuel gas" use for dehydration:', predicted_reboiler_daily_fuel_use, 'mmbtu/day')

# print(predicted_dehydration_pump_electricity_use, dehydration_cooling_fan_electricity_use)


# ### 5.2.2 Produced Water Treatment Electricity Consumption
# 
# OPGEE default assumptions are that the produced water treatment process involves Dissolved Air Flotation (DAF), Rotating Biological Contactors (RBCs), Dual Media Filtration (DMF), and Reverse Osmosis (RO). Each of these stages consumes a small amount of electricity. Again, this is assumed to be identical between all sensitivity cases.

# In[ ]:


# def calculate_water_treatment_energy_consumption(case, sensitivity_variables = None):
#     if sensitivity_variables:
#         number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
#         oil_production = sensitivity_variables.get('oil_production', oil_production_default)
#         field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
#         water_production = sensitivity_variables.get('water_production', water_production_default)
#     else:
#         number_production_wells = number_production_wells_default
#         oil_production = oil_production_default
#         field_lifespan = field_lifespan_default
#         water_production = water_production_default

#         energy_consumption_DAF = 0.03513458 #kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)
#         energy_consumption_RBC = 0.0349756 #kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)
#         energy_consumption_DMF = 0.00429246 #kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)
#         energy_consumption_RO = 0.2019046 #kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)

#         litres_per_barrel = 158.9873 #litres/bbl. Conversion factor from barrels to litres.
#         water_content_oil_emulsion = 14 #wt.% OPGEE default citing Manning, F. and Thompson, R. (1991)
#         post_separation_oil_sg = 0.808306308121055 #unitless. Specific gravity of oil after primary separation. This is calculated based on assumed API gravity etc. Adding these calculations may be a future improvement to this model.


#         water_in_oil_baseline = oil_production_default * post_separation_oil_sg * litres_per_barrel * water_content_oil_emulsion/100/1000 #tonne/day. This is the amount of water that remains in the oil after primary separation.
#         water_in_oil = oil_production * post_separation_oil_sg * litres_per_barrel * water_content_oil_emulsion/100/1000 #tonne/day. This is the amount of water that remains in the oil after primary separation.

#         # Calculate the WOR for the case at hand, by taking the case's flow rate in MMSCFD, muliplying it by the water production rate (in bbl/mmscf of gas) then dividing it by the oil production rate (in bbl/day).
#         case_WOR = production_profile_df[f'{case} Raw Gas Rate, MSCFD'] / 1000 * water_production / oil_production

#         water_after_separator_baseline = oil_production_default * production_profile_df['Baseline WOR'] * litres_per_barrel/1000 - water_in_oil_baseline #tonne/day. This is the total water production over the lifetime of the field.
#         water_after_separator = oil_production * case_WOR * litres_per_barrel/1000 - water_in_oil #tonne/day. This is the total water production over the lifetime of the field.

#         total_water_to_water_treatment = water_after_separator + water_in_oil #tonne/day
#         total_water_to_water_treatment_BPD = total_water_to_water_treatment * 1000 / litres_per_barrel #bbl/day

#         total_water_treatment_energy_consumption = total_water_to_water_treatment_BPD * (energy_consumption_DAF + energy_consumption_RBC + energy_consumption_DMF + energy_consumption_RO) #kWh/day

#         return {
#             'case': case,
#             'case_WOR': case_WOR,
#             'total_water_treatment_energy_consumption': total_water_treatment_energy_consumption
#         }
# #Test Usage:
# print(calculate_water_treatment_energy_consumption('Baseline',sensitivity_assumptions)['total_water_treatment_energy_consumption'])

def calculate_water_treatment_energy_consumption(case, sensitivity_variables=None):
    # Define the constants outside of the if-else block
    energy_consumption_DAF = 0.03513458 # kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)
    energy_consumption_RBC = 0.0349756 # kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)
    energy_consumption_DMF = 0.00429246 # kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)
    energy_consumption_RO = 0.2019046 # kWh/bbl. OPGEE citing Vlasopoulos, N. et al. (2006)

    litres_per_barrel = 158.9873 # litres/bbl. Conversion factor from barrels to litres.
    water_content_oil_emulsion = 14 # wt.% OPGEE default citing Manning, F. and Thompson, R. (1991)
    post_separation_oil_sg = 0.808306308121055 # unitless. Specific gravity of oil after primary separation.

    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
        water_production = sensitivity_variables.get('water_production', water_production_default)
    else:
        number_production_wells = number_production_wells_default
        oil_production = oil_production_default
        field_lifespan = field_lifespan_default
        water_production = water_production_default

    water_in_oil_baseline = oil_production_default * post_separation_oil_sg * litres_per_barrel * water_content_oil_emulsion / 100 / 1000 # tonne/day
    water_in_oil = oil_production * post_separation_oil_sg * litres_per_barrel * water_content_oil_emulsion / 100 / 1000 # tonne/day

    # Calculate the WOR for the case at hand
    case_WOR = production_profile_df[f'{case} WOR']
    # print(oil_production)

    water_after_separator_baseline = oil_production_default * production_profile_df['Baseline WOR'] * litres_per_barrel / 1000 - water_in_oil_baseline # tonne/day
    water_after_separator = oil_production * case_WOR * litres_per_barrel / 1000 - water_in_oil # tonne/day

    total_water_to_water_treatment = water_after_separator + water_in_oil # tonne/day
    total_water_to_water_treatment_BPD = total_water_to_water_treatment * 1000 / litres_per_barrel # bbl/day

    total_water_treatment_energy_consumption = total_water_to_water_treatment_BPD * (energy_consumption_DAF + energy_consumption_RBC + energy_consumption_DMF + energy_consumption_RO) # kWh/day

    return {
        'case': case,
        'case_WOR': case_WOR,
        'total_water_treatment_energy_consumption': total_water_treatment_energy_consumption
    }

# Test Usage:
print(calculate_water_treatment_energy_consumption('Baseline')['total_water_treatment_energy_consumption'])




# ### 5.2.3 Emissions Calculation
# 
# The above estimates of electricity consumption can be converted into assumption of emissions by assuming an emissions intensity of the system that is providing the electricity.

# In[ ]:


def calculate_electricity_emissions(case,sensitivity_variables = None):
    if sensitivity_variables:
        number_production_wells = sensitivity_variables.get('number_production_wells', number_production_wells_default)
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
        water_production = sensitivity_variables.get('water_production', water_production_default)
    else:
        number_production_wells = number_production_wells_default
        oil_production = oil_production_default
        field_lifespan = field_lifespan_default
        water_production = water_production_default

    #First calculate the total electricity consumption:
    total_electricity_consumption = (predicted_dehydration_pump_electricity_use + dehydration_cooling_fan_electricity_use + calculate_water_treatment_energy_consumption(case, sensitivity_variables)['total_water_treatment_energy_consumption'])/1000 #MWh/d

    # print('Total electricity consumption:', total_electricity_consumption, 'MWh/day')

    #First convert the energy consumption to MMBTU/day
    total_electricity_consumption_MMBTU = total_electricity_consumption*3600000000/1055.05/1000000#MMBTU/day
    # print('Total electricity consumption:', total_electricity_consumption_MMBTU, 'MMBTU/day')

    #Now convert to GHG emissions by multiplying by the relevant factor:
    electricity_emission_intensity = 173293.086036584 #gCO2/mmbtu. This is calculated in OPGEE using the "MRO values from GREET 2021" to inform the grid electricity mix, which in turn informs the weighted emissions intensity.

    total_electricity_emissions = total_electricity_consumption_MMBTU * electricity_emission_intensity / 1E6 #tCO2e/day
    # print('Total electricity emissions:', total_electricity_emissions, 'tCO2/day')

    return {
        'case': case,
        'total_electricity_emissions': total_electricity_emissions
    }

#Test Usage:
calculate_electricity_emissions('Baseline')


# ## 5.3 Total Emissions from "Other" Offsite Emissions
# 
# Finally, calculate the aggregate of each of the above sources in this section.

# In[ ]:


def calculate_total_other_offsite_emissions(case, sensitivity_variables=None):
    if sensitivity_variables:
        water_production = sensitivity_variables.get('water_production', water_production_default)
    else:
        water_production = water_production_default

       
    total_diesel_emissions = calculate_total_diesel_emissions(case,sensitivity_variables)['total_diesel_emissions'] #tCO2e/day. This is the total emissions associated with diesel use.
    total_electricity_emissions = calculate_electricity_emissions(case,sensitivity_variables)['total_electricity_emissions'] #tCO2e/day. This is the total emissions associated with electricity use.
    total_other_offsite_emissions = (total_diesel_emissions + total_electricity_emissions) * 1000 #kgCO2e/day
    return {
        'case': case,
        'total_other_offsite_emissions': total_other_offsite_emissions, #kgCO2e/day
        }

#Test Usage:
calculate_total_other_offsite_emissions('Baseline')


# # 6. Small Sources of Emissions
# 
# Brandt/OPGEE account for miscellaneous, "small sources" of emissions (e.g. light vehicles driven around the field location) as 10% of "direct sources". That is, emissions from Combustion, Land Use, Venting, Flaring and Fugitives throughout all stages of development (Exploration, Drilling and Development, Production & Extraction, and Surface Processing), all of which have been calculated above.

# In[ ]:


# Defining a function to calculate total direct emissions in kg/day depending on the case:

def calculate_total_direct_emissions(case, sensitivity_variables=None):
    if sensitivity_variables:
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
    else:
        oil_production = oil_production_default 
    
    total_operational_VFF_emissions = calculate_total_operational_VFF_emissions(case,sensitivity_variables)['total_operational_VFF_emissions']

    operational_combustion_emissions = calculate_operational_combustion_emissions(case,sensitivity_variables)
    
    # calculate_exploration_emissions and calculate_development_drilling_emissions return dictionaries with a key that includes the emissions value, so need to refer to the correct key here:
    exploration_emissions_info = calculate_exploration_emissions(case, sensitivity_variables)
    development_drilling_emissions_info = calculate_development_drilling_emissions(case,sensitivity_variables)
    
    exploration_emissions = exploration_emissions_info['exploration_emissions']
    development_drilling_emissions = development_drilling_emissions_info['development_drilling_emissions']
    
    total_direct_emissions_value = (
        total_operational_VFF_emissions + #kgCO2e/day
        operational_combustion_emissions * 1000 + #kgCO2e/day
        exploration_emissions * 1000 + #kgCO2e/day
        development_drilling_emissions * 1000 #kgCO2e/day
    )
    return {
        'case': case,
        'total_direct_emissions': total_direct_emissions_value, #kgCO2e/day
        'operational_VFF_emissions': total_operational_VFF_emissions,
        'exploration_emissions': exploration_emissions,
        'development_drilling_emissions': development_drilling_emissions
    }


def calculate_small_source_emissions(case, sensitivity_variables=None):
    if sensitivity_variables:
        small_source_emissions_percentage = sensitivity_variables.get('small_source_emissions_percentage', small_source_emissions_percentage_default)
    else:
        small_source_emissions_percentage = small_source_emissions_percentage_default
    
    total_direct_emissions_info = calculate_total_direct_emissions(case,sensitivity_variables)
    total_direct = total_direct_emissions_info['total_direct_emissions'] #kgCO2e/day
    small_source_emissions_value = small_source_emissions_percentage / 100 * total_direct #kgCO2e/day
    return {
        'case': case,
        'small_source_emissions': small_source_emissions_value #kgCO2e/day
    }

# sensitivity_assumptions = {
#     'number_production_wells': 50,  # Example sensitivity variable
#     'oil_production': 0.1,  # Example sensitivity variable
#     'field_lifespan': 30,
#     'GWP_H2': 1,
#     'water_production': 0.1,
#     'small_source_emissions_percentage': np.random.uniform(1,15)
# }

# Example usage:
print(calculate_total_direct_emissions('Baseline'))
# print(calculate_total_direct_emissions('Baseline',sensitivity_assumptions)['total_direct_emissions']/calculate_small_source_emissions('Baseline',sensitivity_assumptions)['small_source_emissions'])


# # 7. Total CO2e Emissions & Emissions Intensity
# 
# Now to sum all of the previously-calculated emissions and enable calculation of emissions intensity of H2 production.

# ## 7.1 Function to calculate emission statistics for each case, per Brandt

# In[ ]:


def calculate_total_emissions(case, sensitivity_variables=None):
    if sensitivity_variables:
        GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        water_production = sensitivity_variables.get('water_production', water_production_default)
        small_source_emissions_percentage = sensitivity_variables.get('small_source_emissions_percentage', small_source_emissions_percentage_default)
        total_producing_wells = sensitivity_variables.get('Total Producing Wells', number_production_wells_default)
        field_lifespan = sensitivity_variables.get('Field Life', field_lifespan_default)
        water_cut = sensitivity_variables.get('Water Cut (bbl/mmscf)', water_cut_default)
        h2_purification_loss_rate = sensitivity_variables.get('H2 purification loss rate', h2_purification_loss_rate_default)
        pressure_decline_rate = sensitivity_variables.get('pressure_decline_rate', pressure_decline_rate_default)
    else:
        GWP_H2 = GWP_H2_default
        oil_production = oil_production_default
        water_production = water_production_default
        small_source_emissions_percentage = small_source_emissions_percentage_default
        total_producing_wells = number_production_wells_default
        field_lifespan = field_lifespan_default
        water_cut = water_cut_default
        h2_purification_loss_rate = h2_purification_loss_rate_default
        pressure_decline_rate = pressure_decline_rate_default
    
    # #Call the sensitivity handling function:
    # sensitivity_variables = sensitivity_variable_handling(sensitivity_variables)

    #Call the functions to calculate the total emissions for each case:
    daily_field_H2_exported = calculate_exploration_emissions(case,sensitivity_variables)['daily_field_H2_exported']
    total_direct_emissions = calculate_total_direct_emissions(case,sensitivity_variables)['total_direct_emissions']
    small_source_emissions = calculate_small_source_emissions(case,sensitivity_variables)['small_source_emissions']
    total_other_offsite_emissions = calculate_total_other_offsite_emissions(case,sensitivity_variables)['total_other_offsite_emissions']
    embodied_emissions_daily_rate = calculate_embodied_emissions_daily_rate(case,sensitivity_variables)['embodied_emissions_daily_rate']

    total_emissions = total_direct_emissions + embodied_emissions_daily_rate + total_other_offsite_emissions + small_source_emissions

    production_weighted_emissions = total_emissions / (daily_field_H2_exported * 1000)

    production_weighted_embodied_emissions = embodied_emissions_daily_rate / (daily_field_H2_exported * 1000)

    total_emissions_absolute = sum(total_emissions * 365) #kg CO2e. This is the total emissions over the lifetime of the field. Assumes emissions are constant over each year of the field's lifetime.

    # Calculate statistics
    min_production_weighted_emissions = min(production_weighted_emissions.tolist())  # Convert Series to list
    mean_production_weighted_emissions = statistics.mean(production_weighted_emissions.tolist())  # Convert Series to list
    median_production_weighted_emissions = statistics.median(production_weighted_emissions.tolist())  # Convert Series to list
    max_production_weighted_emissions = max(production_weighted_emissions.tolist())  # Convert Series to list
    mean_production_weighted_embodied_emissions = statistics.mean(production_weighted_embodied_emissions.tolist())  # Convert Series to list
    percent_embodied_to_total = (mean_production_weighted_embodied_emissions / mean_production_weighted_emissions) * 100

    # print(type(min_production_weighted_emissions))

    return {
        'case': case,
        'Min (Year 1) Emissions kgCO2e/kgH2': min_production_weighted_emissions,
        'Mean Emissions kgCO2e/kgH2': mean_production_weighted_emissions,
        'Median Emissions kgCO2e/kgH2': median_production_weighted_emissions,
        'Max (Year 30) Emissions kgCO2e/kgH2': max_production_weighted_emissions,
        'Mean Embodied Emissions kgCO2e/kgH2': mean_production_weighted_embodied_emissions,
        'Percent Embodied to Total Emissions': percent_embodied_to_total,
        'Total Emissions kgCO2e/day': total_emissions,
        'Total lifetime emissions kgCO2e': total_emissions_absolute,
        'Total direct emissions kgCO2e/day': total_direct_emissions,
        'Total small source emissions kgCO2e/day': small_source_emissions,
        'Total other offsite emissions kgCO2e/day': total_other_offsite_emissions,
        'Total embodied emissions kgCO2e/day': embodied_emissions_daily_rate
    }

result = calculate_total_emissions('Baseline')
print('Total emissions:', result['Mean Emissions kgCO2e/kgH2'], 'kgCO2e/kgH2')

# # def summarise_production_weighted_emissions(case):
#     print('Minimum (Year 1) production-weighted emissions: ' , min(production_weighted_emissions), 'kgCO2e/kgH2')
#     print('Mean production-weighted emissions: ', statistics.mean(production_weighted_emissions), 'kgCO2e/kgH2')
#     print('Median production-weighted emissions: ', statistics.median(production_weighted_emissions), 'kgCO2e/kgH2')
#     print('Maximum (Year 30) production-weighted emissions: ', max(production_weighted_emissions), 'kgCO2e/kgH2')


# print('Mean production weighted embodied emissions: ', statistics.mean(production_weighted_embodied_emissions), 'kgCO2e/kgH2')

# print('Percentage of mean production-weighted embodied emissions to mean production-weighted total emissions:',statistics.mean(production_weighted_embodied_emissions)/statistics.mean(production_weighted_emissions)*100,'%')

# #Calculate the total emissions over the lifetime of the field:
# total_emissions_absolute = sum(total_emissions * 365) #kg CO2e. This is the total emissions over the lifetime of the field. Assumes emissions are constant over each year of the field's lifetime.

# print('Total emissions over the lifetime of the field:', total_emissions_absolute, 'tonne CO2e')
# print('Relative contribution of embodied emissions to total emissions:', total_embodied_emissions/total_emissions_absolute*100, '%')

# Test Usage:
# print(calculate_total_emissions('Baseline',GWP_H2))


# ## 7.2 Functions to calculate total emissions and total hydrogen produced over field life.
# 
# Calculate abolute total emissions and productions to calculate average emissions intensity over whole field life.

# In[ ]:


#Create a function to calculate the total emissions for each case, over the entire field lifetime. This will return a single value for each case, representing the total emissions over the lifetime of the field.:

def calculate_total_lifetime_emissions(case, sensitivity_variables=None):
    total_emissions = calculate_total_emissions(case, sensitivity_variables)['Total lifetime emissions kgCO2e']
    return {
        'case': case,
        'total_lifetime_emissions': total_emissions
    }

# Test Usage:
print(calculate_total_lifetime_emissions('Baseline'))

#Create a function to calculate the total amount of hydrogen produced over the lifetime of the field. This will return a single value for each case, representing the total amount of hydrogen produced over the lifetime of the field.:
def calculate_total_hydrogen_produced(case,sensitivity_variables=None):
    if sensitivity_variables:
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
    else:
        field_lifespan = field_lifespan_default

    daily_field_H2_exported = calculate_exploration_emissions(case,sensitivity_variables)['daily_field_H2_exported'] #tonnes/day. This is the daily amount of hydrogen exported from the field.
    total_hydrogen_produced = sum(daily_field_H2_exported * 365 *1000) #kg. This is the total amount of hydrogen produced over the lifetime of the field.
    return {
        'case': case,
        'total_hydrogen_produced': total_hydrogen_produced
    }

# Test Usage:
print(calculate_total_hydrogen_produced('Baseline'))

#Create a function to calculate average emissions per kg of hydrogen produced. This will return a single value for each case, representing the average emissions per kg of hydrogen produced over the lifetime of the field.:
def calculate_average_emissions_per_kg_hydrogen(case,sensitivity_variables=None):
    if sensitivity_variables:
        field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
        oil_production = sensitivity_variables.get('oil_production', oil_production_default)
        small_source_emissions_percentage = sensitivity_variables.get('small_source_emissions_percentage', small_source_emissions_percentage_default)
    else:
        field_lifespan = field_lifespan_default
        oil_production = oil_production_default
        small_source_emissions_percentage = small_source_emissions_percentage_default

    total_emissions = calculate_total_emissions(case,sensitivity_variables)['Total lifetime emissions kgCO2e']
    total_hydrogen_produced = calculate_total_hydrogen_produced(case,sensitivity_variables)['total_hydrogen_produced']
    average_emissions_per_kg_hydrogen = total_emissions / total_hydrogen_produced
    return {
        'case': case,
        'average_emissions_per_kg_hydrogen': average_emissions_per_kg_hydrogen
    }

# Test Usage:
# print(calculate_average_emissions_per_kg_hydrogen('Baseline'))
case = 'Baseline'
print('Total emissions per total amount of hydrogen produced in the lifetime of the field in ',case,' case:', calculate_average_emissions_per_kg_hydrogen(case)['average_emissions_per_kg_hydrogen'], 'kgCO2e/kgH2')
print('Production-weighted mean emissions per total amount of hydrogen produced in the lifetime of the field in ',case,' case:', calculate_total_emissions(case)['Mean Emissions kgCO2e/kgH2'], 'kgCO2e/kgH2')


# ## 7.4 Calculate and Store Emissions Results for All Cases

# In[ ]:


# First, define the cases based on the reservoir data dataframe:
cases = reservoir_data['Case']

# Gather data for each case
emissions_data = [calculate_total_emissions(case) for case in cases]
# Filter out None values from unsuccessful calculations
emissions_data = [data for data in emissions_data if data is not None]

# Convert to DataFrame
emissions_df = pd.DataFrame(emissions_data)

# Set 'case' as the index for easier data manipulation
emissions_df.set_index('case', inplace=True)

# Check and retain only columns with numeric data
numeric_columns = [col for col in emissions_df.columns if pd.api.types.is_numeric_dtype(emissions_df[col])]
emissions_df = emissions_df[numeric_columns]

# Pivot the DataFrame for better comparison
# Since 'case' is already the index, we directly use the transpose for a simple pivot
pivoted_emissions_df = emissions_df.T  # Transpose to have cases as columns and statistics as rows

# Display the DataFrame
print(pivoted_emissions_df)


# In[ ]:


#Create a horizontal bar plot of mean emissions per kg of hydrogen produced for each case:

# Set the figure size
plt.figure(figsize=(10, 6))

# Create the horizontal bar plot
plt.barh(pivoted_emissions_df.columns, pivoted_emissions_df.loc['Mean Emissions kgCO2e/kgH2'])

# Add title and labels
plt.title('Mean Emissions per kg of Hydrogen Produced')
plt.xlabel('Mean Emissions (kgCO2e/kgH2)')
plt.ylabel('Case')

# Display the plot
plt.show()


# ## 7.5 Monte Carlo Uncertainty Analysis
# 
# Create structures to perform Monte Carlo analysis to examine effects of uncertain assumptions on emissions intensity results

# ### 7.5.1 Set up functions to handle the MC analysis

# In[ ]:


# # Define a function that handle the presence or absense of sensitivity variables and return the appropriate results, to be used within each calculation function

# def sensitivity_variable_handling(sensitivity_variables):
#     if sensitivity_variables:
#         GWP_H2 = sensitivity_variables.get('GWP_H2', GWP_H2_default)
#         oil_production = sensitivity_variables.get('oil_production', oil_production_default)
#         water_production = sensitivity_variables.get('water_production', water_production_default)
#         small_source_emissions_percentage = sensitivity_variables.get('small_source_emissions_percentage', small_source_emissions_percentage_default)
#         total_producing_wells = sensitivity_variables.get('Total Producing Wells', number_production_wells_default)
#         field_lifespan = sensitivity_variables.get('Field Life', field_lifespan_default)
#         water_cut = sensitivity_variables.get('Water Cut (bbl/mmscf)', water_cut_default)
#         h2_purification_loss_rate = sensitivity_variables.get('H2 purification loss rate', h2_purification_loss_rate_default)
#         pressure_decline_rate = sensitivity_variables.get('pressure_decline_rate', pressure_decline_rate_default)
#     else:
#         GWP_H2 = GWP_H2_default
#         oil_production = oil_production_default
#         water_production = water_production_default
#         small_source_emissions_percentage = small_source_emissions_percentage_default
#         total_producing_wells = number_production_wells_default
#         field_lifespan = field_lifespan_default
#         water_cut = water_cut_default
#         h2_purification_loss_rate = h2_purification_loss_rate_default
#         pressure_decline_rate = pressure_decline_rate_default 

#     return {GWP_H2, oil_production, water_production, small_source_emissions_percentage, total_producing_wells, field_lifespan, water_cut, h2_purification_loss_rate, pressure_decline_rate}


# In[ ]:


# Convert this notebook to a python file to enable more efficient, parallel calcluation in a separate notebook.

get_ipython().system('jupyter nbconvert --to script BrandtModelReplication_v0_2.ipynb')


# In[272]:


# All functions have been designed to optionally include sensitivity variables. If sensitivity variables are not provided, the default values will be used.
# This section defines functions to perform sensitivity analyses as well as defining the assumed probability density functions and distributions for the sensitivity variables.

# Repeat the definition of calculate_average_emissions_per_kg_hydrogen to test parallel computation code

# def calculate_average_emissions_per_kg_hydrogen(case,sensitivity_variables=None):
#     if sensitivity_variables:
#         field_lifespan = sensitivity_variables.get('field_lifespan', field_lifespan_default)
#         oil_production = sensitivity_variables.get('oil_production', oil_production_default)
#         small_source_emissions_percentage = sensitivity_variables.get('small_source_emissions_percentage', small_source_emissions_percentage_default)
#     else:
#         field_lifespan = field_lifespan_default
#         oil_production = oil_production_default
#         small_source_emissions_percentage = small_source_emissions_percentage_default

#     total_emissions = calculate_total_emissions(case,sensitivity_variables)['Total lifetime emissions kgCO2e']
#     total_hydrogen_produced = calculate_total_hydrogen_produced(case,sensitivity_variables)['total_hydrogen_produced']
#     average_emissions_per_kg_hydrogen = total_emissions / total_hydrogen_produced
#     return {
#         'case': case,
#         'average_emissions_per_kg_hydrogen': average_emissions_per_kg_hydrogen
#     }

def perform_sensitivity_analysis(case, sensitivity_variables):
    first_variable = sensitivity_variables[list(sensitivity_variables.keys())[0]]
    N = len(first_variable)
    results = np.zeros(N)
    all_cases = [{key: value[i] for key, value in sensitivity_variables.items()} for i in range(N)]
    for counter, case_data in enumerate(all_cases):
        average_emissions_per_kg_hydrogen = calculate_average_emissions_per_kg_hydrogen(case, case_data)
        results[counter] = average_emissions_per_kg_hydrogen['average_emissions_per_kg_hydrogen']
    return pd.DataFrame({**sensitivity_variables, 'average_emissions_per_kg_hydrogen': results})


def perform_sensitivity_analysis_parallel(case, sensitivity_variables):
    first_variable = sensitivity_variables[list(sensitivity_variables.keys())[0]]
    N = len(first_variable)
    all_cases = [{key: value[i] for key, value in sensitivity_variables.items()} for i in range(N)]
    
    # Limit number of processes to avoid overwhelming the system
    # num_cores = min(4, os.cpu_count() - 1)  # Adjust number of processes if needed
    num_cores = os.cpu_count() - 2 # Test performance using all but two cores
    results = Parallel(n_jobs=num_cores)(
        delayed(calculate_average_emissions_per_kg_hydrogen)(case, case_data) for case_data in all_cases
    )
    
    results_values = [result['average_emissions_per_kg_hydrogen'] for result in results]
    return pd.DataFrame({**sensitivity_variables, 'average_emissions_per_kg_hydrogen': results_values})

# Define the sensitivity analysis assumptions
N = 100  # Number of samples to be generated for the Monte Carlo simulation

# Set the random seed so the results are repeatable
np.random.seed(123)
sensitivity_assumptions = {
    'GWP_H2': np.random.uniform(2, 15, N),
    'oil_production': np.random.uniform(0.01, 10, N),
    'water_production': np.random.uniform(0.01, 10, N),
    'small_source_emissions_percentage': np.random.uniform(1, 15, N),
    'Total Producing Wells': np.random.randint(1, 100, N),
    'Field Life': np.random.randint(5, 50, N),
    'Water Cut (bbl/mmscf)': np.random.uniform(0.01, 10, N),
    'H2 purification loss rate': np.random.uniform(1, 20, N),
    'pressure_decline_rate': np.random.uniform(0.98, 85, N),
    'number_production_wells': np.random.randint(10, 100, N)
}

# # Run sensitivity analysis for the 'Baseline' case
# sensitivity_results = perform_sensitivity_analysis('Baseline', sensitivity_assumptions)
# print(sensitivity_results)

# Run sensitivity analysis for the 'Baseline' case
sensitivity_results = perform_sensitivity_analysis_parallel('Baseline', sensitivity_assumptions)

# Calculate basic statistics for the sensitivity analysis results:
sensitivity_statistics = sensitivity_results['average_emissions_per_kg_hydrogen'].describe()
print(sensitivity_statistics)

# Describe the statistics of sensitivity_assumptions
sensitivity_assumptions_df = pd.DataFrame(sensitivity_assumptions)
sensitivity_assumptions_statistics = sensitivity_assumptions_df.describe()
print(sensitivity_assumptions_statistics)

# Set simplistic sensitivity assumptions to help debug tornado plots:
# sensitivity_assumptions = {
    # 'GWP_H2': np.array([1, 15]),
    # 'oil_production': np.array([0.01, 10]),
    # 'water_production': np.array([0.01, 10]),
    # 'small_source_emissions_percentage': np.array([1,20]),
    # 'Total Producing Wells': np.array([1, 100]),
    # 'Field Life': np.array([5, 50]),
    # 'Water Cut (bbl/mmscf)': np.array([0.01, 10]),
    # 'H2 purification loss rate': np.array([1, 20]),
    # 'pressure_decline_rate': np.array([0.98, 85]),
    # 'number_production_wells': np.array([10, 100])
# }

# # Run sensitivity analysis for the 'Baseline' case
# sensitivity_results = perform_sensitivity_analysis_parallel('Baseline', sensitivity_assumptions)
# # print(sensitivity_results['average_emissions_per_kg_hydrogen'])

# # Calculate basic statistics for the sensitivity analysis results:
# sensitivity_statistics = sensitivity_results['average_emissions_per_kg_hydrogen'].describe()
# # print(sensitivity_statistics)

# Describe the statistics of sensitivity_assumptions

# sensitivity_assumptions_df = pd.DataFrame(sensitivity_assumptions)
# sensitivity_assumptions_statistics = sensitivity_assumptions_df.describe()
# print(sensitivity_assumptions_statistics)



# In[273]:


# Use a loop to calculate the sensitivity statistics for each case and store the results in a dataframe for display:
sensitivity_results_dict = {}
for case in cases:
    sensitivity_results = perform_sensitivity_analysis_parallel(case, sensitivity_assumptions)
    sensitivity_statistics = sensitivity_results['average_emissions_per_kg_hydrogen'].describe()
    sensitivity_results_dict[case] = sensitivity_statistics

# Create a DataFrame from the sensitivity results dictionary
sensitivity_results_df = pd.DataFrame(sensitivity_results_dict)

# Display the DataFrame
print(sensitivity_results_df)


# In[247]:


# Plot a histogram of the sensitivity analysis results for the 'Baseline' case:

# Replacing infinite values with NaN to ensure future compatibility with seaborne plotting:
sensitivity_results.replace([np.inf, -np.inf], np.nan, inplace=True)

plt.figure(figsize=(10, 6))
plt.hist(sensitivity_results['average_emissions_per_kg_hydrogen'], bins=30, color='skyblue', edgecolor='black')
plt.title('Sensitivity Analysis Results for Baseline Case')
plt.xlabel('Average Emissions per kg of Hydrogen Produced (kgCO2e/kgH2)')
plt.ylabel('Frequency')
plt.show()

# Plot a cumulative distribution function (CDF) of the sensitivity analysis results for the 'Baseline' case:
plt.figure(figsize=(10, 6))
sns.ecdfplot(sensitivity_results['average_emissions_per_kg_hydrogen'], color='skyblue')
plt.title('Cumulative Distribution Function (CDF) of Sensitivity Analysis Results for Baseline Case')
plt.xlabel('Average Emissions per kg of Hydrogen Produced (kgCO2e/kgH2)')
plt.ylabel('Cumulative Probability')
plt.show()



# ### 7.5.2 Tornado charts based on MC analysis

# In[249]:


# Make a Tornado Chart that is centered around the deterministic value of the case under consideration:

# # Functions to perform sensitivity analysis while only changing a single variable at a time (i.e. assuming variables are independent), using parallel computation methods to reduce computation time.

def calculate_effect(case, key, value, default_case):
    case_data = default_case.copy()
    case_data[key] = value
    return calculate_average_emissions_per_kg_hydrogen(case, case_data)['average_emissions_per_kg_hydrogen']

def perform_sensitivity_analysis_single_variable(case, sensitivity_variables):
    # Calculate the deterministic baseline result for the specified case
    baseline_result = calculate_average_emissions_per_kg_hydrogen(case)['average_emissions_per_kg_hydrogen']

    results = {}
    num_cores = -1  # Use all available CPU cores

    for key, values in sensitivity_variables.items():
        # Prepare the default case with default values
        default_case = {
            'field_lifespan': field_lifespan_default,
            'GWP_H2': GWP_H2_default,
            'number_production_wells': number_production_wells_default,
            'oil_production': oil_production_default,
            'water_production': water_production_default,
            'small_source_emissions_percentage': small_source_emissions_percentage_default,
            'Total Producing Wells': number_production_wells_default,
            'Field Life': field_lifespan_default,
            'Water Cut (bbl/mmscf)': water_cut_default,
            'H2 purification loss rate': h2_purification_loss_rate_default,
            'pressure_decline_rate': pressure_decline_rate_default
        }

        # Parallelize the computation of effects
        effects = Parallel(n_jobs=num_cores)(
            delayed(calculate_effect)(case, key, value, default_case) for value in values
        )

        results[key] = {
            'min_effect': np.min(effects),
            'max_effect': np.max(effects)
        }

    return baseline_result, results  # Return the baseline result and sensitivity analysis results


# baseline_result, tornado_sensitivity_results = perform_sensitivity_analysis_single_variable('Baseline', sensitivity_assumptions)

# # Extract the min and max effects and center them around the baseline result
# min_effects = [result['min_effect'] for result in tornado_sensitivity_results.values()]
# max_effects = [result['max_effect'] for result in tornado_sensitivity_results.values()]

# # Calculate the deviations from the baseline result
# min_deviation = [min_effect - baseline_result for min_effect in min_effects]
# max_deviation = [max_effect - baseline_result for max_effect in max_effects]

# # Create a DataFrame for plotting
# tornado_df = pd.DataFrame({
#     'Variable': list(tornado_sensitivity_results.keys()),
#     'Min Deviation': min_deviation,
#     'Max Deviation': max_deviation
# })

# # Sort the DataFrame by the absolute maximum deviation
# tornado_df['Range of Effect'] = tornado_df['Max Deviation'] - tornado_df['Min Deviation']
# tornado_df.sort_values(by='Range of Effect', ascending=False, inplace=True)

# # Plot the tornado chart centered around the baseline result
# plt.figure(figsize=(10, 6))
# ax = sns.barplot(x='Min Deviation', y='Variable', data=tornado_df, palette='viridis', label='Min Deviation')
# sns.barplot(x='Max Deviation', y='Variable', data=tornado_df, palette='viridis', label='Max Deviation')

# plt.axvline(0, color='k', linestyle='--')
# plt.xlabel(f'Deviation from Baseline Result ({baseline_result:.2f} kg CO2e/kg H2)')
# plt.ylabel('Sensitivity Variable')
# plt.title(f'Tornado Chart: Sensitivity Analysis Centered on Baseline ({baseline_result:.2f} kg CO2e/kg H2)')
# # plt.legend(title='Deviation Type')
# plt.show()

# print(tornado_df)

#Using the logic above, create a function that creates a tornado plot for a given case and sensitivity assumptions:

# def create_tornado_plot(case, sensitivity_assumptions):
#     baseline_result, tornado_sensitivity_results = perform_sensitivity_analysis_single_variable(case, sensitivity_assumptions)

#     # Extract the min and max effects and center them around the baseline result
#     min_effects = [result['min_effect'] for result in tornado_sensitivity_results.values()]
#     max_effects = [result['max_effect'] for result in tornado_sensitivity_results.values()]

#     # Calculate the deviations from the baseline result
#     min_deviation = [min_effect - baseline_result for min_effect in min_effects]
#     max_deviation = [max_effect - baseline_result for max_effect in max_effects]

#     #  Update the variable names to include min and max values used in Monte Carlo analysis in parentheses:
#     variable_names = []
#     for variable in tornado_df['Variable']:
#         min_value = min(sensitivity_assumptions[variable])
#         max_value = max(sensitivity_assumptions[variable])
#         variable_names.append(f'{variable} ({min_value:.2f}, {max_value:.2f})')

#     # Create a DataFrame for plotting
#     tornado_df = pd.DataFrame({
#         'Variable': variable_names,
#         'Min Deviation': min_deviation,
#         'Max Deviation': max_deviation
#     })

#     # Sort the DataFrame by the absolute maximum deviation
#     tornado_df['Range of Effect'] = tornado_df['Max Deviation'] - tornado_df['Min Deviation']
#     tornado_df.sort_values(by='Range of Effect', ascending=False, inplace=True)




#     # Plot the tornado chart centered around the baseline result
#     plt.figure(figsize=(10, 6))
#     ax = sns.barplot(x='Min Deviation', y='Variable', data=tornado_df, palette='viridis', label='Min Deviation')
#     sns.barplot(x='Max Deviation', y='Variable', data=tornado_df, palette='viridis', label='Max Deviation')

#     plt.axvline(0, color='k', linestyle='--')
#     plt.xlabel(f'Deviation from {case} Result ({baseline_result:.2f} kg CO2e/kg H2)')
#     plt.ylabel('Sensitivity Variable')

#     # # For each of the variables, display the minimum and maximum value used within the Monte Carlo simulation as text on the plot:

#     # for i, variable in enumerate(tornado_df['Variable']):
#     #     min_value = min(sensitivity_assumptions[variable])
#     #     max_value = max(sensitivity_assumptions[variable])
#     #     plt.text(-0.1, i, f'{min_value:.2f}', va='center', ha='right', color='black')
#     #     plt.text(0.1, i, f'{max_value:.2f}', va='center', ha='left', color='black')

 
#     #Plot the title such that it refers to the case under consideration:
#     plt.title(f'Tornado Chart: Sensitivity Analysis Centered on {case} ({baseline_result:.2f} kg CO2e/kg H2)', loc='center')
#     plt.show()



# Function to create a tornado plot for a given case and sensitivity assumptions
def create_tornado_plot(case, sensitivity_assumptions):
    baseline_result, tornado_sensitivity_results = perform_sensitivity_analysis_single_variable(case, sensitivity_assumptions)

    # Extract the min and max effects and center them around the baseline result
    min_effects = [result['min_effect'] for result in tornado_sensitivity_results.values()]
    max_effects = [result['max_effect'] for result in tornado_sensitivity_results.values()]

    # Calculate the deviations from the baseline result
    min_deviation = [min_effect - baseline_result for min_effect in min_effects]
    max_deviation = [max_effect - baseline_result for max_effect in max_effects]

    # Create a DataFrame for plotting
    tornado_df = pd.DataFrame({
        'Variable': list(tornado_sensitivity_results.keys()),
        'Min Deviation': min_deviation,
        'Max Deviation': max_deviation
    })

    

    # Update the variable names to include min and max values used in Monte Carlo analysis in parentheses
    variable_names_with_numbers = []
    for variable in tornado_df['Variable']:
        min_value = min(sensitivity_assumptions[variable])
        max_value = max(sensitivity_assumptions[variable])
        variable_names_with_numbers.append(f'{variable} \n ({min_value:.2f}, {max_value:.2f})')

    tornado_df['Variable with numbers'] = variable_names_with_numbers

    # Sort the DataFrame by the absolute maximum deviation
    tornado_df['Range of Effect'] = tornado_df['Max Deviation'] - tornado_df['Min Deviation']
    tornado_df.sort_values(by='Range of Effect', ascending=False, inplace=True)

    # Plot the tornado chart centered around the baseline result
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Min Deviation', y='Variable with numbers', data=tornado_df, palette='viridis', label='Min Deviation')
    sns.barplot(x='Max Deviation', y='Variable with numbers', data=tornado_df, palette='viridis', label='Max Deviation') 

    # # Add text labels for the min and max values of each variable, located to the left of the minimum deviation and to the right of the maximum deviation, for each variable
    # for i, variable in enumerate(tornado_df['Variable']):
    #     plt.text(variable['Min Effect']-0.001, i, f'{min_value:.2f}', va='center', ha='right', color='black')
    #     plt.text(variable['Max Effect']+0.001, i, f'{max_value:.2f}', va='center', ha='left', color='black')

    #     # Add text labels for the min and max values at the ends of the bars

    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel(f'Deviation from {case} Result ({baseline_result:.2f} kg CO2e/kg H2)')
    plt.ylabel('Sensitivity Variable')
    plt.title(f'Tornado Chart: Sensitivity Analysis Centered on {case} ({baseline_result:.2f} kg CO2e/kg H2)', loc='center')
    plt.show()

    return tornado_df

tornado_df = create_tornado_plot('Baseline', sensitivity_assumptions)
print(tornado_df)
# # Test the function with the 'Baseline' case and sensitivity assumptions
# create_tornado_plot('Baseline', sensitivity_assumptions)


# In[250]:


# #Create another tornado plot using the same data but excluding small_source_emissions_percentage

# # Run sensitivity analysis for the 'Baseline' case
# tornado_sensitivity_results = perform_sensitivity_analysis_single_variable('Baseline', sensitivity_assumptions)

# # Extract the min and max effects
# min_effects = [result['min_effect'] for key, result in tornado_sensitivity_results.items() if key != 'small_source_emissions_percentage']
# max_effects = [result['max_effect'] for key, result in tornado_sensitivity_results.items() if key != 'small_source_emissions_percentage']
# variables = [key for key in tornado_sensitivity_results.keys() if key != 'small_source_emissions_percentage']

# # Calculate the ranges of effects
# ranges_of_effects = np.array(max_effects) - np.array(min_effects)

# # Create a DataFrame for plotting
# tornado_df = pd.DataFrame({
#     'Variable': variables,
#     'Min Effect': min_effects,
#     'Max Effect': max_effects,
#     'Range of Effect': ranges_of_effects
# })

# # Sort the DataFrame by the range of effect
# tornado_df.sort_values(by='Range of Effect', ascending=False, inplace=True)

# # Plot the tornado chart
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Range of Effect', y='Variable', data=tornado_df, palette='viridis')
# plt.xlabel('Range of Effect on Average Emissions (kg CO2e/kg H2)')
# plt.ylabel('Sensitivity Variable')
# plt.title('Tornado Chart: Sensitivity Analysis')
# plt.show()


# In[251]:


def perform_sensitivity_analysis(case, sensitivity_variables):
    first_variable = sensitivity_variables[list(sensitivity_variables.keys())[0]]
    N = len(first_variable)
    results = np.zeros(N)
    all_cases = [{key: value[i] for key, value in sensitivity_variables.items()} for i in range(N)]
    for counter, case_data in enumerate(all_cases):
        average_emissions_per_kg_hydrogen = calculate_average_emissions_per_kg_hydrogen(case, case_data)
        results[counter] = average_emissions_per_kg_hydrogen
    return pd.DataFrame({**sensitivity_variables, 'average_emissions_per_kg_hydrogen': results})

def perform_sensitivity_analysis_parallel(case, sensitivity_variables):
    num_cores = max([1, mp.cpu_count()-1])
    first_variable = sensitivity_variables[list(sensitivity_variables.keys())[0]]
    N = len(first_variable)
    all_cases = [{key: value[i] for key, value in sensitivity_variables.items()} for i in range(N)]
    with mp.Pool(num_cores) as p:
        results = p.starmap(calculate_average_emissions_per_kg_hydrogen, zip(repeat(case), all_cases))
    return pd.DataFrame({**sensitivity_variables, 'average_emissions_per_kg_hydrogen': results})


# In[252]:


sensitivity_statistics = sensitivity_results.describe()
print(sensitivity_statistics)


# ### 7.5.3 Rank-Order Correlation & Regression Coefficients

# In[253]:


# Calculate the rank-order correlation between the sensitivity variables and the average emissions per kg of hydrogen produced
correlation_matrix = sensitivity_results.corr()
correlation_with_target = correlation_matrix['average_emissions_per_kg_hydrogen']
correlation_with_target = correlation_with_target.drop('average_emissions_per_kg_hydrogen')  # Remove the correlation with itself
correlation_with_target = correlation_with_target.abs().sort_values(ascending=True)
print(correlation_with_target)

#Plot the rank-order correlation results in a horizontal bar chart:
plt.figure(figsize=(10, 6))
correlation_with_target.plot(kind='barh', color='skyblue')
plt.title('Rank-order Correlation with Average Emissions per kg of Hydrogen Produced')
plt.xlabel('Correlation')
plt.ylabel('Sensitivity Variable')
plt.show()



# Calculate the rank-order regression water_mass_coefficients between the sensitivity variables and the average emissions per kg of hydrogen produced
X = sensitivity_results.drop(columns='average_emissions_per_kg_hydrogen')
y = sensitivity_results['average_emissions_per_kg_hydrogen']
regression_water_mass_coefficients = {}
for variable in X.columns:
    model = LinearRegression().fit(X[[variable]], y)
    regression_water_mass_coefficients[variable] = model.coef_[0]
regression_water_mass_coefficients = pd.Series(regression_water_mass_coefficients).abs().sort_values(ascending=True)
print(regression_water_mass_coefficients)

# Plot the rank-order regression water_mass_coefficients in a separate horizontal bar chart:
plt.figure(figsize=(10, 6))
regression_water_mass_coefficients.plot(kind='barh', color='skyblue')
plt.title('Rank-order Regression water_mass_coefficients with Average Emissions per kg of Hydrogen Produced')
plt.xlabel('Regression Coefficient')
plt.ylabel('Sensitivity Variable')
plt.show()


# # 8. Summary Plots

# In[254]:


#First summarise all of the emissions data in a single DataFrame

# Define your cases based on the reservoir data
cases = reservoir_data['Case']

# Initialize a list to store results for each case
emissions_summary_by_case = []

# Loop through each case and calculate emissions
for case in cases:
    # Calculate emissions for each case
    emissions_data = calculate_total_emissions(case)
    
    # Organize data into a dictionary for DataFrame creation
    emissions_summary_by_case.append({
        'Case': case,
        'Embodied emissions': emissions_data['Total embodied emissions kgCO2e/day'],
        'Direct emissions': emissions_data['Total direct emissions kgCO2e/day'],
        'Other offsite emissions': emissions_data['Total other offsite emissions kgCO2e/day'],
        'Small source emissions': emissions_data['Total small source emissions kgCO2e/day'],
        # 'Total emissions': emissions_data['Total Emissions kgCO2e/day']
    })

# Convert the list of dictionaries into a DataFrame
emissions_df = pd.DataFrame(emissions_summary_by_case)
emissions_df.set_index('Case', inplace=True)

# Display the DataFrame
emissions_df.head()



# In[255]:


#Now create a function to plot total emissions over time as a stacked bar chart of the different emission types:

def plot_emissions_by_case(case):
    # Ensure the case exists in the DataFrame
    if case not in emissions_df.index:
        print("Case not found in the DataFrame.")
        return
    
    # Extract each type of emission for the specified case into a DataFrame
    # Assuming each cell is already a properly formatted pd.Series or similar iterable
    data = {
        'Embodied emissions': emissions_df.loc[case, 'Embodied emissions'],
        'Direct emissions': emissions_df.loc[case, 'Direct emissions'],
        'Other offsite emissions': emissions_df.loc[case, 'Other offsite emissions'],
        'Small source emissions': emissions_df.loc[case, 'Small source emissions']
    }

    # Create a DataFrame where each column is a type of emission and each row is a year
    yearly_emissions_df = pd.DataFrame(data)

    # Ensure there is numeric data to plot
    if yearly_emissions_df.empty or yearly_emissions_df.dropna().empty:
        print("No data available to plot.")
        return

    #Offset the index by 1 to represent years
    yearly_emissions_df.index += 1

    # Plot a vertical stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    yearly_emissions_df.plot(kind='bar', stacked=True, ax=ax, width=0.9)
    ax.set_title(f'Stacked Emissions by Type Over Field Life ({case} Case)')
    ax.set_xlabel('Year of Field Life')
    ax.set_ylabel('Emissions (kg/day)')
    ax.legend(title='Emission Type')
    # Reverse the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Emission Type')
    plt.xticks(rotation=0)  # Ensure year labels are horizontal for readability 

    # Add the H2 production profile to the plot on a separate axis
    # Calculate daily_field_H2_exported before plotting
    daily_field_H2_exported = calculate_exploration_emissions(case)['daily_field_H2_exported']
    # Add a legend for the H2 production profile
    ax2 = ax.twinx()
    ax2.plot(daily_field_H2_exported, color='black', label='H2 Exported')
    ax2.set_ylabel('H2 Exported (tonne/day)')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
    ax2.grid(False)
    

    # ax.twinx()
    # plt.twinx()
    # plt.plot(daily_field_H2_exported, color='black', label='H2 Exported')
    # plt.ylabel('H2 Exported (tonne/day)')
    # plt.grid(False)

    plt.show()

# Example usage:
plot_emissions_by_case('Baseline')

# #Plots for all cases:
# for case in cases:
#     plot_emissions_by_case(case)


# In[256]:


#Now create a function to plot emissions intensity over time as a stacked bar chart of the different emission types:

def plot_emissions_intensity_by_case(case):
    if case not in emissions_df.index:
        print("Case not found in the DataFrame.")
        return
    
    # Fetch daily_field_H2_exported series for the given case
    daily_field_H2_exported = calculate_exploration_emissions(case)['daily_field_H2_exported']

    # Prepare the emissions data for the specific case, each type assumed to be a series over 30 years
    case_emissions = emissions_df.loc[case]

    # Normalize emissions by daily_field_H2_exported
    emissions_intensity_df = pd.DataFrame({
        'Embodied emissions': case_emissions['Embodied emissions'] / (daily_field_H2_exported * 1000),
        'Direct emissions': case_emissions['Direct emissions'] / (daily_field_H2_exported * 1000),
        'Other offsite emissions': case_emissions['Other offsite emissions'] / (daily_field_H2_exported * 1000),
        'Small source emissions': case_emissions['Small source emissions'] / (daily_field_H2_exported * 1000)
    })

    #Offset the index by 1 to represent years
    emissions_intensity_df.index += 1

    # Plot a vertical stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    emissions_intensity_df.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'Emissions Intensity by Type Over Field Life ({case} Case)')
    ax.set_xlabel('Year of Field Life')
    ax.set_ylabel('Emissions Intensity (kg CO2e/kg H2)')
    ax.legend(title='Emission Type', loc='upper right')
    # Reverse the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.xticks(rotation=0)  # Ensure year labels are horizontal for readability  
    plt.show()

# Example usage:
plot_emissions_intensity_by_case('Baseline')

# #Plots for all cases:
# for case in cases:
#     plot_emissions_intensity_by_case(case)


# In[257]:


# Now create a function to show the relative contribution of each emission type to the total annual for each year of the field's life:

def plot_relative_emissions_by_case(case):
    if case not in emissions_df.index:
        print("Case not found in the DataFrame.")
        return

    # Access the DataFrame row corresponding to the case
    case_data = emissions_df.loc[case]
    
    # Initialize a DataFrame to store the percentage values
    emissions_percentage_df = pd.DataFrame()

    # Calculate the total emissions for each year by summing over rows
    total_emissions_per_year = case_data.sum()

    # Calculate the relative contribution of each type of emission
    for col in case_data.index:
        emissions_percentage_df[col] = case_data[col] / total_emissions_per_year * 100

    #Offset the index by 1 to represent years
    emissions_percentage_df.index += 1

    # Plotting the stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    emissions_percentage_df.plot(kind='bar', stacked=True, ax=ax, width=0.9)
    ax.set_title(f'Relative Emissions by Type Over Field Life ({case} Case)')
    ax.set_xlabel('Year of Field Life')
    ax.set_ylabel('Relative Emissions (%)')
   
    plt.xticks(rotation=0)  # Ensure year labels are horizontal for readability
    
  # Reverse the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],title='Emission Type', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

# Example usage
plot_relative_emissions_by_case('Baseline')

# #Plots for all cases:
# for case in cases:
#     plot_relative_emissions_by_case(case)


# In[ ]:





# In[258]:


### Now creating a function to inspect the results EXCLUDING embodied emissions for each case:

def plot_emissions_excluding_embodied(case):
    if case not in emissions_df.index:
        print("Case not found in the DataFrame.")
        return
    
    # Fetch daily_field_H2_exported series for the given case
    daily_field_H2_exported = calculate_exploration_emissions(case)['daily_field_H2_exported']

    # Ensure it's properly formatted
    if not isinstance(daily_field_H2_exported, pd.Series) or len(daily_field_H2_exported) != 30:
        print("Daily H2 exported data is not correctly formatted or has incorrect length.")
        return

    # Extract the case data for emissions, dropping 'Embodied emissions'
    case_data = emissions_df.loc[case].drop('Embodied emissions')

    # Ensure the data is aligned properly: case_data should have the same length as daily_field_H2_exported
    if any(len(data) != 30 for data in case_data):
        print("Emissions data does not match expected yearly format.")
        return
    
    # Check structure and reformat if necessary
    if isinstance(case_data, pd.Series):
        # Assuming each element in the Series is another Series of yearly data
        # Convert the series of series into a DataFrame
        formatted_data = pd.DataFrame({etype: data.values for etype, data in case_data.items()})

    # Transpose so that rows are years and columns are emission types
    # formatted_data = formatted_data.T

    # Plot
    ax = formatted_data.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.ylabel('Emissions (kg/day)')
    plt.xlabel('Year')
    plt.xticks(rotation=0)  # Ensure year labels are horizontal for readability 
    plt.title(f'Emissions by Type Over Field Life (Excluding Embodied Emissions) for {case}')
    plt.legend(title='Emission Type')
    plt.show()

    # Normalize emissions by daily_field_H2_exported to get intensity
    emissions_intensity_df = case_data.apply(lambda x: x / (daily_field_H2_exported * 1000))

    # Plot the emissions intensity by type for each year
    ax = emissions_intensity_df.T.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.ylabel('Emissions Intensity (kg CO2e/kg H2)')
    plt.xlabel('Year')
    plt.xticks(rotation=0)  # Ensure year labels are horizontal for readability 
    plt.title(f'Breakdown of Emissions Intensity by Type (Excluding Embodied Emissions) for {case}')
    plt.legend(title='Emission Type', loc='upper right')
    plt.show()
    

    # Calculate the relative contributions of each emission type to the total emissions per year
    total_emissions_no_embodied = case_data.sum()
    emissions_percent_df = case_data.apply(lambda x: (x / total_emissions_no_embodied) * 100)
    ax = emissions_percent_df.T.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.ylabel('Percent of Total Emissions (%)')
    plt.xlabel('Year')
    plt.xticks(rotation=0)  # Ensure year labels are horizontal for readability 
    plt.title(f'Breakdown of Emissions by Type as Percent of Total Emissions (Excluding Embodied Emissions) for {case}')
    plt.legend(title='Emission Type', loc='upper right')
    plt.show()

# Example usage:
plot_emissions_excluding_embodied('Baseline')

#Plots for all cases:
# for case in cases:
#     plot_emissions_excluding_embodied(case)


# In[259]:


# #Print minimum, mean, median, and maximum production-weighted emissions, excluding embodied emissions
# production_weighted_emissions_no_embodied = total_emissions_no_embodied / (daily_field_H2_exported * 1000)

# print(f'Minimum (Year 1) production-weighted emissions (excluding embodied emissions): {min(production_weighted_emissions_no_embodied):.3g} kgCO2e/kgH2')
# print(f'Mean production-weighted emissions (excluding embodied emissions): {statistics.mean(production_weighted_emissions_no_embodied):.3g} kgCO2e/kgH2')
# print(f'Median production-weighted emissions (excluding embodied emissions): {statistics.median(production_weighted_emissions_no_embodied):.3g} kgCO2e/kgH2')
# print(f'Maximum (Year 30) production-weighted emissions (excluding embodied emissions): {max(production_weighted_emissions_no_embodied):.3g} kgCO2e/kgH2')

#Create a function to calculate summary statistics for each case:
def calculate_emissions_statistics_excluding_embodied(case):
    if case not in emissions_df.index:
        print("Case not found in the DataFrame.")
        return
    
    # Fetch daily_field_H2_exported series for the given case
    daily_field_H2_exported = calculate_exploration_emissions(case)['daily_field_H2_exported']

    # Ensure it's properly formatted
    if not isinstance(daily_field_H2_exported, pd.Series) or len(daily_field_H2_exported) != 30:
        print("Daily H2 exported data is not correctly formatted or has incorrect length.")
        return

    # Extract the case data for emissions, dropping 'Embodied emissions'
    case_data = emissions_df.loc[case].drop('Embodied emissions')

    # Ensure the data is aligned properly: case_data should have the same length as daily_field_H2_exported
    if any(len(data) != 30 for data in case_data):
        print("Emissions data does not match expected yearly format.")
        return
    
    # Check structure and reformat if necessary
    if isinstance(case_data, pd.Series):
        # Assuming each element in the Series is another Series of yearly data
        # Convert the series of series into a DataFrame
        formatted_data = pd.DataFrame({etype: data.values for etype, data in case_data.items()})

    # Transpose so that rows are years and columns are emission types
    # formatted_data = formatted_data.T

    # Calculate the total emissions for each year by summing over rows
    total_emissions_per_year = case_data.sum()

    # Normalize emissions by daily_field_H2_exported to get intensity
    emissions_intensity_df = case_data.apply(lambda x: x / (daily_field_H2_exported * 1000))

    # Calculate the relative contributions of each emission type to the total emissions per year
    total_emissions_no_embodied = case_data.sum()
    emissions_percent_df = case_data.apply(lambda x: (x / total_emissions_no_embodied) * 100)

    # Calculate the production-weighted emissions:
    production_weighted_emissions_no_embodied = total_emissions_no_embodied / (daily_field_H2_exported * 1000)

    # Calculate statistics
    min_production_weighted_emissions = min(production_weighted_emissions_no_embodied)  # Convert Series to list
    mean_production_weighted_emissions = statistics.mean(production_weighted_emissions_no_embodied)  # Convert Series to list
    median_production_weighted_emissions = statistics.median(production_weighted_emissions_no_embodied)  # Convert Series to list
    max_production_weighted_emissions = max(production_weighted_emissions_no_embodied)  # Convert Series to list
    
    return {
        'case': case,
        'Min (Year 1) Emissions kgCO2e/kgH2 (excluding embodied)': min_production_weighted_emissions,
        'Mean Emissions kgCO2e/kgH2 (excluding embodied)': mean_production_weighted_emissions,
        'Median Emissions kgCO2e/kgH2 (excluding embodied)': median_production_weighted_emissions,
        'Max (Year 30) Emissions kgCO2e/kgH2 (excluding embodied)': max_production_weighted_emissions
    }

# Example usage:
calculate_emissions_statistics_excluding_embodied('Baseline')


# In[260]:


production_profile_df.head()


# In[261]:


def plot_gas_production(case):
    #Convert baseline gas production from MMSCF/day to tonnes/day
    MMSCF_to_tonnes = 0.178107606679035 #tonnes/MMSCF. Conversion factor from MMSCF to tonnes.

    total_gas_mass_flow_upstream_separator[case]= calculate_development_drilling_emissions(case)['total_gas_mass_flow_upstream_separator']
    H2_after_separator = calculate_total_production_vent_emissions(case)['H2_after_separator']
    daily_field_H2_exported = calculate_exploration_emissions(case)['daily_field_H2_exported']    

    #Give me a line plot of baseline gas production, H2 production, and H2 exported
    plt.figure(figsize=(10,6))
    plt.plot(total_gas_mass_flow_upstream_separator[case], label='Total Gas Production')
    plt.plot(H2_after_separator, label='H2 Production')
    plt.plot(daily_field_H2_exported, label='H2 Exported')
    plt.xlabel('Year')
    plt.ylabel('Production (tonnes / day)')
    plt.title('Comparison of Baseline Gas Production, H2 Production, and H2 Exported')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
plot_gas_production('Baseline')

# # #Plots for all cases:
# for case in cases:
#     plot_gas_production(case)



# In[262]:


total_steel_emissions = (calculate_well_steel_mass_MC('Baseline')['total_steel_mass_all_wells'] * steel_emissions_intensity)/1000 #kgCO2e. This is the total emissions associated with steel use in well construction.

total_steel_emissions


# In[263]:


# Create a function to calculate the total amount of emissions in the steel and cement needed to make the wells, and express this as a percentage of all embodied emissions for each case:
# Need to update embodied emissions calculations to consider case differences before this function is necessary.

def calculate_steel_and_cement_emissions(case):
    #Calculate the total mass of steel needed for all wells:
    # total_steel_mass_all_wells = calculate_development_drilling_emissions(case)['total_steel_mass_all_wells'] #kg. This is the total mass of steel needed for all wells in the field.

    #Calculate the total amount of emissions in the steel needed to make the wells:
    total_steel_emissions = (calculate_well_steel_mass_MC(case)['total_steel_mass_all_wells'] * steel_emissions_intensity)/1000 #kgCO2. This is the total emissions associated with the steel used in the wells.

    #Calculate the total amount of emissions in the cement needed to make the wells:
    total_cement_emissions_kg = calculate_total_cement_emissions(case)['total_cement_emissions']/1000 #kgCO2. This is the total emissions associated with the cement used in the wells.

    #Calculate this as a percentage of all embodied emissions:
    total_steel_and_cement_emissions_percent = (total_steel_emissions + total_cement_emissions_kg) / calculate_total_embodied_emissions(case)['total_embodied_emissions'] * 100

    return {
        'case': case,
        'total_steel_and_cement_emissions': total_steel_emissions + total_cement_emissions_kg,
        'total_steel_and_cement_emissions_percent': total_steel_and_cement_emissions_percent
    }


check_case = 'Deep'
steel_and_cement_emissions = calculate_steel_and_cement_emissions(check_case)

# Total embodied emissions in tonnes CO2
total_emissions_tonnes = steel_and_cement_emissions['total_steel_and_cement_emissions'] / 1000
# Percentage of total embodied emissions
total_emissions_percent = steel_and_cement_emissions['total_steel_and_cement_emissions_percent']

print(f"Total embodied emissions associated with steel and cement in wells in {check_case}: {total_emissions_tonnes:.2f} tonnes CO2")
print(f"Total embodied emissions associated with steel and cement in wells as a percentage of total embodied emissions in {check_case}: {total_emissions_percent:.2f}%")

# print(f"'Total embodied emissions associated with steel and cement in wells in", check_case,': '{calculate_steel_and_cement_emissions(check_case)['total_steel_and_cement_emissions']/1000} tonnes CO2")
# print(f"'Total embodied emissions associated with steel and cement in wells as a percentage of total embodied emissions in", check_case,': {calculate_steel_and_cement_emissions('Baseline')['total_steel_and_cement_emissions_percent']:.2f}%")



# In[264]:


# Define a function that calculates the percentage of VFF emissions relative to total emissions for a given case:


def calculate_emissions_percentages(case):
    # Calculate the total emissions for the given case
    total_emissions = np.array(calculate_total_emissions(case)['Total Emissions kgCO2e/day'])

    # Calculate the VFF emissions for the given case
    VFF_emissions = np.array(calculate_total_operational_VFF_emissions(case)['total_operational_VFF_emissions'])

    # Calculate total direct emissions for the given case
    total_direct_emissions = calculate_total_direct_emissions(case)['total_direct_emissions']

    # Handle cases where total emissions might be zero to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        VFF_emissions_percentage = np.where(total_emissions != 0, (VFF_emissions / total_emissions) * 100, 0)

    total_direct_emissions_percentage = np.where(total_emissions != 0, (total_direct_emissions / total_emissions) * 100, 0)

    return VFF_emissions_percentage, VFF_emissions, total_direct_emissions, total_direct_emissions_percentage

# Example usage
VFF_emissions_percentage, VFF_emissions, total_direct_emissions, total_direct_emissions_percentage = calculate_emissions_percentages('Baseline')

# Print the result
print(f"VFF emissions as a percentage of total emissions in the 'Baseline' case: {total_direct_emissions_percentage}")


# # 9. Cost Estimation

# ## 9.1 Cost Assumptions

# In[265]:


# Establish key assumptions regarding development and operating costs for the hydrogen production facility. For the time being, do this both on a 'cost of materials' basis (i.e. cost allowance
# per mass of material used) and on a 'historical average cost' basis (i.e. CAPEX & OPEX per ft of well depth).git

# Define the cost of materials for the hydrogen production facility
cost_of_steel_per_lb = 1.5 # USD/lb. DUMMY VALUE ONLY!!! This is the cost of steel per kg.
cost_of_cement_per_volume = 0.1 # USD/ft^3. This is the cost of cement per kg. 

# Define the historical average costs for the hydrogen production facility
cost_per_ft_drilling = 1000 # USD/ft. This is the cost per foot of drilling.
cost_per_ft_completion = 50 # USD/ft. This is the cost per foot of completion.
cost_per_ft_production = 25 # USD/ft. This is the cost per foot of production.



# ## 9.2 Cost Estimate Calculations

# In[266]:


# Estimate the total cost on a 'cost of materials' basis by case and sensitivity values:
def calculate_cost_materials_basis(case,sensitivity_variables=None):
    # if sensitivity_variables:
    #     cost_of_steel_per_lb = sensitivity_variables.get('cost_of_steel_per_lb', cost_of_steel_per_lb)
    #     cost_of_cement_per_volume = sensitivity_variables.get('cost_of_cement_per_volume', cost_of_cement_per_volume)
    # else:
    #     cost_of_steel_per_lb = cost_of_steel_per_lb
    #     cost_of_cement_per_volume = cost_of_cement_per_volume

    total_steel_cost_materials_basis = calculate_total_steel_mass(case, sensitivity_variables)['total_steel_mass'] * cost_of_steel_per_lb
    total_cement_cost_materials_basis = calculate_cement_volume_mass(case,sensitivity_variables)['total_well_cement_volume'] * cost_of_cement_per_volume

    total_cost_materials_basis = total_steel_cost_materials_basis + total_cement_cost_materials_basis

    return {
        'case': case,
        'total_steel_cost_materials_basis': total_steel_cost_materials_basis,
        'total_cement_cost_materials_basis': total_cement_cost_materials_basis,
        'total_cost_materials_basis': total_cost_materials_basis
    }

# Estimate the total cost on a 'historical average cost' basis by case and sensitivity values:

def calculate_cost_historical_basis(case,sensitivity_variables=None):
    # if sensitivity_variables:
    #     cost_per_ft_drilling = sensitivity_variables.get('cost_per_ft_drilling', cost_per_ft_drilling)
    #     cost_per_ft_completion = sensitivity_variables.get('cost_per_ft_completion', cost_per_ft_completion)
    #     cost_per_ft_production = sensitivity_variables.get('cost_per_ft_production', cost_per_ft_production)
    # else:
    #     cost_per_ft_drilling = cost_per_ft_drilling
    #     cost_per_ft_completion = cost_per_ft_completion
    #     cost_per_ft_production = cost_per_ft_production

    total_drilling_cost_historical_basis = depths[case]* cost_per_ft_drilling
    total_surface_equipment_cost_historical_basis = 0 #Need to determine a valid method for estimating this cost.

    total_cost_historical_basis = total_drilling_cost_historical_basis + total_surface_equipment_cost_historical_basis

    return {
        'case': case,
        'total_drilling_cost_historical_basis': total_drilling_cost_historical_basis,
        'total_surface_equipment_cost_historical_basis': total_surface_equipment_cost_historical_basis,
        'total_cost_historical_basis': total_cost_historical_basis
    }


# In[267]:


# Example usage:
cost_materials_basis = calculate_cost_materials_basis('Baseline')['total_cost_materials_basis']
cost_historical_basis = calculate_cost_historical_basis('Baseline')['total_cost_historical_basis']

print('Estimated total cost on a "cost of materials" basis for the Baseline case: $' , cost_materials_basis)
print('Estimated total cost on a "historical average cost" basis for the Baseline case: $' , cost_historical_basis)


# # 10. Production Tax Credit (PTC) Estimates

# # 11. Approximate Baseline Case as Exponential Decay
# 
# In order to investigate the impact of the assumption of field life, it will be helpful to define a case that approximates the assumed, empirical flow profile of the Baseline case with an idealised case based on exponential decay. The idealised case will match the initial flowrate, final flowrate and 30-year field life. 
# 
# This will then enable assessment of shorter and longer field life assumptions.

# In[268]:


#Extract the Baseline case flow rate at year 1:
Baseline_year_1_flow_rate = production_profile_df.loc[0, 'Baseline Raw Gas Rate, MSCFD']
print(Baseline_year_1_flow_rate)
Baseline_year_30_flow_rate = production_profile_df.loc[29, 'Baseline Raw Gas Rate, MSCFD']
print(Baseline_year_30_flow_rate)


# In[269]:


Q1 = Baseline_year_1_flow_rate  # Flow rate at year 1
Q2 = Baseline_year_30_flow_rate    # Flow rate at year 30
t1 = 1      # Year 1
t2 = 30     # Year 10

# Calculate decay constant k
k = np.log(Q1 / Q2) / (t2 - t1)

# Calculate initial quantity Q0
Q0 = Q1 / np.exp(-k * t1)

# Define the exponential decay function
def decay_curve(t, Q0, k):
    return Q0 * np.exp(-k * t)

# Generate data for plotting
years = np.linspace(1, 30, 100)
flow_rates = decay_curve(years, Q0, k)

# Plot the decay curve
plt.figure(figsize=(10, 6))
plt.plot(years, flow_rates, label='Exponential Decay Curve')
plt.scatter([t1, t2], [Q1, Q2], color='red', zorder=5, label='Given Points')
plt.title('Exponential Decay Curve')
plt.xlabel('Year')
plt.ylabel('Flow Rate')
plt.legend()
plt.grid(True)
plt.show()

# Print the decay constant and initial quantity
print(f"Decay constant (k): {k}")
print(f"Initial quantity (Q0): {Q0}")


# In[270]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the decay curve function
def decay_curve(t, Q0, k):
    return Q0 * np.exp(-k * t)

# Extract the production profile for the baseline case
baseline_production_profile = production_profile_df['Baseline Raw Gas Rate, MSCFD'].values

# Create the time vector (assuming one year intervals)
years = np.arange(1, len(baseline_production_profile) + 1)

# Define the objective function to minimize the sum of squared errors
def exponential_decay_objective(params, t, data):
    Q0, k = params
    model = decay_curve(t, Q0, k)
    return np.sum((model - data) ** 2)

# Define the constraint to ensure the sum of the production rates equals the desired total
desired_sum = 181657.174829267
def constraint(params, t):
    Q0, k = params
    model = decay_curve(t, Q0, k)
    return np.sum(model) - desired_sum

# Perform optimization to find the best parameters with the constraint
initial_guess = [10000, 0.1]  # Initial guess for Q0 and k
constraints = {'type': 'eq', 'fun': constraint, 'args': (years,)}
optimal_params = minimize(exponential_decay_objective, initial_guess, args=(years, baseline_production_profile), constraints=constraints).x

# Calculate the optimal Q0 and k values
optimal_Q0, optimal_k = optimal_params

# Generate the exponential decay curve using the optimal parameters
fitted_curve = decay_curve(years, optimal_Q0, optimal_k)

# Plot the fitted curve and the original data
plt.figure(figsize=(10, 6))
plt.plot(years, fitted_curve, label='Fitted Exponential Decay Curve')
plt.scatter(years, baseline_production_profile, color='red', zorder=5, label='Baseline Production Profile')
plt.title('Fitted Exponential Decay Curve for Baseline Production Profile')
plt.xlabel('Year')
plt.ylabel('Flow Rate (MSCFD)')
plt.legend()
plt.grid(True)
plt.show()

# Print the optimal parameters
print(f"Optimal Q0: {optimal_Q0}")
print(f"Optimal k: {optimal_k}")


# In[271]:


field_life_assumption = 30  #years

# Generate a series of flow rates (one for each year of field life) using the exponential decay formula fit above

# Define the time vector for the field life
years = np.arange(1, field_life_assumption + 1)

# Calculate the flow rates for each year using the fitted exponential decay curve
fitted_production_profile = decay_curve(years, optimal_Q0, optimal_k)

# Create a DataFrame to store the fitted production profile
fitted_production_profile_df = pd.DataFrame({
    'Year': years,
    'Fitted Production Profile, MSCFD': fitted_production_profile
})
# Set the 'Year' column as the index
fitted_production_profile_df.set_index('Year', inplace=True)
# Display the fitted production profile
fitted_production_profile_df

# Calculate the sum of the fitted production profile over the field life
total_production_fitted = fitted_production_profile.sum() * 365
total_production_fitted

