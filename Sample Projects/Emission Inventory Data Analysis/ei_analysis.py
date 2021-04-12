#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:13:15 2021

@author: mhaskariyeh
"""
# =============================================================================
# ## Initializing
# =============================================================================

# Importing packages
import pandas as pd
import numpy as np
import os
import glob 
import matplotlib.pyplot as plt
from sys import exit
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Reading input sets
df_area = pd.read_csv('def_area.csv')
df_fuel = pd.read_csv('def_fuel.csv')
df_veh = pd.read_csv('def_veh_category.csv')

## Reading emission factors in emfac_data folder 
path = 'Please update this line!'

# Read all emfac files
mylist = []

for file in glob.glob(f'{path}/emfac_data/*.csv'):
    print(file) # to show which file is being read
    temp= pd.read_csv(file, header = 5, nrows = 5)
    
    # Some of the emfac files have 5 lines above header, others have 6
    if temp.columns[0] == 'Region':
        tryheader = 5
    else:
        tryheader = 6

    # loads csv using large data method
    for chunk in  pd.read_csv(file,
                              sep = ',',
                              low_memory = False,
                              header = tryheader,
                              chunksize = 20000):
        mylist.append(chunk)

df_em = pd.concat(mylist, axis= 0)
del mylist
# exit()

# Know the data
# =============================================================================
# # know the df_area data
# print('df_area info:')
# print(df_area.head(10))
# print(df_area.shape)
# print(df_area.columns)
# print(df_area.describe())
# print(df_area.info())
# print('nan values:', df_area.isna().sum())
# 
# # know the df_fuel data
# print('df_fuel info:')
# print(df_fuel.head(10))
# print(df_fuel.shape)
# print(df_fuel.columns)
# print(df_fuel.describe())
# print(df_fuel.info())
# print('nan values:', df_fuel.isna().sum())
# 
# # know the df_veh data
# print('df_veh info:')
# print(df_veh.head(10))
# print(df_veh.shape)
# print(df_veh.columns)
# print(df_veh.describe())
# print(df_veh.info())
# print('nan values:', df_veh.isna().sum())
# 
# # know the df_em data
# print('df_em info:')
# print(df_em.head(10))
# print(df_em.shape)
# print(df_em.columns)
# print(df_em.describe())
# print(df_em.info())
# print('nan values:', df_veh.isna().sum())
# print('vehicle categories:', df_em['Vehicle Category'].unique())
# =============================================================================

# Merge emission data and def_area (for Q1 and Q3)
df_em = df_em.merge(df_area, left_on = 'Region', right_on = 'sub_area')

# Merge emission data and def_veh_category (for Q3c)
df_em = df_em.merge(df_veh, left_on = 'Vehicle Category', right_on = 'EMFAC2011_Vehicle_Category')

# Calculate the age- round up the calculated values (for Q2)
df_em['Age'] = df_em['Calendar Year']- df_em['Model Year'] + 1

# print('Age:', np.sort(df_em['Age'].unique()))

# Calculate the Mileage Accrual Rate (for Q2)
# Calculate miles per year per vehicle using VMT per day (times 365) and population
try:
    df_em['Mileage Accrual Rate'] = df_em['VMT'] * 365 /df_em['Population']
except:
    df_em['Mileage Accrual Rate'] = 0

# To conver inf and nan 'Mileage Accrual Rate' values to zero
df_em.loc[df_em['Population'] == 0,'Mileage Accrual Rate'] = 0

# Check mileage accrual rate
# df_em['Mileage Accrual Rate'].describe()
    
# Calculate NOxTotal in g/mile using NOx_TOTEX in ton per day and VMT per day (for Q4)
df_em['Tot_NOx_per_VMT'] = df_em['NOx_TOTEX']* 10**6 /df_em['VMT']
# Remove inf values because of some zero values for VMT in years 2025 and 2030  
df_em['Tot_NOx_per_VMT'].replace([np.inf, -np.inf], 0, inplace = True)

# Check Tot_NOx_per_VMT
# df_em['Tot_NOx_per_VMT'].describe()
# exit()

# =============================================================================
# ## Question1-a: How does electric light-duty vehicle (LDA, LDT1, LDT2, and MDV)
# ##  population change statewide from 2020 to 2050?
# =============================================================================

# Filter the data for Electric and LDVs
ldv_list = ['LDA','LDT1','LDT2','MDV']
df_em_eldv = df_em[(df_em['Vehicle Category'].isin(ldv_list)) & (df_em['Fuel'] == 'ELEC')]

# Classify the data by 'Calendar Year' and 'Vehicle Category'
df_em_eldv_by_year = df_em_eldv.groupby(['Calendar Year','Vehicle Category'])['Population'].sum()
df_em_eldv_by_year = df_em_eldv_by_year.reset_index()

# Plot a bar chart to show the populartion of eldv bay year and vehicle category
fig = px.bar(df_em_eldv_by_year,
              x = "Calendar Year",
              y = 'Population',
              color = 'Vehicle Category',
              title ='The Statewide Electric Light-Duty Vehicle Population (2020 to 2050)')
fig.update_layout(hovermode = 'x unified',
                  title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q1a.html')

# =============================================================================
# ## Question1-b-1: County with highest number of electric light duty vehicles
# ## in 2020 
# =============================================================================

# Filter the eldv data for year 2020 
year = 2020
df_em_eldv_2020 = df_em_eldv[df_em_eldv['Calendar Year'] == year]

# Classify and calculate the SUM of eldv Population based on County  
df_em_eldv_county_pop_2020 = df_em_eldv_2020.groupby('county')['Population'].sum()

# Select the county with max eldv in 2020
countyname_highest_eldv_2020 = df_em_eldv_county_pop_2020.idxmax()
pop_highest_eldv_2020 = df_em_eldv_county_pop_2020.max()
print('County with highest number of electric light duty vehicles in 2020: ',
      countyname_highest_eldv_2020,
      'with ',
      round(pop_highest_eldv_2020,1),
      'ELDV')

# =============================================================================
# ## Question1-b-2: County with highest number of electric light duty vehicles
# ## in 2050
# =============================================================================

# Filter the eldv data for year 2050 
year = 2050
df_em_eldv_2050 =df_em_eldv[df_em_eldv['Calendar Year'] == year]

# Classify and calculate the SUM of eldv Population based on County 
df_em_eldv_county_pop_2050= df_em_eldv_2050.groupby(['county'])['Population'].sum()

# Select the county with max eldv in 2050
countyname_highest_eldv_2050 = df_em_eldv_county_pop_2050.idxmax()
pop_highest_eldv_2050 = df_em_eldv_county_pop_2050.max()
print('County with highest number of electric light duty vehicles in 2050: ',
      countyname_highest_eldv_2050,
      'with ',
      round(pop_highest_eldv_2050,1),
      'ELDV')

# =============================================================================
# ## Question 2: The mileage accrual rate distribution of gasoline light-duty 
# ##             vehicles statewide by age in 2020
# =============================================================================

# Filter the data to get Gasoline ldv data for year 2020 
year = 2020
ldv_list = ['LDA','LDT1','LDT2','MDV']
df_em_gldv = df_em[(df_em['Calendar Year'] == year) 
                  & (df_em['Vehicle Category'].isin(ldv_list)) 
                  & (df_em['Fuel'] == 'GAS')]

# Classify and calculate the SUM of gldv VMT by age
df_em_gldv_by_year = df_em_gldv.groupby(['Age','Vehicle Category'])['VMT'].sum()
df_em_gldv_by_year = df_em_gldv_by_year.reset_index()
fig = px.bar(df_em_gldv_by_year,
              x = 'Age',
              y = 'VMT',
              color = 'Vehicle Category',
              title ='Miles Gasoline Light-Duty Vhicles Travel Statewide per Day by Age (2020)',
              labels = dict(VMT = "Vehicles Miles Traveled per Day",
                            Age = 'Age (Year)'))
fig.update_layout(hovermode = 'x unified',
                  title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q2_1.html')


# Calculate the average of gldv Mileage Accrual Rate (mar) weighted by population 
wm = lambda x: np.average(x, weights = df_em_gldv.loc[x.index, "Population"])
df_em_gldv_avg_mar = df_em_gldv.groupby(['Age']).agg(weighted_mar=("Mileage Accrual Rate", wm))
df_em_gldv_avg_mar = df_em_gldv_avg_mar.reset_index()
fig = px.bar(df_em_gldv_avg_mar,
              x = 'Age',
              y = 'weighted_mar',
              title ='The Statewide Average Mileage Accrual Rate of Gasoline Light-Duty Vhicles (2020)',
              labels = dict(weighted_mar = "Mileage Accrual Rate (mile/year/vehicle)",
                            Age = 'Age (Year)'))
fig.update_layout(hovermode = 'x unified',
                  title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q2_2.html')

# =============================================================================
# ## Question 3a: Which air basin is the largest contributor of 
# ##       diesel-fueled medium heavy-duty truck (MHDT) NOx emissions in 2030?
# =============================================================================

# Filter year 2030
year = 2030 

# Filter the data to just include mhdt (begin with T6)
mhdt_list = [x for x in df_em['Vehicle Category'] if x.startswith('T6')]
df_em_yr_mhdt = df_em[(df_em['Vehicle Category'].isin(mhdt_list))
                      & (df_em['Calendar Year'] == year)
                      & (df_em['Fuel'] == 'DSL')]

# Classify the df_em_yr_mhdt baed on air basin name 
df_em_yr_mhdt_nox_airbasin = df_em_yr_mhdt.groupby(['airbasin'])['NOx_TOTEX'].sum()
airbasiname_highest2030_mhdt_nox = df_em_yr_mhdt_nox_airbasin.idxmax() 
tonperday_highest2030_mhdt_nox = df_em_yr_mhdt_nox_airbasin.max() 
print('The largest contributor (Air Basin) of diesel-fueled medium heavy-duty truck (MHDT) NOx emissions in 2030: ',
      airbasiname_highest2030_mhdt_nox,
      'with ',
      round(tonperday_highest2030_mhdt_nox,1),
      'Tons per Day')

# =============================================================================
# ## Question 3b: How does total NOx emissions in this air basin change  
# ##              from 2020 through 2030?
# =============================================================================

# Filter the data for each year and the air basin with highest NOx from mhdt in 2030
df_em_yr_airbasin_highest2030 = df_em[(df_em['airbasin'] == airbasiname_highest2030_mhdt_nox)
                                      & (df_em['Calendar Year'] <= 2030)]

# Calculate the sum of nox emissions in the air basin with highest NOx from mhdt in 2030
df_em_yr_airbasin_highest2030_tot_nox = df_em_yr_airbasin_highest2030.groupby(['Calendar Year'])['NOx_TOTEX'].sum()
df_em_yr_airbasin_highest2030_tot_nox = df_em_yr_airbasin_highest2030_tot_nox.reset_index()

fig = px.bar(df_em_yr_airbasin_highest2030_tot_nox,
              x = 'Calendar Year',
              y = 'NOx_TOTEX',
              title =' The Total NOx Emissions in South Coast Air Basin (2020 to 2030)',
              labels = dict(NOx_TOTEX = 'NOx Emission (ton/day)'))
fig.update_layout(hovermode = 'x unified',
                  title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q3b_1.html')

# =============================================================================
# 
# =============================================================================

# Calculate the sum of nox emissions in the air basin with highest NOx from mhdt in 2030
df_em_yr_airbasin_highest2030_nox = df_em_yr_airbasin_highest2030.groupby(['Calendar Year','Vehicle Category'])['NOx_TOTEX'].sum()
df_em_yr_airbasin_highest2030_nox = df_em_yr_airbasin_highest2030_nox.reset_index()

fig = px.bar(df_em_yr_airbasin_highest2030_nox,
              x = 'Calendar Year',
              y = 'NOx_TOTEX',
              color = 'Vehicle Category',
              title =' The Total NOx Emissions in South Coast Air Basin (2020 to 2030)',
              labels = dict(NOx_TOTEX = 'NOx Emission (ton/day)'))
fig.update_layout(title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q3b_2.html')

# =============================================================================
# ## Question 3c_1: What are total NOx emissions from this air basin broken out    
# ## by EMFAC2007 vehicle class in 2020 and 2030?
# =============================================================================

# Calculate the sum of nox emissions in the air basin with highest NOx from mhdt in 2030 by EMFAC2007 vehicle class
df_em_yr_airbasin_highest2030_nox = df_em_yr_airbasin_highest2030.groupby(['Calendar Year','EMFAC2007_Vehicle'])['NOx_TOTEX'].sum()
df_em_yr_airbasin_highest2030_nox = df_em_yr_airbasin_highest2030_nox.reset_index()

fig = px.bar(df_em_yr_airbasin_highest2030_nox,
              x = 'Calendar Year',
              y = 'NOx_TOTEX',
              color = 'EMFAC2007_Vehicle',
              title='The Total NOx Emissions in South Coast Air Basin 2020 to 2030 (by EMFAC2007 vehicle class)',
              labels = dict(NOx_TOTEX = 'NOx Emission (ton/day)',
                            EMFAC2007_Vehicle= 'EMFAC2007<br>Vehicle Category'))
fig.update_layout(title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q3c_1.html')

# =============================================================================
# ## Question 3c_2: What are total NOx emissions from this air basin broken out    
# ## by EMFAC2007 vehicle class in 2020 and 2030?
# =============================================================================

# Filter the data for year 2020 and the air basin with highest NOx from mhdt in 2030
year= 2020
df_em_yr2020_airbasin_highest2030= df_em[(df_em['airbasin'] == airbasiname_highest2030_mhdt_nox) & (df_em['Calendar Year'] == year)]
# Classify the dataset by EMFAC2007 vehicle classes
df_em_yr2020_airbasin_highest2030_EMFAC2007= df_em_yr2020_airbasin_highest2030.groupby(['EMFAC2007_Vehicle'])['NOx_TOTEX'].sum()
df_em_yr2020_airbasin_highest2030_EMFAC2007 = df_em_yr2020_airbasin_highest2030_EMFAC2007.reset_index()
df_em_yr2020_airbasin_highest2030_EMFAC2007['Year'] = 2020

year= 2030
df_em_yr2030_airbasin_highest2030 = df_em[(df_em['airbasin'] == airbasiname_highest2030_mhdt_nox) & (df_em['Calendar Year'] == year)]
# Classify the dataset by EMFAC2007 vehicle classes
df_em_yr2030_airbasin_highest2030_EMFAC2007 = df_em_yr2030_airbasin_highest2030.groupby(['EMFAC2007_Vehicle'])['NOx_TOTEX'].sum()
df_em_yr2030_airbasin_highest2030_EMFAC2007 = df_em_yr2030_airbasin_highest2030_EMFAC2007.reset_index()
df_em_yr2030_airbasin_highest2030_EMFAC2007['Year'] = 2030

southcoast_nox_20and30 = pd.concat([df_em_yr2020_airbasin_highest2030_EMFAC2007,
                                    df_em_yr2030_airbasin_highest2030_EMFAC2007])

southcoast_nox_20and30['Year'] = southcoast_nox_20and30['Year'].astype(str)

fig = px.bar(southcoast_nox_20and30,
             x = 'EMFAC2007_Vehicle',
             y = 'NOx_TOTEX',
             color = 'Year',
             barmode = 'group',
             title = 'The Total NOx Emissions in South Coast Air Basin by EMFAC2007 Vehicle Class (2020 and 2030)',
             labels = dict(NOx_TOTEX = 'NOx Emission (ton/day)',
                            EMFAC2007_Vehicle_y= 'EMFAC2007 Vehicle Category',
                            Year = 'Calendar Year'))
fig.update_layout(title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q3c_2.html')

# =============================================================================
# ## Question 4a: Calculate the statewide fleet ave NOx emission rate of (HDDT) from 2020 to 2040 
# =============================================================================

# Filter the data to just include hhdt (Vehicle Category begins with T7)
hhdt_list = [x for x in df_em['Vehicle Category'] if x.startswith('T7')]
df_em_hhdt_20to40 = df_em[(df_em['Vehicle Category'].isin(hhdt_list))
                   & (df_em['Calendar Year'] <= 2040)]

# Filter to exclude zero values for VMT (VMT for 2025 and 2030 years is zero)
df_em_hhdt_20to40 = df_em_hhdt_20to40[df_em_hhdt_20to40['VMT'] != 0]

# Weight the Vehicle Category NOx emission factor by VMT to obtain the state average emission factor 
wm = lambda x: np.average(x, weights=df_em_hhdt_20to40.loc[x.index, "VMT"])
avg_nox_hhdt_state_bycat = df_em_hhdt_20to40.groupby(["Calendar Year", 'Vehicle Category']).agg(weighted_NOx=("Tot_NOx_per_VMT", wm))
avg_nox_hhdt_state_bycat = avg_nox_hhdt_state_bycat.reset_index()

# Plot a bar chart for weighted average hhdt 2020 to 2040
fig4 = px.bar(avg_nox_hhdt_state_bycat,
              x = 'Calendar Year',
              y = 'weighted_NOx',
              color = 'Vehicle Category',
              title='Average NOx Emission Rate of Heavy Heavy-Duty Trucks (HHDT)',
              labels = dict(weighted_NOx = 'Average NOx Emission Rate (g/mile)'))
fig4.update_layout(title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig4.write_html(f'{path}/Q4a_1.html') 

# =============================================================================
# 
# =============================================================================

# Weight the Vehicle Category NOx emission factor by VMT to obtain the state average emission factor 
wm = lambda x: np.average(x, weights=df_em_hhdt_20to40.loc[x.index, "VMT"])
avg_nox_hhdt_state = df_em_hhdt_20to40.groupby(["Calendar Year"]).agg(weighted_NOx=("Tot_NOx_per_VMT", wm))
avg_nox_hhdt_state = avg_nox_hhdt_state.reset_index()

# Plot a bar chart for weighted average hhdt 2020 to 2040
fig4_1 = px.bar(avg_nox_hhdt_state,
              x = 'Calendar Year',
              y = 'weighted_NOx',
              # color = 'Vehicle Category',
              title='Average NOx Emission Rate of Heavy Heavy-Duty Trucks (HHDT)',
              labels = dict(weighted_NOx = 'Average NOx Emission Rate (g/mile)'))
fig4_1.update_layout(hovermode = 'x unified',
                     title_x = 0.5,
                     title_y = 0.85,
                     title_font_size = 20)
fig4_1.write_html(f'{path}/Q4a_2.html') 

# =============================================================================
# # Q4a: With VMT 
# =============================================================================

# Filter the data to just include hhdt (begin with T7)
hhdt_list= [x for x in df_em['Vehicle Category'] if x.startswith('T7')]
df_em_hhdt= df_em[df_em['Vehicle Category'].isin(hhdt_list)]

# Calculate the total VMT per year
df_em_hhdt_yr = df_em_hhdt.groupby(['Calendar Year','Vehicle Category'])['VMT'].sum()
df_em_hhdt_yr = df_em_hhdt_yr.reset_index()

fig4_2 = px.bar(df_em_hhdt_yr,
              x = 'Calendar Year',
              y = 'VMT',
              color = 'Vehicle Category',
              title='The Total Vehicle Miles Traveled by Heavy Heavy-Duty Trucks (HHDT)',
              labels = dict(VMT = 'Vehicle Miles Traveled (mile)'))
fig4_2.update_layout(title_x = 0.5,
                     title_y = 0.85,
                     title_font_size = 20)
fig4_2.write_html(f'{path}/Q4a_vmt.html') 

# =============================================================================
# ## Question 4b: Which air district has the highest NOx emission factor in 2035 
# =============================================================================

# To obtain the air district that has the highest (HHDT) NOx emission factor in 2035
year= 2035 
df_em_hhdt_2035 = df_em_hhdt[(df_em_hhdt['Calendar Year'] == year) & (df_em_hhdt['VMT'] != 0)]

# Calculate the average NOx emission factor weighted by VMT 
wm = lambda x: np.average(x, weights=df_em_hhdt.loc[x.index, "VMT"])
avg_nox_hhdt_district_2035 = df_em_hhdt_2035.groupby(['district']).agg(weighted_NOx=("Tot_NOx_per_VMT", wm))
distrcitname_highest_nox_2035 = avg_nox_hhdt_district_2035.idxmax().to_list()[0]
highest_nox_2035_district_value = avg_nox_hhdt_district_2035.max()

print('The air basin with highest HHDT NOx emission factor in 2035: ',
      distrcitname_highest_nox_2035,
      'with ',
      round(highest_nox_2035_district_value,1),
      'g/mile')

# =============================================================================
# ## Question 5a: The statewide HHDT NOx emission change from 2020 through 2050 
# =============================================================================

# Filter the data to just include hhdt (begin with T7)
hhdt_list= [x for x in df_em['Vehicle Category'] if x.startswith('T7')]
df_em_hhdt= df_em[df_em['Vehicle Category'].isin(hhdt_list)]

# Classify and calculate the some of NOx emission by year and vehicle category
df_em_hhdt_yr = df_em_hhdt.groupby(['Calendar Year','Vehicle Category'])['NOx_TOTEX'].sum()
df_em_hhdt_yr = df_em_hhdt_yr.reset_index()

fig = px.bar(df_em_hhdt_yr,
              x = 'Calendar Year',
              y = 'NOx_TOTEX',
              color = 'Vehicle Category',
              title='The Total Statewide Heavy Heavy-Duty Trucks (HHDT) NOx Emissions (2020 to 2050)',
              labels = dict(NOx_TOTEX = 'NOx Emission (ton/day)'))
fig.update_layout(title_x = 0.5,
                     title_y = 0.85,
                     title_font_size = 20)
fig.write_html(f'{path}/Q5a.html')

# =============================================================================
# ## Question 5b: The statewide HHDT NOx emission change with the proposed regulation 
# =============================================================================

# Exclusion list (out-of-state trucks) 
imp_scenario_notlist=['T7 NNOOS','T7 NOOS']

# Reduction secnario: 
red_scenario= {2024: 0.25, 2025: 0.25, 2026: 0.5, 2027: 0.5, 2028: 0.5, 2029: 0}

# Application coefficient dictionary
app_coeff_scenario= {2024: 0.75,
                     2025: 0.75,
                     2026: 0.5,
                     2027: 0.5,
                     2028: 0.5}

# A function to apply application coefficient to total nox emission
def application_weight(year, vehicle_category, tot_nox):
    if vehicle_category in imp_scenario_notlist:
        return tot_nox
    elif year < 2024:
        return tot_nox
    else:
        try: 
            return app_coeff_scenario[year] * tot_nox
        except:
            return 0
 
# Apply scenario application coefficient to total nox and calculate the total 
# nox emission with the proposed regulation
df_em_hhdt['NOx_TOTEX_RedScenario'] = df_em_hhdt.apply(lambda row: application_weight(row['Model Year'],
                                                                                      row['Vehicle Category'],
                                                                                      row['NOx_TOTEX']),
                                                       axis = 1)

# Classify and calculate the some of NOx emission with with the proposed regulation by year and vehicle category
df_em_hhdt_yr = df_em_hhdt.groupby(['Calendar Year','Vehicle Category'])['NOx_TOTEX_RedScenario'].sum()
df_em_hhdt_yr = df_em_hhdt_yr.reset_index()

fig = px.bar(df_em_hhdt_yr,
              x = 'Calendar Year',
              y = 'NOx_TOTEX_RedScenario',
              color = 'Vehicle Category',
              title='The Total Statewide Heavy Heavy-Duty Trucks (HHDT) NOx Emissions with Proposed Regulation (2020 to 2050)',
              labels = dict(NOx_TOTEX_RedScenario = 'NOx Emission (ton/day)'))
fig.update_layout(title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q5b_1.html')

# =============================================================================
# ## Question 5b: Calculate and visualize the difference due to proposed regulation
# =============================================================================

# Calculate and visualize the difference between original dataset and the proposed regulation
df_em_hhdt['NOx_TOTEX_Reduction'] = df_em_hhdt['NOx_TOTEX'] - df_em_hhdt['NOx_TOTEX_RedScenario']
df_em_hhdt_yr = df_em_hhdt.groupby(['Calendar Year','Vehicle Category'])['NOx_TOTEX_Reduction'].sum()
df_em_hhdt_yr = df_em_hhdt_yr.reset_index()

fig = px.bar(df_em_hhdt_yr,
              x = 'Calendar Year',
              y = 'NOx_TOTEX_Reduction',
              color = 'Vehicle Category',
              title='The Total HHDT NOx Emissions Reduction Statewide due to Proposed Regulation (2020 to 2050)',
              labels = dict(NOx_TOTEX_Reduction = 'NOx Emission Reduction (ton/day)'))
fig.update_layout(title_x = 0.5,
                  title_y = 0.85,
                  title_font_size = 20)
fig.write_html(f'{path}/Q5b_2.html')


