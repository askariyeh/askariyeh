#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:05:55 2021

@author: mhaskariyeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_in = 'path'
in_file_name = 'traffic_inp.csv'

df = pd.read_csv(path_in + in_file_name)

# =============================================================================
# # Step 1:
# =============================================================================
    
# Examine and clean the data
df.head(5)
df.isnull()
df.isnull().sum()
df.shape

df.drop('county_name',
        axis = 'columns',
        inplace = True)

df.dtypes

df.dropna(subset=['stop_date', 'stop_time'],
          inplace = True)

# Examine the head of the 'is_arrested' column
print(df.is_arrested.head())

# Change the data type of 'is_arrested' to 'bool'
df['is_arrested'] = df.is_arrested.astype('bool')

# Check the data type of 'is_arrested' 
print(df.is_arrested.head())

# Concatenate 'stop_date' and 'stop_time' (separated by a space)
combined = df.stop_date.str.cat(df.stop_time, sep=' ')

# Convert 'combined' to datetime format
df['stop_datetime'] = pd.to_datetime(combined)

# Examine the data types of the DataFrame
print(df.dtypes)

# Set 'stop_datetime' as the index
df.set_index('stop_datetime', inplace=True)

# Examine the index
print(df.index)

# Examine the columns
print(df.columns)

# =============================================================================
# # Step 2:
# =============================================================================

  # Count the unique values in 'violation'
print(df.violation.value_counts())

# Express the counts as proportions
print(df.violation.value_counts(normalize = True))

# Create a DataFrame of female drivers
female = df[df.driver_gender == 'F']

# Create a DataFrame of male drivers
male = df[df.driver_gender == 'M']

# Compute the violations by female drivers (as proportions)
print(female.violation.value_counts(normalize = True))

# Compute the violations by male drivers (as proportions)
print(male.violation.value_counts(normalize = True))  

# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = df[(df.driver_gender == 'F') &
                         (df.violation == 'Speeding')]

# Create a DataFrame of male drivers stopped for speeding
male_and_speeding = df[(df.driver_gender == 'M') &
                         (df.violation == 'Speeding')]

# Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize = True))

# Compute the stop outcomes for male drivers (as proportions)
print(male_and_speeding.stop_outcome.value_counts(normalize = True))

# Check the data type of 'search_conducted'
print(df.search_conducted.dtype)

# Calculate the search rate by counting the values
print(df.search_conducted.value_counts(normalize=True))

# Calculate the search rate by taking the mean
print(df.search_conducted.mean())

# Calculate the search rate for female drivers
print(df[df.driver_gender == 'F'].search_conducted.mean())

# Calculate the search rate for each combination of gender and violation
print(df.groupby(['driver_gender','violation']).search_conducted.mean())

# Count the 'search_type' values
print(df.search_type.value_counts())

# Check if 'search_type' contains the string 'Protective Frisk'
df['frisk'] = df.search_type.str.contains('Protective Frisk', na=False)

# Check the data type of 'frisk'
print(df.frisk.dtype)

# Take the sum of 'frisk'
print(df.frisk.sum())

# Create a DataFrame of stops in which a search was conducted
searched = df[df.search_conducted == True]

# Calculate the overall frisk rate by taking the mean of 'frisk'
print(searched.frisk.mean())

# Calculate the frisk rate for each gender
print(searched.groupby('driver_gender').frisk.mean())

# =============================================================================
# # Step 3:
# =============================================================================

# Calculate the overall arrest rate
print(df.is_arrested.mean())

# Calculate the hourly arrest rate
print(df.groupby(df.index.hour).is_arrested.mean())

# Save the hourly arrest rate
hourly_arrest_rate = df.groupby(df.index.hour).is_arrested.mean()

# Create a line plot of 'hourly_arrest_rate'
plt.plot(hourly_arrest_rate)

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()

# Calculate the annual rate of drug-related stops
print(df.drugs_related_stop.resample('A').mean())

# Save the annual rate of drug-related stops
annual_drug_rate = df.drugs_related_stop.resample('A').mean()

# Create a line plot of 'annual_drug_rate'
plt.plot(annual_drug_rate)

# Display the plot
plt.show()   
    
# Calculate and save the annual search rate
annual_search_rate = df.search_conducted.resample('A').mean()

# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_rate,annual_search_rate], axis='columns')

# Create subplots from 'annual'
annual.plot(subplots= True)

# Display the subplots
plt.show()

# Create a frequency table of districts and violations
print(pd.crosstab(df.district, df.violation))

# Save the frequency table as 'all_zones'
all_zones = pd.crosstab(df.district, df.violation)

# Select rows 'Zone K1' through 'Zone K3'
print(all_zones.loc['Zone K1':'Zone K3'])

# Save the smaller table as 'k_zones'
k_zones = all_zones.loc['Zone K1':'Zone K3']

# Create a bar plot of 'k_zones'
k_zones.plot(kind='bar')

# Display the plot
plt.show()

# Print the unique values in 'stop_duration'
print(df.stop_duration.unique())

# Create a dictionary that maps strings to integers
mapping = {'0-15 Min': 8, '16-30 Min': 23, '30+ Min': 45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
df['stop_minutes'] = df.stop_duration.map(mapping)

# Print the unique values in 'stop_minutes'
print(df.stop_minutes.unique())

# Calculate the mean 'stop_minutes' for each value in 'violation_raw'
print(df.groupby('violation_raw').stop_minutes.mean())

# Save the resulting Series as 'stop_length'
stop_length = df.groupby('violation_raw').stop_minutes.mean()

# Sort 'stop_length' by its values and create a horizontal bar plot
stop_length.sort_values().plot(kind= 'barh')

# Display the plot
plt.show()

# =============================================================================
# # Step 4
# =============================================================================

# Read 'weather.csv' into a DataFrame named 'weather'
filename1 = 'weather_inp.csv'
weather = pd.read_csv(path_in + in_file_name)

# Describe the temperature columns
print(weather[['TMIN','TAVG','TMAX']].describe())

# Create a box plot of the temperature columns
weather[['TMIN','TAVG','TMAX']].plot(kind='box')

# Display the plot
plt.show()

# Create a 'TDIFF' column that represents temperature difference
weather['TDIFF'] = weather.TMAX - weather.TMIN 

# Describe the 'TDIFF' column
print(weather.TDIFF.describe())

# Create a histogram with 20 bins to visualize 'TDIFF'
weather.TDIFF.plot(kind='hist', bins = 20)

# Display the plot
plt.show()

# Copy 'WT01' through 'WT22' to a new DataFrame
WT = weather.loc[:, 'WT01':'WT22']

# Calculate the sum of each row in 'WT'
weather['bad_conditions'] = WT.sum(axis= 'columns')

# Replace missing values in 'bad_conditions' with '0'
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

# Create a histogram to visualize 'bad_conditions'
weather.bad_conditions.plot(kind='hist')

# Display the plot
plt.show()

# Count the unique values in 'bad_conditions' and sort the index
print(weather.bad_conditions.value_counts().sort_index())

# Create a dictionary that maps integers to strings
mapping = {0:'good', 1:'bad', 2:'bad', 3:'bad', 4:'bad', 5:'worse',6:'worse', 7:'worse', 8:'worse', 9:'worse'}

# Convert the 'bad_conditions' integers to strings using the 'mapping'
weather['rating'] = weather.bad_conditions.map(mapping)

# Count the unique values in 'rating'
print(weather.rating.value_counts())

# Create a list of weather ratings in logical order
cats = ['good', 'bad', 'worse']

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype('category', ordered=True,categories=cats)

# Examine the head of 'rating'
print(weather.rating.head())

# Reset the index of 'df'
df.reset_index(inplace=True)

# Examine the head of 'df'
print(df.head())

# Create a DataFrame from the 'DATE' and 'rating' columns
weather_rating = weather[['DATE', 'rating']]

# Examine the head of 'weather_rating'
print(weather_rating.head())

# Examine the shape of 'df'
print(df.shape)

# Merge 'df' and 'weather_rating' using a left join
df_weather = pd.merge(left=df, right=weather_rating, left_on='stop_date', right_on='DATE', how='left')

# Examine the shape of 'df_weather'
print(df_weather.shape)

# Set 'stop_datetime' as the index of 'df_weather'
df_weather.set_index('stop_datetime', inplace=True)

# Calculate the overall arrest rate
print(df_weather.is_arrested.mean())

# Save the output of the groupby operation from the last exercise
arrest_rate = df_weather.groupby(['violation', 'rating']).is_arrested.mean()

# Print the 'arrest_rate' Series
print(arrest_rate)

# Print the arrest rate for moving violations in bad weather
print(arrest_rate.loc['Moving violation', 'bad'])

# Print the arrest rates for speeding violations in all three weather conditions
print(arrest_rate.loc['Speeding violation'])

# Unstack the 'arrest_rate' Series into a DataFrame
print(arrest_rate.unstack())

# Create the same DataFrame using a pivot table
print(df_weather.pivot_table(index='violation', columns='rating', values='is_arrested'))




