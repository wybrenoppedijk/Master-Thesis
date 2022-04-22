# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 07:43:53 2021

@author: dgeco
"""

import pandas as pd
import os

from datetime import datetime
import datetime as dt
from meteostat import Point, Hourly, Daily
import math

import matplotlib.pyplot as plt

# Convert comma to dot, for python to correctly register values. Done, apparently being done automatically in read_excel.
# Variability in water height translates to poor close-contact tracking of total water volume in tank, but good over a given range. Maybe average over a couple of measuring samples!
# 3- or 7- moving average 
# advanced idea: transform all df to one second entries? 4x more data tho...

# Might need some workaround if using even large files: https://stackoverflow.com/questions/11622652/large-persistent-dataframe-in-pandas

# Creating a path for the datasheet. Modify according to desired entry.
# df_path = os.path.join("C:/DTU Action/Bornholm/PST239_Oktober", "v3_PST239_Oktober_Short_DateTime.xlsx")

# df_path = os.path.join("C:/DTU Action/Bornholm/PST239_Februar", "PST239_Februar_DateTime.xlsx")
# df_path = os.path.join("C:/DTU Action/Bornholm/PST239_Juni", "PST239_Juni_DateTime.xlsx")
# df_path = os.path.join("C:/DTU Action/Bornholm/PST239_August", "PST239_August_DateTime.xlsx")
# df_path = os.path.join("C:/DTU Action/Bornholm/PST239_Juli", "PST239_Juli_DateTime.xlsx")
# df_path = os.path.join("C:/DTU Action/Bornholm/PST239_Oktober", "v3_PST239_Oktober_DateTime.xlsx")

Select_month = "Oktober"
df_path = os.path.join("/Users/wybrenoppedijk/Documents/DTU/master thesis/code/daniel/data/", "PST239_" + Select_month + "_DateTime.xlsx")



# Reads the csv or excel, and converts to dataframe, converting decimal separator from comma to dot.
# test_df = pd.read_csv(df_path, sep=";", decimal=",") # No longer used because xlsx reads automatically.  
test_df = pd.read_excel(df_path) 


# Original CSV files, presumably output of SCADA system, do not have a good format of time stamps, recognized by Excel but not Pandas.
# A workaround for initial data analysis was using Excel saving formatted date-time after processing CSV. Executed outside of Python!
test_df.set_index(pd.to_datetime(test_df['Time'], infer_datetime_format=True), inplace=True)#, utc=True))#, drop=True))
# test_df.set_index(pd.to_datetime(test_df['Time'], format= '%d-%m-%Y %H:%M:%S')) # Might be faster for bigger datasets.

#test_df.set_index('Time', inplace=True)

# print(test_df['Time'].iloc[20].minute)

def fetch_historic_weather(): # More details in ATTACHMENT 1
    # Set time period
    start_fetch = datetime(2020, 10, 1, 0, 0)
    end_fetch = datetime(2020, 10, 31, 23, 59)
    start_fetch_day = datetime(2020, 10, 4, 0, 0)
    end_fetch_day = datetime(2020, 10, 4, 23, 59)
    
    # Create Point for Alling, DK
    Allinge_coord = Point(55.274, 14.8)
     
    # Get hourly data for 2020
    weather_data_hour = Hourly(Allinge_coord, start_fetch_day, end_fetch_day)
    weather_data_hour = weather_data_hour.normalize()                         # Ensures there is one data point per hour
    weather_data_hour = weather_data_hour.fetch()                             # Fetches data from given coordinates
    
    weather_data_day = Daily(Allinge_coord, start_fetch, end_fetch)
    weather_data_day = weather_data_day.normalize()                         # Ensures there is one data point per hour
    weather_data_day = weather_data_day.fetch()                             # Fetches data from given coordinates
    
    #print(weather_data_hour)
    return weather_data_hour, weather_data_day

# Weather_Data_Hour, Weather_Data_Day = fetch_historic_weather()
# print(Weather_Data_Day)

#test_df['prcp'] = 0

# test_df.info()

#print(weather_data.index.hour)
# print(weather_data.index)
#print(test_df.index.hour)

# for x in weather_data.index:
    # print(x.hour)
    
""" The following yields 0.0 in very first entry an NaN for other 'prcp' entries. Also, it take many minutes because it is a bad nest of fors. """
# for x in test_df.index:
#     for y in weather_data.index:
#         if (x.hour == y.hour and x.day == y.day):
#             test_df['prcp'] = weather_data['prcp'] 

#test_df['prcp'] = ['Bad' if x<7.000 else 'Good' if 7.000<=x<8.000 else 'Very Good' for x in test_df['Score']]
#test_df['prcp'] = [weather_data['prcp'] if (test_df.index.minute == weather_data.index.minute and test_df.index.hour == weather_data.index.hour and test_df.index.day == weather_data.index.day) for x in test_df.index]
""" Create a more pythonic way with a lookup table for the minute and hour or weather_data['prcp']"""

#test_df['RolAvg_Level_14'] = test_df.iloc[:,1].rolling(window=14).mean()
test_df['Volume_RolAvg_Vol_7'] = test_df.iloc[:,1].rolling(window=7).mean()*(1.502367*1.502367*3.141592)*1000

test_df['Converted_Liter_diff'] = (test_df['Level'].diff()) * (1.502367*1.502367*3.141592)*1000
test_df['CLd_RolAvg'] = test_df.iloc[:,6].rolling(window=7).mean()
test_df['Converted_Outflow'] = test_df.iloc[:,2].rolling(window=7).mean()*4*1000/3600 # 4x seconds per /4s; 1000x liters in m3; divided by 3600 seconds in an hour 

# test_df['CLd_RolAvg_5min'] = test_df.iloc[:,6].rolling(window=35).mean()


#Diferenca do rolling window eh diferente da rolling window da diferenca!
#test_df['Outflow_RolAvg_7'] = test_df.iloc[:,2].rolling(window=7).mean()*(1.502367*1.502367*3.141592)*1000 / (
# test_df['Inflow_Calculated'] = (test_df['Volume_RolAvg_Vol_7'].diff())/4 + test_df['Flow'] * 4 * 1000 / ( # 4x seconds per /4s; 1000x liters in m3; divided by 3600 seconds in an hour 
    # 3600 * test_df.index.to_series().diff().dt.total_seconds())
#test_df['Inflow_Calculated'] = (test_df['Volume_RolAvg_Vol_7'].diff()) + test_df['Converted_Outflow'] / ( 
#    test_df.index.to_series().diff().dt.total_seconds()) 
test_df['Inflow_Calculated'] = test_df['CLd_RolAvg'] + test_df['Converted_Outflow'] 
# test_df['Inflow_Calculated_5min'] = test_df['CLd_RolAvg_5min'] + test_df['Converted_Outflow'] 
#
## We want to plot in per four seconds!!!
    
# test_df['Time_diff'] = test_df.index.to_series().diff().dt.total_seconds()
#print(test_df['Volume_RolAvg_Vol_7'])



# Plotting water volume, flows, and pumps

print(test_df.head(n=10).to_string(index=False))

print("Daily average energy consumption (kWh):")
print("Total monthly energy consumption (kWh):")
print("Daily average water inflow (liters):")
print("Total monthly water inflow (liters):")

fig, ax1 = plt.subplots(1,1)
test_df.Volume_RolAvg_Vol_7.plot(ax=ax1, color='blue', label='Tank volume')
ax2 = ax1.twinx()
test_df.CurrentP1.plot(ax=ax2, color='magenta', label='Pump 1')
test_df.CurrentP2.plot(ax=ax2, color='k', label='Pump 2')
test_df.Flow.plot(ax=ax2, color='c', label='Outflow')
test_df.Inflow_Calculated.plot(ax=ax2, color='purple', label='Inflow')
# test_df.Inflow_Calculated_5min.plot(ax=ax2, color='red', label='In Flow slow')
ax1.set_ylabel('Tank volume [liters]')
ax2.set_ylabel(r'Pump Current [A] or flow [$m^{3}/s$]')
ax1.axhline(y=2836, color='magenta', linestyle=':')
ax1.axhline(y=7800, color='magenta', linestyle=':')
ax1.axhline(y=10636, color='k', linestyle=':')
ax1.axhline(y=17372, color='r', linestyle=':')
ax1.legend(loc=2)
ax2.legend(loc=1)
ax2.set_ylim([0, 200])
plt.show()

Precp_per_day = []
record_day = 100; i = -1

for index, row in test_df.iterrows():
    if math.isnan(row['Inflow_Calculated']):
        None
    else:
        if record_day == index.day:
            Precp_per_day[i] = Precp_per_day[i] + row['Inflow_Calculated']
        else:
            Precp_per_day.append(row['Inflow_Calculated'])
            record_day = index.day
            i += 1
        
    #row['Inflow_Calculated'] 

print(Precp_per_day)
# print(Weather_Data_Day)

#Weather_Data_Day['Inflow'] = Weather_Data_Day[['prcp']].sum().where(Weather_Data_Day['time'].day==1, 0)
# Weather_Data_Day['Inflow'] = Precp_per_day
# for index, row in Weather_Data_Day.iterrows():
#     print(index.day)
#     """use insert() method or simply new column assigned with list."""
#     row['prcp'] = 1 # does not work
#print( Weather_Data_Day['prcp'].sum().where(Weather_Data_Day['time'].day==1, 0) )

# print(Weather_Data_Day)






## Plotting rainfall and pump inflow variability

# fig, ax1 = plt.subplots(1,1)
# Weather_Data_Day.Inflow.plot(ax=ax1, color='brown', label='PST Inflow')
# ax2 = ax1.twinx()
# Weather_Data_Day.prcp.plot(ax=ax2, color='blue', label='Rainfall')
# # test_df.Inflow_Calculated_5min.plot(ax=ax2, color='red', label='In Flow slow')
# ax1.set_ylabel('Total tank inflow [liters]')
# ax2.set_ylabel('Daily rainfall [mm]')
# ax1.legend(loc=2)
# ax2.legend(loc=1)
# plt.show()

# fig, ax1 = plt.subplots(1,1)
# Weather_Data_Hour.prcp.plot(ax=ax1, color='blue', label='Rainfall')
# # test_df.Inflow_Calculated_5min.plot(ax=ax2, color='red', label='In Flow slow')
# ax1.set_ylabel('Hourly rainfall [mm]')
# ax1.legend(loc=1)
# plt.show()

# plt.figure(figsize=[15,10])
# plt.grid(True)
# plt.plot(test_df['Level'],label='Level sensor')
# # plt.plot(test_df['RolAvg_Level_14'],label='Level SMA 14 entries')
# plt.plot(test_df['Volume_RolAvg_Vol_7'],label='Volume (SMA 7 from level)')
# plt.axhline(y=2836, color='k', linestyle='-')
# plt.axhline(y=10636, color='c', linestyle='-')
# plt.axhline(y=17372, color='r', linestyle='-')

# #plt.plot(test_df['CurrentP1'],label='Current Pump 1', secondary_y = True)
# plt.plot(secondary_y = test_df['CurrentP2'],label='Current Pump 2')

# plt.legend(loc=2)


# Results organized per day, per week, per month, or whole year

# Sum of water flow over a whole month, current to power to energy, 

# Average ramp-up and ramp-down time, with variability
# Average duration of each pump episode, with variability
# Quantity of pumping episodes per hour & per day - does hourly operation change during the day? comparing diff days too (dot dot dot)

#Counting pumping operations:
"""if p1-1 0.04 and p1 not 0.04: count +1 and register activation in separate column
if p1-1 not 0.04 and p1 0.04: register deactivation in separate column (different), count time between activation and deactivation as new column 
for actiation through deactivation, sum water flow output:
create volume column based on moving average of water height """


    # https://www.researchgate.net/publication/339646926/figure/fig3/AS:865023043846145@1583248973392/displays-the-neighbors-of-the-curve-nodes-Steps-4-5-and-6-above-Each-red-dot.png
# Relate to days of the week
# From pumping episodes => quantify water flow output
# Quantify water flow input: variation of volume in tank (convert from height) - water flow output
    # input (output) over an hour & over a day, w.v.

# How often higher water inflow, differentiate between rainfall and inflow of other pumps (How??)
# Standard (local catchment) inflow. Variates over days or seasons?

# Relate 3ø current to pumps' power. Relate energy consumption to pumping operations (avg energy consumption w.v., minumum, mxáximum) and per hour/ per day w.v.

# First focus per hour/per day, then per episode?

# Non linear relationship between water flow output and pump (power) operation?

## Check documentation for tank volume

# Define clear m/s outputs, check if p1 p2 always operate within given boundaries

# Start w visualization akin to xls oktober

# Energy use by P1 or P2, for same amount of water output

# Verify highest ever level, then check within 5% and 10% of such highest how often it happens (number of occurrences)

# In intense rain, would it be better to activate pump 2 at lower power rating? Or switching p1 and p2 consecutively the best approach?











# Identify rainfall episodes and differentiate from upstream pumping. Relate volume, frequency, dates. 
# Try to understand how the conveyor belt works during rainfall.
# ATTACHMENT 1

















# ATTACHMENT 1
# from meteostat import Stations
# stations = Stations()
# stations=stations.nearby(55.270025, 14.807155)
# station = stations.fetch(1) # 06193  Hammer Odde      DK   3659.412656 meters apart

# from datetime import datetime
# from meteostat import Hourly

# # Set time period
# start = datetime(2020, 1, 1)
# end = datetime(2020, 1, 1, 23, 59)

# # Get hourly data
# data = Hourly('06193', start, end)
# data = data.fetch()

# # Print DataFrame
# print(data)

# Not working: Cannot load hourly/full/2018/06193.csv.gz from https://bulk.meteostat.net/v2
# Alternative by Javiera below

# from datetime import datetime
# from meteostat import Point, Hourly
 
# # Set time period
# start = datetime(2018, 1, 1)
# end = datetime(2018, 12, 31)
 
# # Create Point for Allinge, DK
# allinge = Point(55.274, 14.8)
 
# # Get hourly data for 2020
# data = Hourly(allinge, start, end)
# data = data.fetch()
 
# print(data)
