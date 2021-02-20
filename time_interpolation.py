#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:55:09 2021

@author: lorenzoversini
"""
### perform an interpolation to adapt the time of the
### temperature measurements to the time of the EVI measurements

import numpy as np

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


gdf = gpd.read_file("../data/illinois-counties.geojson")
df = pd.read_csv("../data/EVI_stacked.csv")


df_plot = df[df["year"]==2019]

fig, ax = plt.subplots(figsize=(5, 5))
gdf.plot(ax=ax, facecolor='none', edgecolor='red')
plt.scatter(df_plot["long"], df_plot["lat"], c=df_plot["evi_1"])




print(gdf.head)

#%%
print(gdf.describe())
print(gdf.columns)

print('df:')

print(df.describe())
print(df.columns)

#%%
# =============================================================================
# Temperatures
# =============================================================================
tempset=pd.read_csv('../data/ERA5.csv')
yieldset=pd.read_csv('../data/IL_yield.csv')


print(tempset.describe())
# print(tempset.)

#%%
dfnostacked=pd.read_csv('../data/EVI.csv')
#%%
import datetime
# Obtaining list of times I want to interpolate throgh

years = df.year.unique()
print(years) # years in EVI
days = np.arange(1,365,16,dtype='int')
print(days) # days in EVI
days_list_EVI = [] # days to perform interpolation to

for i,year in enumerate(years):
    yearstring=[]
    for day in days:
        timestamp = pd.to_datetime(datetime.datetime(year, 1, 1)+
                                   datetime.timedelta(int(day-1)))
        # dayyear=365.25 * (i+1) + day
        days_list_EVI.append(timestamp)

timestart = pd.to_datetime('2000-01-01T00:00:00') # as zeroth
#datetime.datetime(year, 1, 1)+datetime.timedelta(days-1)
days_list_EVI=pd.to_datetime(days_list_EVI)
days_list_EVI=days_list_EVI-timestart
days_list_EVI=days_list_EVI.days.array.astype('int')



#%%
from scipy.ndimage import convolve1d
import scipy as sp

#obtain the list of all possible coordinates
listlatitudes=tempset.lat.unique()
listlongitudes=tempset.long.unique()

finallist=pd.DataFrame(columns=['lat','long','time','temp'])


for i,lat in enumerate(listlatitudes):
    for j,long in enumerate(listlongitudes):
        subset = tempset[(tempset['lat']==lat) 
                 & (tempset['long']==long)]
        if not subset.empty:
            days_temp = pd.to_datetime(subset.time)-timestart #datset.empty
            temps = subset.t2m
            argsort = days_temp.dt.days.array.argsort()
            
            sortedtime = days_temp.dt.days.to_numpy()[argsort]
            sortedtemp = temps.array.to_numpy()[argsort]
            
            #smooth the temperature
            gaussiankernel = np.exp(-np.arange(-3,4)**2/(2*0.8))
            gaussiankernel=gaussiankernel/np.sum(gaussiankernel)
            filteredtemp = convolve1d(sortedtemp,gaussiankernel,mode="reflect")
            
            #interpolate temperatures in EVI times
            interpolated_temperatures = np.interp(days_list_EVI,sortedtime,filteredtemp)
            
            lentimes = len(days_list_EVI)
            
            timedataframe = pd.DataFrame(np.c_[[lat]*lentimes,[long]*lentimes,days_list_EVI, interpolated_temperatures],
                                         columns=['lat','long','time','temp'])
            #append to final dataset
            finallist=finallist.append(timedataframe)
        
print(finallist)
#save final dataset
finallist.to_csv('temperatures_interpolated.csv')
        
    
    
    
    
    
    