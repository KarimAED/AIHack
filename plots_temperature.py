import pandas as pd
import geopandas as gpd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df_yield = pd.read_csv("IL_yield.csv")
df_evi = pd.read_csv("EVI.csv")
df_temperature = pd.read_csv("ERA5.csv")
gdf_counties = gpd.read_file("illinois-counties.geojson")
df_temperature["t2m"] = df_temperature["t2m"] - 273
#%%
"""
following plots average temperature for selected counties over a specified year
"""

county_list = ["JO DAVIESS", "STEPHENSON", "WINNEBAGO", "ADAMS", "ALEXANDER", "BOND"]
time = []
colour_list = ["navy", "green", "purple", "red", "orange", "grey"]

sns.set_style("darkgrid")

plt.figure(1)
for county in county_list:
      df_county = df_temperature[df_temperature['county'] == county]
      df_county['time'] = pd.to_datetime(df_county['time'])
      df_county.groupby('time').mean()
      df_county_2015 = df_county[pd.DatetimeIndex(df_county['time']).year == 2015]
      df_county_2015 = df_county_2015.sort_values(by = ["time"])
      
      plt.plot(df_county_2015["time"], df_county_2015["t2m"], "o-", lw=0.5, ms=2, label= county.capitalize())
      
    #  df_county_2012.plot.line("time", "t2m", color= "red", label="Temperature")
        
plt.xlabel("Time")
plt.ylabel("Temperature (C)")
plt.legend()
plt.title("Average County Temperature over Time for some Illinois Counties")
plt.savefig("plot.png", dpi=500)


"""
following plots average temperature over a year for a particular county for a range of years
"""
#years_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
years_list = [2010, 2011, 2012, 2014, 2015]

fig, ax = plt.subplots(figsize=(6,4))

                              
for year in years_list:
    df_bond = df_temperature[df_temperature['county'] == "BOND"]
    df_bond['time'] = pd.to_datetime(df_bond['time'])
    df_bond.groupby('time').mean()
    df_bond = df_bond.sort_values(by = ["time"])
    df_bond_year = df_bond[pd.DatetimeIndex(df_county['time']).year == year]
    
    time_series_w_year = df_bond_year["time"]
    x = time_series_w_year.dt.strftime('%m-%d')
    xlabel = time_series_w_year.dt.strftime('%m-%d').values
    xlabel = [xlabel[i] for i in range(len(xlabel)) if i%2==0]
    t2m = df_bond_year["t2m"].values
    t2m = [t2m[i] for i in range(len(t2m)) if i%2 == 0]
# need to create series with just the dates without years
    ax.plot(xlabel,t2m, "o-", lw=0.5, ms=2, label= year)
    x = time_series_w_year.dt.strftime('%m-%d')


every_other = lambda i, x: x[i] if i%2 == 0 else " "
fig.autofmt_xdate()
ax.set_xlabel("Time")
ax.set_ylabel("Temperature")
ax.set_xticklabels([every_other(i, list(xlabel)) for i in range(len(xlabel))])
ax.legend()
ax.set_title("Average County Temperature over Time for Bond")
plt.show()     


