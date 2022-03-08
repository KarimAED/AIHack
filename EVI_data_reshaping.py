#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:42:00 2021

@author: lorenzoversini
"""
import datetime

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./EVI_stacked.csv")


columnlabels = [
    "year",
    "county",
    "long",
    "lat",
    "evi_1",
    "evi_17",
    "evi_33",
    "evi_49",
    "evi_65",
    "evi_81",
    "evi_97",
    "evi_113",
    "evi_129",
    "evi_145",
    "evi_161",
    "evi_177",
    "evi_193",
    "evi_209",
    "evi_225",
    "evi_241",
    "evi_257",
    "evi_273",
    "evi_289",
    "evi_305",
    "evi_321",
    "evi_337",
    "evi_353",
]
newcolumnlabels = ["lat", "long", "time", "county", "evi"]
evi_list = [
    "evi_1",
    "evi_17",
    "evi_33",
    "evi_49",
    "evi_65",
    "evi_81",
    "evi_97",
    "evi_113",
    "evi_129",
    "evi_145",
    "evi_161",
    "evi_177",
    "evi_193",
    "evi_209",
    "evi_225",
    "evi_241",
    "evi_257",
    "evi_273",
    "evi_289",
    "evi_305",
    "evi_321",
    "evi_337",
    "evi_353",
]
day_list = [int(ele[4:]) for ele in evi_list]
# columns_titles = ["B","A"]
# df=df.reindex(columns=columns_titles)


#%%


def getDay(year, day):
    timestamp = pd.to_datetime(
        datetime.datetime(year, 1, 1) + datetime.timedelta(int(day - 1))
    )

    timestart = pd.to_datetime("2000-01-01T00:00:00")  # as zeroth
    # datetime.datetime(year, 1, 1)+datetime.timedelta(days-1)
    timestamp = pd.to_datetime(timestamp)
    finalday = timestamp - timestart
    finalday = finalday.days
    return finalday


listlatitudes = df.lat.unique()
listlongitudes = df.long.unique()

finallist = pd.DataFrame(columns=newcolumnlabels)

temporary_list = []
for i, lat in enumerate(listlatitudes):
    for j, long in enumerate(listlongitudes):
        subdata = df[(df["lat"] == lat) & (df["long"] == long)]
        if not subdata.empty:
            county = subdata.county.unique()[0]
            # temporary_list = []
            allyears = subdata.year.unique()
            for m, y in enumerate(allyears):
                for k, day in enumerate(day_list):
                    absoluteday = getDay(y, day)
                    evi = subdata["evi_" + repr(day)].to_numpy()[m]

                    temporary_list.append([lat, long, absoluteday, county, evi])


temporary_list = pd.DataFrame(temporary_list, columns=newcolumnlabels)
# print(np.array(temporary_list))

print(temporary_list)

#%%

# save data
temporary_list.to_csv("reordered_evi.csv")
