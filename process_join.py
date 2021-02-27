import pandas as pd
import numpy as np

inp = pd.read_csv("formatted_inp.csv.gz")
inp["time"] = inp["time"] - 365  # 2001 jan 1st is now day 1

inp["year"] = 1 + inp["time"] // 365  # Get year after 2001
out = pd.read_csv("IL_yield.csv")
out["year"] = out["year"].astype(int) - 2000

out_counties = np.sort(out["county"].unique())


inp = inp.loc[:, ~inp.columns.str.contains('^Unnamed')]

years = inp["year"].unique()
counties = np.sort(inp["county"].unique())

counties = np.intersect1d(out_counties, counties)


events_inp = []
events_out = []
for year in years:
    for county in counties:
        print(year, county)
        event = []
        raw_event = inp[(inp["year"] == year) & (inp["county"] == county)].copy()
        raw_event["pos"] = raw_event["lat"].astype(str) + raw_event["long"].astype(str)
        u_pos = raw_event["pos"].unique()
        for pos in u_pos:
            loc = raw_event[raw_event["pos"] == pos]
            loc_data = np.array([loc["evi"].values, loc["temp"].values])
            event.append(loc_data.T)
        event = np.array(event)
        events_inp.append(event)
        events_out.append(out[(out["year"] == year) & (out["county"] == county)]["yield"].values)


events_inp = np.array(events_inp)
events_out = np.array(events_out)

np.savez_compressed(inp=events_inp, out=events_out, file="joint_set.npz")
