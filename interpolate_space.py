import pandas as pd
import numpy as np
from scipy import interpolate


inp_1 = "temperatures_interpolated.csv"
inp_2 = "reordered_evi.csv"

data_1 = pd.read_csv(inp_1)
data_2 = pd.read_csv(inp_2)

times = data_1["time"].unique()

sliced_data_1 = []
sliced_data_2 = []
for t in times:
    sliced_data_1.append(data_1[data_1["time"] == t])
    sliced_data_2.append(data_2[data_2["time"] == t])

joint_sliced_data = []

for i, df in enumerate(sliced_data_1):
    df2 = sliced_data_2[i]
    joint_df = df2.copy()
    long = df["long"].values
    lat = df["lat"].values
    temp = df["temp"].values
    f = interpolate.LinearNDInterpolator(list(zip(lat, long)), temp, np.mean(temp))

    joint_df["temp"] = f(joint_df["lat"], joint_df["long"])
    joint_df["time"] = times[i]
    joint_sliced_data.append(joint_df)

joint_data = pd.concat(joint_sliced_data)

joint_data = joint_data.loc[:, ~joint_data.columns.str.contains('^Unnamed')]

joint_data.reset_index(drop=True, inplace=True)

print(joint_data.head())

"""t_1_data = joint_data[joint_data["time"] == times[10]]

fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
ax1, ax2 = axes
ax1.scatter(t_1_data["long"], t_1_data["lat"], s=10, c=t_1_data["evi"])
ax2.scatter(t_1_data["long"], t_1_data["lat"], s=10, c=t_1_data["temp"], cmap="hot")
plt.show()"""

joint_data.to_csv("formatted_inp.csv.gz", index=False, compression="gzip")
