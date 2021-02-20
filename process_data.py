import pandas as pd
import numpy as np

df = pd.read_csv("formatted_inp.csv.gz")
df["time"] = df["time"]-365 # 2001 jan 1st is now day 0
          

df["year"] = 1 + df["time"] // 365 
df.to_csv("reformatted_inp.csv.gz")


