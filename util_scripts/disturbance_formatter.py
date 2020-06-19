import pandas as pd


dist_df = pd.read_csv("data/disturbance_data/hour_poop_rain.csv", names=["Hour", "Poop", "Rain"])


poop_gain = 1
rain_gain = 1

dist_df["Poop"] = dist_df["Poop"] * poop_gain
dist_df["Poop"] = dist_df["Rain"] * rain_gain
dist_df["combined"] = dist_df["Poop"] + dist_df["Rain"]


output_df = dist_df.drop(["Rain", "Poop"], axis=1)

output_df.to_csv('data/disturbance_data/hour_poop_rain.dat', header=None,  index=False)

print("done")