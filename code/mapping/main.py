import pandas as pd 
import numpy as np
import itertools
from tools import *

#Load the cleaned dataset
cleaned_data = pd.read_json("df_cleaned.json")

#Replace -1 values that represented the defense by 0 (used as target for the pretrained model)
cleaned_data["QB_Flag"] = cleaned_data["QB_Flag"].replace({-1:0})

#Explode the dataset along the axis of trajectory, each row represents a timestep of player during a play.
Traj_df = cleaned_data[["gameId", "playId", "nflId", "position", "QB_Flag", "Trajectory"]].explode("Trajectory").reset_index(drop = True)

#Convert previous trajectories as X and Y variables
positions = np.stack(Traj_df["Trajectory"])
Traj_df.insert(3, "X", positions[:,0])
Traj_df.insert(3, "Y", positions[:,1])

#Check maximum and minimum values of X and Y
min_x = min(Traj_df["X"])
max_x = max(Traj_df["X"])
min_y = min(Traj_df["Y"])
max_y = max(Traj_df["Y"])

print("Min X is : ", min_x)
print("Max X is : ", max_x)
print("Min Y is : ", min_y)
print("Max Y is : ", max_y)

#The side's length of the created zone in yards. We use 1 yard squared as a zone
step_parameter = 1.0

#Generates all the possible zones between -40 and +40
X_start = np.arange(-40, 40.01, step_parameter)
X_start = np.concatenate([np.array([-100]), X_start])
X_end = np.arange(-40, 40.01, step_parameter)
X_end = np.concatenate([X_end, np.array([100])])
Xs = np.stack([X_start, X_end], axis = 1)

Y_start = np.arange(-40, 40.01, step_parameter)
Y_start = np.concatenate([np.array([-100]), Y_start])
Y_end = np.arange(-40, 40.01, step_parameter)
Y_end = np.concatenate([Y_end, np.array([100])])
Ys = np.stack([Y_start, Y_end], axis = 1)

zones  = list(itertools.product(Xs, Ys))
zones = np.stack([np.concatenate(v) for v in zones])
zones_df = pd.DataFrame(zones, columns = ["X_start", "X_end", "Y_start", "Y_end"])
zones_df["Zones"] = range(1, zones_df.shape[0]+1)

#Convert current coordinates expressed in continuous values into integers to merge relevant zones.
Traj_df["X_start"] = Traj_df["X"].apply(round_off)
Traj_df["X_start"] = Traj_df["X_start"].apply(cap_off)
Traj_df["Y_start"] = Traj_df["Y"].apply(round_off)
Traj_df["Y_start"] = Traj_df["Y_start"].apply(cap_off)

#Merge zones on the specific time step of the trajectory of a player
to_merge = (Traj_df.
            drop_duplicates(["X_start", "Y_start"]).
            reset_index(drop = True)[["X_start", "Y_start"]].
            merge(zones_df[["X_start", "Y_start", "Zones"]], on = ["X_start", "Y_start"], how = "left"))

#Affect a single ID to each zone 
Zone_ID = to_merge[["Zones"]].drop_duplicates()
Zone_ID["Zone_ID"] = range(1, Zone_ID.shape[0]+1)

# Merge Zone ID to the trajectory dataset and drop unrelevant variables
to_merge = to_merge.merge(Zone_ID, on = "Zones", how = "left").drop("Zones", axis = "columns")
Traj_df = Traj_df.merge(to_merge, on = ["X_start", "Y_start"], how = "left")
Traj_df = Traj_df.drop(["Trajectory",	"X_start",	"Y_start", "Y", "X"], axis = "columns")

#Group Zone ID in to a single list by player by play.
Mapped_df = Traj_df.groupby(["gameId", "playId", "nflId", "position", "QB_Flag"]).agg({"Zone_ID" : list}).reset_index()

#Write file
Mapped_df.to_json("mapped_df.json")
to_merge.to_json("zones_df.json")
