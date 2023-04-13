import pandas as pd 
import numpy as np
from tools import *

#Loading datasets for trajectories, games and plays
raw_data = pd.read_parquet("tracking.gzip")
games = pd.read_csv("games.csv")
plays = pd.read_csv("plays.csv")

#For each game, stroing which team was Away and which was home
game_indice = games[["gameId", "homeTeamAbbr", "visitorTeamAbbr"]]
game_indice.columns = ["gameId", "home", "away"]
game_indice = game_indice.melt(id_vars = "gameId", var_name = "team", value_name = "name")

#For each play flagging by a value 1 in QB_Flag which team had the ball.
play_indice = plays[["gameId", "playId", "possessionTeam"]]
play_indice.columns = ["gameId", "playId", "name"]
play_indice.insert(3, "QB_Flag", np.ones(play_indice.shape[0]))

#Storing for each play in each game, which team attacks or defends
offdef = game_indice.merge(play_indice, on = ["gameId", "name"], how = "left")
offdef = offdef[["gameId", "playId", "team", "QB_Flag"]]

#Removing the football tracking from the dataset
raw_data = raw_data[raw_data["team"] != "football"].reset_index(drop = True)
#Keeping only frames every 0.4 seconds for the 9 first seconds of the action
frames_to_keep = range(1,90,4) 
raw_data = raw_data[raw_data["frameId"].isin(frames_to_keep)].reset_index(drop = True)

# Select data of interest and aggregate them by Game, Play, Team, Player as list
cols_to_keep = ["gameId", "playId", "team", "nflId", "position", "frameId", "x", "y"]
selected_data = raw_data[cols_to_keep]
grouped_data = selected_data.groupby(["gameId", "playId", "team", "nflId", "position"]).agg({"x" : list, "y": list}).reset_index()
grouped_data = grouped_data.merge(offdef, on = ["gameId", "playId", "team"], how = "outer").dropna(subset = ["nflId"]).reset_index(drop = True).fillna(-1)

# First difference for X coordinates. If the player defends, we inverse his differences.
grouped_data["x_diff"] = grouped_data["x"].apply(process_traj) * grouped_data["QB_Flag"]
grouped_data["x_reb"] = grouped_data["x_diff"].apply(rebuild_positions)

# First difference for Y coordinates. If the player defends, we inverse his differences.
grouped_data["y_diff"] = grouped_data["y"].apply(process_traj) * grouped_data["QB_Flag"]
grouped_data["y_reb"] = grouped_data["y_diff"].apply(rebuild_positions)

#Storing X and Y coordinates in a single array called "Trajectory"
grouped_data["Trajectory"] = grouped_data[["x_reb", "y_reb"]].apply(create_trajectory, axis = 1)

#Write dataset
grouped_data.to_json("df_cleaned.json")