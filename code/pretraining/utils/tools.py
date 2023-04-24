import pandas as pd 
import numpy as np
import random

def define_sample(df: pd.DataFrame, bound: int) -> pd.DataFrame:
  if df.shape[0] < bound:
    return df
  else:
    return df.sample(bound).reset_index(drop = True)

def under_sample(df: pd.DataFrale, var_to_split: str, upper_bound : int)-> pd.DataFrame:
  splits = [df[df[var_to_split] == v].reset_index(drop = True) for v in df[var_to_split].unique()]
  sampled = [define_sample(v, upper_bound) for v in splits]
  return pd.concat(sampled).reset_index(drop = True)

def get_true_examples(game: int, play: int, key, dict_of_df : dict) -> pd.DataFrame:
  ret = dict_of_df[game][play].sample(1)
  ret.insert(6, "key", np.array([key]))
  return ret

def get_false_example(length : int, game : int, games_possible : list, key, dict_of_df : dict) -> pd.DataFrame:
  length__games = dict_of_df[length]
  possible_games = [v for v in games_possible if v != game]
  possible_games = [g for g in length__games if g in possible_games]
  selected_game = random.sample(possible_games, 1)[0]
  ret = length__games[selected_game].sample(1)
  ret.insert(6, "key", np.array([key]))
  return ret

def insert_mask(arr : np.array) -> list:
  base = np.zeros(len(arr))
  filtered = arr[arr !=0]
  idx_to_insert = np.random.randint(low = 4, high = len(filtered)-1)
  base[idx_to_insert] = 1
  return list(base)

def mask_zone(mask_id : int, arr: np.array) -> list:
  idx = list(mask_id).index(1)
  if idx <2:
    print("Error")

  masked = arr
  masked[idx] = 6063
  return masked

def get_mask_id(mask_id: int, arr : np.array) -> list:
  idx = list(mask_id).index(1)
  if idx <2:
    print("Error")
  return arr[idx]


def append_special_token(arr: np.array) -> np.array:
  special_token = np.array([6064, 6065, 6066])
  return np.concatenate([special_token, arr])

def append_special_temp(arr: np.array) -> np.array:
  special_temp = np.array([24, 24, 24])
  return np.concatenate([special_temp, arr])

def append_ids_maks(arr: np.array) -> np.array:
  return np.concatenate([np.array([0, 0, 0]), arr])

