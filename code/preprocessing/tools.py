import pandas as pd 
import numpy as np

def standardize_origin(l: list) -> np.array:
    """Convert a trajectory expressed with X,Y coordinates relative to the field into a trajectory starting by the [0,0] coordinate.

    Args:
        l (list): list of coordinates

    Returns:
        np.array: array of standardized coordinates
    """
    arr = np.array(l)
    return arr - arr[0]

def differentiated_traj(arr: np.array) -> np.array:
    """Compute the difference between two coordinates at time t and t+1. Returns the differences.

    Args:
        arr (np.array): array of standardized coordinates

    Returns:
        np.array: first difference of coordinates
    """
    return arr[1:]-arr[:-1]

def process_traj(traj: list) -> np.array:
    """Apply standardization and first difference to a list of coordinates.

    Args:
        traj (list): list of coordinates

    Returns:
        np.array: first difference of the trajectory
    """
    standardized = standardize_origin(traj)
    differenced = differentiated_traj(standardized)
    return differenced

def rebuild_positions(arr: np.array) -> np.array:
    """Takes a first differenced array as input and rebuild it as a trajectory starting by 0.

    Args:
        arr (np.array): array of first differences

    Returns:
        np.array: the rebuilt trajectory
    """
    rebuilt = np.cumsum(arr)
    final = np.concatenate([[0], rebuilt])
    return final

def create_trajectory(arr : list) -> np.array:
    """Takes X and Y coordinates and stacks them into a single 2D array.

    Args:
        arr (list): list of X and Y coordinates

    Returns:
        np.array: 2D array with N time steps where X and Y are recorded.
    """
    x = arr[0]
    y = arr[1]

    x = np.expand_dims(x, axis = 1)
    y = np.expand_dims(y, axis = 1)

    return np.concatenate((x, y), axis = 1)