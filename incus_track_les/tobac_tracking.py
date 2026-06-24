# incus_track_les/tobac_tracking.py

from pathlib import Path
import pandas as pd
import tobac
import datetime as dt

from incus_track_les import read_data, subset_data, find_model_metadata
from incus_track_les.data_readers import find_dxy_from_grid_level

def run_feature_detection_timestep(filepath: Path, coords: dict, parameters: dict[str, dict]) -> dict[str, pd.DataFrame]:

    grid_level = find_model_metadata(filepath)['grid_level']
    dxy = find_dxy_from_grid_level(grid_level)

    # set up the output which is a dictionary of features corresponding to each input parameter dictionary
    output_features = {}

    vertical_velocity = read_data(filepath, coords=coords, variables=['vertical_velocity'])
    vertical_velocity = subset_data(vertical_velocity)

    for experiment_name, experiment_params in parameters.items():
        features = tobac.feature_detection.feature_detection_multithreshold(vertical_velocity,
                                                                            dxy = dxy,
                                                                            **experiment_params)
        output_features[experiment_name] = features
        
    return(output_features)


def run_tracking(features_path: Path, grid_level: int, parameters: dict) -> pd.DataFrame:

    # get grid spacing
    dxy = find_dxy_from_grid_level(grid_level)

    # read features file
    features = pd.read_parquet(features_path)
    
    tracks = tobac.tracking.linking_trackpy(
                        features,
                        None,
                        dt=30,  # time in seconds separating each frame
                        dxy=dxy,
                        **parameters,)

    # remove untracked features
    tracks = tracks[tracks.cell>0]

    tracks['lifetime'] = tracks.cell.map(tracks.groupby('cell').time_cell.max()/dt.timedelta(minutes=1))
    tracks['frac_lifetime'] = (tracks['time_cell']/(tracks['lifetime']*dt.timedelta(minutes=1))) # fraction of lifetime from 0 to 1
    
    for var in ['xdist','ydist','altitude_stag']:
        tracks[f"{var}_speed"] = tracks.groupby('cell')[var].diff()/30 # speed in m/s

    return(tracks)