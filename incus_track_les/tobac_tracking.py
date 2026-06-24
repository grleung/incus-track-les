# incus_track_les/tobac_tracking.py

from pathlib import Path
import pandas as pd
import tobac

from incus_track_les import read_data, subset_data, find_model_metadata
from incus_track_les.data_readers import find_dxy_from_grid_level

def feature_detection_timestep(filepath: Path, coords: dict, parameters: dict[str, dict]) -> dict[str, pd.DataFrame]:

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








    
