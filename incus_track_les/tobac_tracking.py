# incus_track_les/tobac_tracking.py

from pathlib import Path
import pandas as pd
import tobac
import datetime as dt
import numpy as np

from incus_track_les import read_data, subset_data, find_model_metadata
from incus_track_les.data_readers import find_dxy_from_grid_level
import incus_track_les.tobac_families as tobac_fam

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

def run_segmentation_timestep(filepath: Path, trackspath: Path, maskspath: Path, coords: dict, parameters: dict[str, dict]) -> dict[str, pd.DataFrame]:
    meta = find_model_metadata(filepath)
    grid_level = meta['grid_level']
    dxy = find_dxy_from_grid_level(grid_level)

    time = meta['time']

    data = read_data(filepath, coords=coords, variables=['vertical_velocity','cloud_condensate'])

    print('data read done', flush=True)

    data = subset_data(data)

    tracks = pd.read_parquet(trackspath)
    tracks = tracks[tracks.time==time]

    print('tracks read done', flush=True)

    statistics = {
        'max_w': np.max,
        'mean_w': np.mean
    }

    
    # set up the output which is a dictionary of features corresponding to each input parameter dictionary
    output_features = {}
    
    for experiment_name, experiment_params in parameters.items():

        # first, identify families based on cloud condensate regions
        cloud_fam, cloud_masks = tobac_fam.identify_feature_families_from_data(tracks, 
                                                                           data.cloud_condensate, 
                                                                           threshold = experiment_params['cloud_threshold'], 
                                                                           return_grid=True, 
                                                                           family_column_name='cloud_feature_id')
        
        output_features[experiment_name] = {'cloud_families': cloud_fam}

        print('cloud mask done', flush=True)

        # second, watershed thermals around each updraft point
        thermal_masks, thermal_fts = tobac.segmentation.segmentation(tracks, 
                                                                     data.vertical_velocity,
                                                                     dxy=dxy,
                                                                     **experiment_params['thermal_segmentation_params'],
                                                                     statistic=statistics)
        output_features[experiment_name]['thermal_features'] = thermal_fts        


        print('thermal segmentation done', flush=True)

        # third, check for overlap between thermal mask and cloud mask, discarding any thermal features which do not overlap with cloud mask

        overlap_masks = (thermal_masks > 0) & (cloud_masks > 0)
        thermal_fts_keep = np.unique(thermal_masks.where(overlap_masks).compute())
        thermal_fts_keep = thermal_fts_keep[~np.isnan(thermal_fts_keep) & (thermal_fts_keep > 0)]

        thermal_masks_keep = thermal_masks.where(thermal_masks.isin(thermal_fts_keep), 0)

        del thermal_masks

        print('overlap mask done', flush=True)

        # fourth, identify updraft families based on thermal masks
        updraft_fam, updraft_masks = tobac_fam.identify_feature_families_from_data(tracks, 
                                                                                   thermal_masks_keep, 
                                                                                   threshold=1,
                                                                                   return_grid=True,
                                                                                   family_column_name = 'updraft_feature_id')

        output_features[experiment_name]['updraft_families'] = updraft_fam


        print('updraft mask done', flush=True)

        # finally, merge all the masks into one Dataset and save
        masks = xr.Dataset({
            'cloud_mask': cloud_masks,
            'thermal_mask': thermal_masks_keep,
            'updraft_mask': updraft_masks
        })

        Path(maskspath, f'masks-{experiment_name}').mkdir(exist_ok=True)

        masks.to_netcdf(Path(maskspath, 
                             f'masks-{experiment_name}', f"{time.strftime('%Y-%m-%d-%H%M%S')}.h5"),
                             engine='h5netcdf')
        
        del cloud_masks,thermal_masks_keep,updraft_masks,masks

    return(output_features)




                              