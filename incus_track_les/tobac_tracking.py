# incus_track_les/tobac_tracking.py

from pathlib import Path
import pandas as pd
import tobac
import datetime as dt
import numpy as np
import xarray as xr

from incus_track_les import read_data, subset_data, interpolate_w, find_model_metadata
from incus_track_les.data_readers import find_dxy_from_grid_level
import incus_track_les.tobac_families as tobac_fam

def run_feature_detection_timestep(filepath: Path, coords: dict, parameters: dict[str, dict]) -> dict[str, pd.DataFrame]:
    meta = find_model_metadata(filepath)
    grid_level = meta['grid_level']
    dxy = find_dxy_from_grid_level(grid_level)

    # set up the output which is a dictionary of features corresponding to each input parameter dictionary
    output_features = {}

    vertical_velocity = read_data(filepath, coords=coords, variables=['vertical_velocity','zonal_velocity'])
    vertical_velocity = subset_data(vertical_velocity)
    vertical_velocity = interpolate_w(vertical_velocity, meta['model_type'],drop_others=True)

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
    
    for var in ['xdist','ydist','altitude']:
        tracks[f"{var}_speed"] = tracks.groupby('cell')[var].diff()/30 # speed in m/s

    return(tracks)

def filter_unused_masks(in_feature_df, in_mask, mask_name = 'feature'):
    valid_features = in_feature_df[mask_name].dropna().unique().astype(int)

    out_mask = xr.apply_ufunc(
        lambda arr, ids: np.where(np.isin(arr, ids), arr, 0),
        in_mask,
        valid_features,
        dask="allowed",
        vectorize=False
    )

    return(out_mask)

                              
def run_segmentation_timestep(filepath: Path, trackspath: Path, maskspath: Path, coords: dict, parameters: dict[str, dict]) -> dict[str, pd.DataFrame]:
    #from scipy.ndimage import gaussian_filter
    from dask_image.ndfilters import gaussian_filter

    meta = find_model_metadata(filepath)
    grid_level = meta['grid_level']
    dxy = find_dxy_from_grid_level(grid_level)

    time = meta['time']

    data = read_data(filepath, coords=coords, variables=['vertical_velocity','cloud_condensate'])
    data = subset_data(data)
    data = interpolate_w(data, model_type=meta['model_type'])

    # TESTING: what if we add a filter to data before running the masks?
    sigma_threshold = 1
    """data['vertical_velocity'] = gaussian_filter(
        data.vertical_velocity.values, sigma=sigma_threshold
    )
    data['cloud_condensate'] = gaussian_filter(
        data.cloud_condensate.values, sigma=sigma_threshold
    )"""
    data['vertical_velocity'] = xr.apply_ufunc(
        gaussian_filter, 
        data.vertical_velocity, 
        kwargs={'sigma': (0,sigma_threshold,sigma_threshold,sigma_threshold)},
        dask="allowed",
    )
    
    data['cloud_condensate'] = xr.apply_ufunc(
        gaussian_filter, 
        data.cloud_condensate, 
        kwargs={'sigma': (0,sigma_threshold,sigma_threshold,sigma_threshold)},
        dask="allowed",
    )
    
    print('data read done', flush=True)

    # test loading into memory from beginning?

    tracks = pd.read_parquet(trackspath)
    tracks = tracks[tracks.time==time]

    print('tracks read done', flush=True)
    
    # set up the output which is a dictionary of features corresponding to each input parameter dictionary
    output_features = {}
    
    for experiment_name, experiment_params in parameters.items():

        # first, watershed thermals around each updraft point
        thermal_masks, thermal_fts = tobac.segmentation.segmentation(tracks, 
                                                                     data.vertical_velocity,
                                                                     dxy=dxy,
                                                                     **experiment_params['thermal_segmentation_params'],
                                                                     )   
        

        print('thermal segmentation done', flush=True)
        
        # second, identify families based on cloud condensate regions
        cloud_fam, cloud_masks = tobac_fam.identify_feature_families_from_data_with_mask(thermal_fts,
                                                                            thermal_masks,
                                                                            data.cloud_condensate, 
                                                                            threshold = experiment_params['cloud_threshold'], 
                                                                            return_grid=True, 
                                                                            family_column_name='cloud_feature_id',
                                                                            min_overlap_count=64)
        
        #cloud_masks = cloud_masks.rename({'z':'z_stag'})

        thermal_masks_keep = filter_unused_masks(cloud_fam, thermal_masks, mask_name='feature')
        
        del thermal_masks
        
        print('cloud mask done', flush=True)

        # fourth, identify updraft families based on thermal masks
        updraft_fam, updraft_masks = tobac_fam.identify_feature_families_from_data_with_mask(cloud_fam, 
                                                                                   thermal_masks_keep, 
                                                                                   thermal_masks_keep,
                                                                                   threshold=0,
                                                                                   return_grid=True,
                                                                                   family_column_name = 'updraft_feature_id',
                                                                                   min_overlap_count=0)

        
        cloud_masks_keep = filter_unused_masks(updraft_fam, cloud_masks, mask_name='cloud_feature_id')
        del cloud_masks

        output_features[experiment_name] = updraft_fam

        print('updraft mask done', flush=True)

        # finally, merge all the masks into one Dataset and save
        masks = xr.Dataset({
            'cloud_mask': cloud_masks_keep,
            'thermal_mask': thermal_masks_keep,
            'updraft_mask': updraft_masks
        })

        if experiment_name=='prod':
            filename= 'masks'
        elif experiment_name:
            filename=f"masks_{experiment_name}"
        else:
            filename="masks"
        
        Path(maskspath, filename).mkdir(exist_ok=True)

        masks.to_netcdf(Path(maskspath, 
                             filename, f"{time.strftime('%Y-%m-%d-%H%M%S')}.h5"),
                             engine='h5netcdf',
                             encoding={"cloud_mask": {"zlib": True, "complevel": 9},
                                       "thermal_mask": {"zlib": True, "complevel": 9},
                                       "updraft_mask": {"zlib": True, "complevel": 9}},)
        
        del cloud_masks_keep,thermal_masks_keep,updraft_masks,masks

    return(output_features)



def run_segmentation_total_cond_timestep(filepath: Path, trackspath: Path, maskspath: Path, coords: dict, parameters: dict[str, dict]) -> dict[str, pd.DataFrame]:
    meta = find_model_metadata(filepath)
    grid_level = meta['grid_level']
    dxy = find_dxy_from_grid_level(grid_level)

    time = meta['time']

    data = read_data(filepath, coords=coords, variables=['vertical_velocity','total_condensate'])
    data = subset_data(data)
    data = interpolate_w(data, model_type=meta['model_type'])

    print('data read done', flush=True)

    # test loading into memory from beginning?

    tracks = pd.read_parquet(trackspath)
    tracks = tracks[tracks.time==time]

    print('tracks read done', flush=True)
    
    # set up the output which is a dictionary of features corresponding to each input parameter dictionary
    output_features = {}
    
    for experiment_name, experiment_params in parameters.items():

        # first, watershed thermals around each updraft point
        thermal_masks, thermal_fts = tobac.segmentation.segmentation(tracks, 
                                                                     data.vertical_velocity,
                                                                     dxy=dxy,
                                                                     **experiment_params['thermal_segmentation_params'],
                                                                     #statistic=statistics
                                                                     )   
        

        print('thermal segmentation done', flush=True)
        
        # second, identify families based on cloud condensate regions
        cloud_fam, cloud_masks = tobac_fam.identify_feature_families_from_data_with_mask(thermal_fts,
                                                                            thermal_masks,
                                                                            data.total_condensate, 
                                                                            threshold = experiment_params['cloud_threshold'], 
                                                                            return_grid=True, 
                                                                            family_column_name='cloud_feature_id',
                                                                            min_overlap_count=64)
        
        #cloud_masks = cloud_masks.rename({'z':'z_stag'})

        thermal_masks_keep = filter_unused_masks(cloud_fam, thermal_masks, mask_name='feature')
        
        del thermal_masks
        
        print('cloud mask done', flush=True)

        # fourth, identify updraft families based on thermal masks
        updraft_fam, updraft_masks = tobac_fam.identify_feature_families_from_data_with_mask(cloud_fam, 
                                                                                   thermal_masks_keep, 
                                                                                   thermal_masks_keep,
                                                                                   threshold=0,
                                                                                   return_grid=True,
                                                                                   family_column_name = 'updraft_feature_id',
                                                                                   min_overlap_count=0)

        
        cloud_masks_keep = filter_unused_masks(updraft_fam, cloud_masks, mask_name='cloud_feature_id')
        del cloud_masks

        output_features[experiment_name] = updraft_fam

        print('updraft mask done', flush=True)

        # finally, merge all the masks into one Dataset and save
        masks = xr.Dataset({
            'cloud_mask': cloud_masks_keep,
            'thermal_mask': thermal_masks_keep,
            'updraft_mask': updraft_masks
        })

        if experiment_name=='prod':
            filename= 'masks'
        elif experiment_name:
            filename=f"masks_{experiment_name}"
        else:
            filename="masks"
        
        Path(maskspath, filename).mkdir(exist_ok=True)

        masks.to_netcdf(Path(maskspath, 
                             filename, f"{time.strftime('%Y-%m-%d-%H%M%S')}.h5"),
                             engine='h5netcdf',
                             encoding={"cloud_mask": {"zlib": True, "complevel": 9},
                                       "thermal_mask": {"zlib": True, "complevel": 9},
                                       "updraft_mask": {"zlib": True, "complevel": 9}},)
        
        del cloud_masks_keep,thermal_masks_keep,updraft_masks,masks

    return(output_features)


