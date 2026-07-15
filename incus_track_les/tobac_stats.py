
# incus_track_les/tobac_stats.py

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import scipy.ndimage

from incus_track_les import read_data, subset_data, find_model_metadata, interpolate_w
from incus_track_les.data_readers import find_dxy_from_grid_level

def get_extent(pos:np.ndarray, feature_ids_at_loc:np.ndarray, feature_ids_to_find:np.ndarray, shift_max=None):
    """Returns the minimum and maximum values of the given index position
    for each feature

    Args:
        pos (np.ndarray): array of positions in given dimension at each point
        feature_ids_at_loc (np.ndarray): array of feature ID at each point
        feature_ids_to_find (np.ndarray): array of feature IDs to consider
        shift_max (int, optional): If this is given a value, then the max position is shifted by 1 so the extent of max_pos-min_pos is never zero; shift_max should be the maximum value possible, typically the length dimension corresponding to pos. Defaults to None.
    """

    min_pos = scipy.ndimage.minimum(pos, labels=feature_ids_at_loc, index=feature_ids_to_find).astype(int)
    max_pos = scipy.ndimage.maximum(pos, labels=feature_ids_at_loc, index=feature_ids_to_find).astype(int)

    if shift_max:
        # shift the max position up by 1 so that we count top of this grid cell
        # otherwise if the feature only spans 1 cell, the depth/extent becomes 0
        max_pos = np.minimum(max_pos + 1, shift_max)

    return(min_pos, max_pos)

def get_footprint(y_idx:np.ndarray, x_idx:np.ndarray, feature_ids_at_loc:np.ndarray, feature_ids_to_find:np.ndarray):
    """Returns the footprint in number of pixels. Footprint of a feature is defined as follows: if you were looking down at feature from a bird's eye view, what is the size of area occupied by the feature?

    Args:
        y_idx (np.ndarray): array of y coordinates at each point
        x_idx (np.ndarray): array of x coordinates at each point
        feature_ids_at_loc (np.ndarray): array of feature ID at each point
        feature_ids_to_find (np.ndarray): array of feature IDs to consider
    """
    
    # pair each feature ID with corresponding (y, x) coordinates for all non-zero mask points
    feature_ids_with_xyloc = np.column_stack((feature_ids_at_loc, y_idx, x_idx))
    # we take only the unique values, so ignore repeats if the mask spans more than one vertical level
    feature_ids_with_xyloc = np.unique(feature_ids_with_xyloc, axis=0)

    # now just take the feature IDs again, where each item corresponds to one (y,x) point occupied by the feature
    footprint_feature_ids = feature_ids_with_xyloc[:, 0]

    # finally just group by the feature IDs, and sum up the count of how footprint pixels belong to each feature
    footprint_counts = scipy.ndimage.sum(np.ones_like(footprint_feature_ids), labels=footprint_feature_ids, index=feature_ids_to_find)

    return(footprint_counts)


def get_altitude_of_max(values, z_idx, alt_km, feature_ids_at_loc:np.ndarray, feature_ids_to_find:np.ndarray):
    max_loc = scipy.ndimage.maximum_position(values, labels=feature_ids_at_loc, index=feature_ids_to_find)
    max_loc = np.array([pos[0] for pos in max_loc], dtype=int)
    max_loc = np.clip(max_loc, 0, len(z_idx) - 1)
    max_loc = z_idx[max_loc]

    return(alt_km[max_loc])
    

def calculate_statistics(filepath,featurespath, maskspath, coords, stats_level='thermal'):
    meta = find_model_metadata(filepath)
    time = meta['time']

    features = pd.read_parquet(featurespath)

    mask = xr.open_dataset(Path(maskspath,f"{time.strftime('%Y-%m-%d-%H%M%S')}.h5"))
    mask = mask[f'{stats_level}_mask']
    
    features_at_time = features[features.time==time].copy()

    if stats_level=='thermal':
        features_at_time = features_at_time[['frame','feature','cell','time','timestr','x','y','z','altitude','latitude','longitude','time_cell','lifetime','frac_lifetime','xdist_speed','ydist_speed','altitude_speed','cloud_feature_id','updraft_feature_id']]

        feature_ids = features_at_time['feature'].to_numpy().astype(int)

        features_at_time['thermal_feature_id'] = features_at_time['feature']
        features_at_time['thermal_cell_id'] = features_at_time['cell']
    elif stats_level=='cloud':
        features_at_time_temp = features_at_time[['frame','feature','cell','time','timestr','cloud_feature_id']].copy()

        features_at_time = features_at_time_temp.groupby('cloud_feature_id')[['frame','time','timestr']].first()
        features_at_time['num_thermals_per_cloud'] = features_at_time_temp.groupby('cloud_feature_id').feature.count()
        features_at_time['thermal_features'] = features_at_time_temp.groupby('cloud_feature_id').feature.unique()
        features_at_time['thermal_cells'] = features_at_time_temp.groupby('cloud_feature_id').cells.unique()

        features_at_time = features_at_time.reset_index()

        feature_ids = features_at_time['cloud_feature_id'].to_numpy().astype(int)
    elif stats_level=='updraft':
        features_at_time_temp = features_at_time[['frame','feature','cell','time','timestr','updraft_feature_id']].copy()

        features_at_time = features_at_time_temp.groupby('updraft_feature_id')[['frame','time','timestr']].first()
        features_at_time['num_thermals_per_updraft'] = features_at_time_temp.groupby('updraft_feature_id').feature.count()
        features_at_time['thermal_features'] = features_at_time_temp.groupby('updraft_feature_id').feature.unique()
        features_at_time['thermal_cells'] = features_at_time_temp.groupby('updraft_feature_id').cells.unique()
        
        features_at_time = features_at_time.reset_index()

        feature_ids = features_at_time['updraft_feature_id'].to_numpy().astype(int)
    else:
        print(f"Stats level {stats_level} is not a valid option. Select feature, cloud, or updraft.")

    mask_array = mask.isel(time=0).values

    # first all the geometry stats

    # pull out only the points corresponding to nonzero values in mask
    # i find the logic a bit more convoluted BUT it's much faster because our arrays are very sparse
    z_idx, y_idx, x_idx = np.nonzero(mask_array)
    feature_id_per_pixel = mask_array[z_idx, y_idx, x_idx]

    alt_km = mask.altitude.values[1:] / 1000
    dz_profile = np.diff(mask.altitude.values) / 1000
    dxy = find_dxy_from_grid_level(grid_level=3) / 1000

    # save the min and max extent of each feature in all dimensions
    for name, pos in zip(['x','y','z'], [x_idx,y_idx,z_idx]):
        min_pos, max_pos = get_extent(pos, feature_id_per_pixel, feature_ids, len(mask[name])-1)
        features_at_time[f'{name}_extent'] =  [(x,y) for x,y in zip(min_pos, max_pos)]

        if name == 'z':
            features_at_time['top_altitude'] =  alt_km[max_pos]
            features_at_time['bottom_altitude'] = alt_km[min_pos]
            features_at_time['depth'] = features_at_time['top_altitude'] - features_at_time['bottom_altitude']

    # save the volume of each feature

    # get the volume corresponding to our list of pixels
    pixel_volumes = dz_profile[z_idx] * dxy * dxy

    # sum individual pixel volumes by feature ID
    volumes = scipy.ndimage.sum(pixel_volumes, labels=feature_id_per_pixel, index=feature_ids)

    features_at_time['volume'] = volumes
    features_at_time['area_verticalmean'] = volumes / features_at_time['depth']
    # assuming the thermal is a perfect cylinder, what would be the area?
    features_at_time['equivalent_radius_verticalmean'] = (features_at_time['area_verticalmean'] / np.pi) ** 0.5

    # save the footprint of each feature: looking down vertically on each mask, how many points does it span overall?

    features_at_time['footprint'] = get_footprint(y_idx,x_idx,feature_id_per_pixel, feature_ids) * dxy*dxy

    # read in data
    data = read_data(filepath, coords=coords, variables=['vertical_velocity','density']).isel(time=0)
    data = subset_data(data)
    data = interpolate_w(data, meta['model_type'])

    data = data.assign(mass_flux = (data.density * data.vertical_velocity))
    
    # now pull the vertical velocity corresponding
    vertical_velocity = data.vertical_velocity.values
    w_features = vertical_velocity[z_idx, y_idx, x_idx]

    features_at_time['vertical_velocity_max'] = scipy.ndimage.maximum(w_features, labels=feature_id_per_pixel, index=feature_ids)
    features_at_time['vertical_velocity_mean'] = scipy.ndimage.mean(w_features, labels=feature_id_per_pixel, index=feature_ids)
    features_at_time['altitude_of_vertical_velocity_max'] = get_altitude_of_max(w_features, z_idx,alt_km, feature_id_per_pixel, feature_ids) 

    # now pull the vertical velocity corresponding
    cmf = data.mass_flux.values
    cmf_features = cmf[z_idx, y_idx, x_idx]

    features_at_time['mass_flux_max'] = scipy.ndimage.maximum(cmf_features, labels=feature_id_per_pixel, index=feature_ids)
    features_at_time['mass_flux_mean'] = scipy.ndimage.mean(cmf_features, labels=feature_id_per_pixel, index=feature_ids)
    features_at_time['mass_flux_integrated'] = scipy.ndimage.sum(cmf_features*pixel_volumes, labels=feature_id_per_pixel, index=feature_ids)
    features_at_time['altitude_of_mass_flux_max'] = get_altitude_of_max(cmf_features, z_idx,alt_km, feature_id_per_pixel, feature_ids) 

    return(features_at_time)
