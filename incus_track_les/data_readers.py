# incus_track_les/data_io.py

from __future__ import annotations  
from pathlib import Path
from typing import List, Union
import numpy as np
import xarray as xr

from incus_track_les.paths import find_grid_level_from_file_path, find_time_from_file_path

RENAMED_RAMS_DIMS = {
    "phony_dim_3": "z",
    "phony_dim_1": "y",
    "phony_dim_2": "x",
}

RENAMED_WRF_DIMS = {
    'west_east': 'x',
    'south_north': 'y',
    'bottom_top': 'z',
    'west_east_stag': 'x_stag',
    'south_north_stag': 'y_stag',
    'bottom_top_stag': 'z_stag'
}

DIMS = {
    'x': {
        'short_name': 'xdist', 
        'rams_header_name': '__xtn', 
        'long_name': 'E-W distance', 
        'standard_name': 'projection_x_coordinate', 
        'units': 'meters'
    },
    'y': {
        'short_name': 'ydist', 
        'rams_header_name': '__ytn', 
        'long_name': 'N-S distance', 
        'standard_name': 'projection_y_coordinate',  
        'units': 'meters'
    },
    'z': {
        'short_name': 'altitude', 
        'rams_header_name': '__ztn', 
        'long_name': 'altitude above ground level', 
        'standard_name': 'altitude', 
        'units': 'meters'
    },
    'x_stag': {
        'short_name': 'xdist_stag', 
        'rams_header_name': '__xmn', 
        'long_name': 'E-W distance', 
        'standard_name': 'projection_x_coordinate', 
        'units': 'meters'
    },
    'y_stag': {
        'short_name': 'ydist_stag', 
        'rams_header_name': '__ymn', 
        'long_name': 'N-S distance', 
        'standard_name': 'projection_y_coordinate', 
        'units': 'meters'
    },
    'z_stag': {
        'short_name': 'altitude_stag', 
        'rams_header_name': '__zmn', 
        'long_name': 'altitude above ground level', 
        'standard_name': 'altitude', 
        'units': 'meters'
    },
}

def get_filepaths(directory: Path,grid_level: int=3) -> List[Path]:
    """Returns a sorted list of filepaths for a given run directory. Detects whether the simulation is RAMS or WRF. 

    Args:
        dir (Path): base simulation directory (e.g., "/monsoon/MODEL/LES_MODEL_DATA/V1/WPO1.1-WM-V1/")
        grid_level (int, optional): model grid to return (1, 2, or 3). Defaults to 3 (finest grid).

    Returns:
        List: sorted list of Path objects pointing to output files

    Raises: 
        ValueError: if directory name doesn't match known RAMS/WRF naming convention
    """    

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"The simulation directory does not exist: '{directory}'")
    
    run_name = directory.name # takes just last folder
    
    if "-R-" in run_name:
        grid_str = f"g{grid_level}"
        filepaths= sorted(directory.glob(f"G3/out_30s/a-L-*-{grid_str}.h5"))
    elif '-WM-' in run_name or '-WT-' in run_name:
        grid_str = f"d0{grid_level}"
        filepaths= sorted(directory.glob(f'G3/wrfout_{grid_str}*'))
    else:
        raise ValueError(f"Could not auto-detect model type from directory name: '{run_name}'. "
                    f"Expected folder name to contain '-R-', '-WM-', or '-WT-'."
                )

    if not filepaths:
        raise FileNotFoundError(
            f"Directory found, but no model files matched grid level {grid_level} in '{directory}'."
        )

    return filepaths

def read_rams_header(path: Path, var: str="__ztn03") -> np.ndarray:
    """read a variable from RAMS header file
    
    Args:
        path (Path): full path to file
        var (str, optional): name of header variable. Defaults to "__ztn03".

    Returns:
        np.ndarray: array of values for specified variable
    """    
    header_file_name = path.with_name(f"{path.name[:-5]}head.txt")

    with open(header_file_name) as f:
        mylist = f.read().splitlines()
    ix = mylist.index(var)
    numlines = int(mylist[ix + 1])
    coord = mylist[ix + 2 : ix + 2 + numlines]
    coord = np.array([float(x) for x in coord])

    return coord

def read_rams_coords(path:Path) -> dict:
    """Returns a dictionary with RAMS x,y,z coordinates in index units and physical units
    and latitude/longitude. This only needs to be run once per run/grid (not every timestep) 

    Args:
        path (Path): _full path to file

    Returns:
        dict: coordinate dictionary
    """    
    # get grid number
    grid_level = find_grid_level_from_file_path(path)

    # get the spatial size of each dimension
    with xr.open_dataset(path, engine='h5netcdf', phony_dims='access') as ds:
        dim_lengths = {v: ds.sizes.get(k) for k, v in RENAMED_RAMS_DIMS.items()}

        # read lat/lon grid once
        lat = ds['GLAT'].values
        lon = ds['GLON'].values

    dim_lengths.update({'x_stag':dim_lengths['x'],
                        'y_stag':dim_lengths['y'],
                        'z_stag':dim_lengths['z']})
    
    coords = {}
    for dim_name, meta in DIMS.items():
        length = dim_lengths[dim_name]
        short_name = meta['short_name']
        header_name = meta['rams_header_name']
        
        # assign grid index range
        coords[dim_name] = range(length)
        
        # pull header file data to get corresponding physical units
        header_var = f"{header_name}{str(grid_level).zfill(2)}"
        header_data = read_rams_header(path, var=header_var)
        
        # attach metadata
        coords[short_name] = (
            dim_name, 
            header_data, 
            {
                'units': meta['units'], 
                'long_name': meta['long_name'],
                'standard_name': meta['standard_name']
            }
        )
        


    # add lat/lon to coords lat
    coords['latitude'] = (('y', 'x'), 
                          lat, 
                          {'units':'degrees_north',
                           'long_name':'latitude',
                           'standard_name':'latitude'})
    coords['longitude'] = (('y', 'x'), 
                           lon,
                           {'units':'degrees_east',
                           'long_name':'longitude',
                           'standard_name':'longitude'})

    return(coords)

def read_rams_data(path:Path, coords:dict, variables:list[str]=['WP'])->Union[xr.Dataset,xr.DataArray]:
    """Read RAMS data into a standard format that can be read by tobac

    Args:
        path (Path): full path to file
        coords (dict): coordinate dictionary from read_rams_coords
        variables (list[str], optional): list of variables to return, see: RAMS documentation for variables available. Defaults to ['WP'].
    Returns:
        Union[xr.Dataset,xr.DataArray]: dataset or dataarray with RAMS data
    """    

    ds = xr.open_dataset(path,
                         chunks=-1, # don't chunk so dask task graph stays managable; tobac uses the whole grid at once anyway
                         engine='h5netcdf',
                         phony_dims='access')
    
    # select only the variables we need
    ds = ds[variables]

    
    # rename dimensions
    ds = ds.rename_dims(RENAMED_RAMS_DIMS)

    # rename the staggered variables manually
    # note that in the Arakawa-C grid, W is staggered in vertical but V and U are on same level as the scalar variables (and same for x and y directions)
    for var in variables:
        if var in ["WP"]:  #note: in RAMS, WP is the vertical velocity at this timestep (WC says "current" in docs but this is actually the unfiltered future value for next time step initial conditions; see the Subroutine predict for details)
            ds[var] = ds[var].rename({"z": "z_stag"})
        elif var in ["VP"]:
            ds[var] = ds[var].rename({"y": "y_stag"})
        elif var in ["UP"]:
            ds[var] = ds[var].rename({"x": "x_stag"})

    ds = ds.assign_coords(coords)

    # if only one variable, just return the dataarray
    if len(variables)==1:
        ds = ds[variables[0]]

    # last let's add time information
    time = find_time_from_file_path(path)
    ds = ds.expand_dims(time=[time])

    return(ds)



def read_wrf_coords(path:Path) -> dict:
    import h5netcdf
    
    with h5netcdf.File(path, 'r') as f: # using this is faster than xarray bc of attributes I think?
        # get dimension sizes
        dim_lengths = {v: f.dimensions[k].size for k, v in RENAMED_WRF_DIMS.items()}

        # get grid size
        dx = f.attrs['DX']

        # note WRF files have a time coord already
        lat = f['XLAT'][0,:,:] 
        lon = f['XLONG'][0,:,:]

        # TODO: test this decision point: WRF uses fixed mass grid, such that the physical altitude changes with time
        # I think we can just take the base state geopoptential at center of domain at first timestep
        # and subtract surface altitude; initial test show this doesnt vary hugely within domains
        # but for some reason physical altitudes end up being much bigger than RAMS (e.g., max alt reaches 33km while in RAMS only ~26km)
        # compute physical altitude from geopotential

        phb = f['PHB'][0,:,dim_lengths['y']//2,dim_lengths['x']//2]
        hgt = f['HGT'][0,dim_lengths['y']//2,dim_lengths['x']//2]
        
        altitude_stag = ((phb/9.81) - hgt)
        altitude = 0.5 * (altitude_stag[:-1] + altitude_stag[1:])

    coords = {}
    for dim_name, meta in DIMS.items():
        length = dim_lengths[dim_name]
        short_name = meta['short_name']

        # assign grid index range
        coords[dim_name] = np.arange(length)

        if 'stag' in dim_name:
            if 'z' not in dim_name:
                # for x and y, just generate physical distances based on indices
                # note dx and dy are equal for these simulations
                coords[short_name] = (
                    dim_name, 
                    np.arange(length)* dx, #TODO: check grid stagger is right
                    {
                        'units': meta['units'], 
                        'long_name': meta['long_name'],
                        'standard_name': meta['standard_name']
                    }
                )
            else:
                coords[short_name] = (
                    dim_name, 
                    altitude_stag, 
                    {
                        'units': meta['units'], 
                        'long_name': meta['long_name'],
                        'standard_name': meta['standard_name']
                    }
                )
        else:
            if 'z' not in dim_name:
                # for x and y, just base on coords
                coords[short_name] = (
                    dim_name, 
                    (np.arange(length)+0.5)* dx, 
                    {
                        'units': meta['units'], 
                        'long_name': meta['long_name'],
                        'standard_name': meta['standard_name']
                    }
                )
            else:
                coords[short_name] = (
                    dim_name, 
                    altitude, 
                    {
                        'units': meta['units'], 
                        'long_name': meta['long_name'],
                        'standard_name': meta['standard_name']
                    }
                )
    # add lat/lon to coords lat
    coords['latitude'] = (('y', 'x'), 
                          lat, 
                          {'units':'degrees_north',
                           'long_name':'latitude',
                           'standard_name':'latitude'})
    coords['longitude'] = (('y', 'x'), 
                           lon,
                           {'units':'degrees_east',
                           'long_name':'longitude',
                           'standard_name':'longitude'})
    
    return(coords)

    
    

def read_wrf_data(path:Path, coords:dict, variables:list[str]=['W'])->Union[xr.Dataset,xr.DataArray]:
    """Read WRF data into a standard format that can be read by tobac

    Args:
        path (Path): full path to file
        coords (dict): coordinate dictionary from read_wrf_coords
        variables (list[str], optional): list of variables to return, see WRF documentation. Defaults to ['W'].
    Returns:
        Union[xr.Dataset,xr.DataArray]: dataset or dataarray with WRF data
    """    
    ds = xr.open_dataset(path,
                         chunks=-1, # don't chunk so dask task graph stays managable; tobac uses the whole grid at once anyway
                         engine='h5netcdf',
                         decode_times=False,
                         decode_cf=False,
                         )
    
    # get rid of all the metadata for speed since we don't need it for tobac
    ds.attrs = {}

    # select only the variables we need
    ds = ds[variables]

    # remove time dimension (this seems to make aligning coordinates faster later)
    if 'Time' in ds.dims:
        ds = ds.squeeze('Time')

    # rename dimensions
    ds = ds.rename_dims({k: v for k, v in RENAMED_WRF_DIMS.items() if k in ds.dims})

    ds = ds.assign_coords(coords)

    # if only one variable, just return the dataarray
    if len(variables)==1:
        ds = ds[variables[0]]
    
    # last let's add time information
    time = find_time_from_file_path(path)
    ds = ds.expand_dims(time=[time])

    return(ds)


def subset_data(ds: xr.DataArray, nbound:int = 25):
    """Subsets model data by removing sponge zone. on G3, 25 points around the boundary are being nudged towards G2 input, so we should remove them to only analyze interior of domain.

    Args:
        ds (xr.DataArray): Input data
        nbound (int, optional): Number of points to remove. Defaults to 25.
    """      

    for dim in ['x','x_stag','y','y_stag']:
        if dim in list(ds.dims):
            ds = ds.isel({dim:slice(nbound,-nbound)})


    return(ds)