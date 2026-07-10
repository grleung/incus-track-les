# incus_track_les/data_readers.py

from __future__ import annotations  
from pathlib import Path
from typing import List, Union
import numpy as np
import xarray as xr
import gc

from incus_track_les.paths import find_model_metadata
from incus_track_les.varnames_config import DIM_MAPPINGS, VAR_MAPPINGS, DIMS, GRID_SPACING_MAPPINGS

def get_filepaths(directory: Path,grid_level: int=3) -> list[Path]:
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

    meta = find_model_metadata(path) 

    # get grid number
    grid_level = meta['grid_level']

    # get the spatial size of each dimension
    with xr.open_dataset(path, engine='h5netcdf', phony_dims='access') as ds:
        dim_lengths = {v: ds.sizes.get(k) for k, v in DIM_MAPPINGS[meta['model_type']].items()}

        # read lat/lon grid once
        lat = ds['GLAT'].values
        lon = ds['GLON'].values

    dim_lengths.update({'x_stag':dim_lengths['x'],
                        'y_stag':dim_lengths['y'],
                        'z_stag':dim_lengths['z']})
    
    coords = {}
    for dim_name, dim_meta in DIMS.items():
        length = dim_lengths[dim_name]
        short_name = dim_meta['short_name']
        header_name = dim_meta['rams_header_name']
        
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
                'units': dim_meta['units'], 
                'long_name': dim_meta['long_name'],
                'standard_name': dim_meta['standard_name']
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

def read_rams_data(path:Path, variables:list[str])->xr.Dataset | xr.DataArray:
    """Read RAMS data into a standard format that can be read by tobac

    Args:
        path (Path): full path to file
        variables (list[str], optional): list of variables to return, see: RAMS documentation for variables available.
    Returns:
        [xr.Dataset,xr.DataArray]: dataset or dataarray with RAMS data
    """

    meta = find_model_metadata(path)    

    ds = xr.open_dataset(path,
                         chunks=-1, # don't chunk so dask task graph stays managable; tobac uses the whole grid at once anyway
                         engine='h5netcdf',
                         phony_dims='access')
    
    # select only the variables we need
    ds = ds[variables]

    # rename dimensions
    ds = ds.rename_dims(DIM_MAPPINGS[meta['model_type']])

    # rename the staggered variables manually
    # note that in the Arakawa-C grid, W is staggered in vertical but V and U are on same level as the scalar variables (and same for x and y directions)
    for var in variables:
        if var in ["WP"]:  
            ds[var] = ds[var].rename({"z": "z_stag"})
        elif var in ["VP"]:
            ds[var] = ds[var].rename({"y": "y_stag"})
        elif var in ["UP"]:
            ds[var] = ds[var].rename({"x": "x_stag"})
    
    return(ds)

def read_wrf_coords(path:Path) -> dict:
    """Returns a dictionary with WRF x,y,z coordinates in index units and physical units
    and latitude/longitude. This only needs to be run once per run/grid (not every timestep) 

    Args:
        path (Path): full path to file

    Returns:
        dict: coordinate dictionary
    """    

    import h5netcdf
    
    meta = find_model_metadata(path) 
    
    with h5netcdf.File(path, 'r') as f: # using this is faster than xarray bc of attributes I think?
        # get dimension sizes
        dim_lengths = {v: f.dimensions[k].size for k, v in DIM_MAPPINGS[meta['model_type']].items()}

        # get grid size
        dx = f.attrs['DX']

        # note WRF files have a time coord already
        lat = f['XLAT'][0,:,:] 
        lon = f['XLONG'][0,:,:]

        # TODO: test this decision point: WRF uses fixed mass grid, such that the physical altitude changes with time
        # I think we can just take the base state geopoptential at center of domain at first timestep
        # and subtract surface altitude; initial test show this doesnt vary hugely within domains

        # compute physical altitude from geopotential

        phb = f['PHB'][0,:,dim_lengths['y']//2,dim_lengths['x']//2]
        hgt = f['HGT'][0,dim_lengths['y']//2,dim_lengths['x']//2]
        
        altitude_stag = ((phb/9.81) - hgt)
        altitude = 0.5 * (altitude_stag[:-1] + altitude_stag[1:])

    coords = {}
    for dim_name, dim_meta in DIMS.items():
        length = dim_lengths[dim_name]
        short_name = dim_meta['short_name']

        # assign grid index range
        coords[dim_name] = np.arange(length)

        if 'stag' in dim_name:
            if 'z' not in dim_name:
                # for x and y, just generate physical distances based on indices
                # note dx and dy are equal for these simulations
                coords[short_name] = (
                    dim_name, 
                    np.arange(length)* dx, 
                    {
                        'units': dim_meta['units'], 
                        'long_name': dim_meta['long_name'],
                        'standard_name': dim_meta['standard_name']
                    }
                )
            else:
                coords[short_name] = (
                    dim_name, 
                    altitude_stag, 
                    {
                        'units': dim_meta['units'], 
                        'long_name': dim_meta['long_name'],
                        'standard_name': dim_meta['standard_name']
                    }
                )
        else:
            if 'z' not in dim_name:
                # for x and y, just base on coords
                coords[short_name] = (
                    dim_name, 
                    (np.arange(length)+0.5)* dx, 
                    {
                        'units': dim_meta['units'], 
                        'long_name': dim_meta['long_name'],
                        'standard_name': dim_meta['standard_name']
                    }
                )
            else:
                coords[short_name] = (
                    dim_name, 
                    altitude, 
                    {
                        'units': dim_meta['units'], 
                        'long_name': dim_meta['long_name'],
                        'standard_name': dim_meta['standard_name']
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

    
    
def read_wrf_data(path:Path, variables:list[str])->Union[xr.Dataset,xr.DataArray]:
    """Read WRF data into a standard format that can be read by tobac

    Args:
        path (Path): full path to file
        variables (list[str], optional): list of variables to return, see WRF documentation.
    Returns:
        Union[xr.Dataset,xr.DataArray]: dataset or dataarray with WRF data
    """   
    meta = find_model_metadata(path) 
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
    ds = ds.rename_dims({k: v for k, v in DIM_MAPPINGS[meta['model_type']].items() if k in ds.dims})

    return(ds)

def read_coords(filepath: Path) -> dict:
    meta = find_model_metadata(filepath)

    if meta['model_type'] == 'RAMS':
        return(read_rams_coords(filepath))
    elif meta['model_type'] == 'WRF':
        return(read_wrf_coords(filepath))
    
def read_data(filepath: Path, coords: Union[xr.Dataset,xr.DataArray], variables: list[str]) -> Union[xr.Dataset,xr.DataArray]:
    meta = find_model_metadata(filepath)
    model = meta['model_type']

    var_map = VAR_MAPPINGS[model]

    # pull out the variables needed in model naming convention
    variables_model_name = []

    for v in variables:
        if isinstance(var_map[v],tuple):
            variables_model_name.extend(var_map[v][0])
        else:
            variables_model_name.append(var_map[v])

    if model == 'RAMS':
        ds = read_rams_data(filepath, variables=variables_model_name)
    elif model == 'WRF':
        ds = read_wrf_data(filepath, variables=variables_model_name)
    else:
        raise ValueError(f"Unrecognized model type for {filepath}: {model}")

    # assign coordinates
    ds = ds.assign_coords(coords)

    for v in variables:
        if isinstance(var_map[v],tuple):
            input_vars, formula = var_map[v]

            ds[v] = formula(ds)
        else:
            ds[v] = ds[var_map[v]]

    vars_to_drop = [v for v in ds.data_vars if v not in variables]
    
    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)
        
        gc.collect()

    # if only one variable, just return the dataarray
    if len(variables)==1:
        ds = ds[variables[0]]

    # last let's add time information
    time = meta['time']
    ds = ds.expand_dims(time=[time])

    return(ds)
    
def subset_data(ds: xr.DataArray, nbound:int = 25) -> xr.DataArray:
    """Subsets model data by removing sponge zone. on G3, 25 points around the boundary are being nudged towards G2 input, so we should remove them to only analyze interior of domain.

    Args:
        ds (xr.DataArray): Input data
        nbound (int, optional): Number of points to remove. Defaults to 25.
    """      

    for dim in ['x','x_stag','y','y_stag']:
        if dim in list(ds.dims):
            ds = ds.isel({dim:slice(nbound,-nbound)})


    return(ds)

def find_dxy_from_grid_level(grid_level):
    return(GRID_SPACING_MAPPINGS.get(grid_level))

def interpolate_w(data, model_type,drop_others=False):
    from xgcm import Grid

    is_dataarray = isinstance(data, xr.DataArray)
    if is_dataarray:
        data = data.to_dataset(name='vertical_velocity')

    if model_type=='RAMS':
        grid = Grid(data,
            coords={'z': {'center':'z','right':'z_stag'}}, 
            periodic=False,
            autoparse_metadata=False)
    elif model_type=='WRF':
        grid = Grid(data,
            coords={'z': {'center':'z','outer':'z_stag'}}, 
            periodic=False,
            autoparse_metadata=False)

    w_unstaggered = grid.interp(data.vertical_velocity, axis='z', boundary='extend')

    data['vertical_velocity'] = w_unstaggered

    if model_type=='RAMS':
        # remove the ghost point below surface
        data = data.isel(z=slice(1,None))

    dims_to_drop = {'x_stag', 'y_stag', 'z_stag'} & set(data.dims)
    if dims_to_drop:
        data = data.drop_dims(dims_to_drop)
        
    if drop_others:
        data = data['vertical_velocity']

    return(data)