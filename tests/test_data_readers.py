# incus_track_les/tests/test_readers.py

import pytest
import numpy as np
import xarray as xr
import dask.array as da
from pathlib import Path

from incus_track_les.data_readers import read_coords, read_data

@pytest.fixture
def dummy_rams_file(tmp_path):

    Path(tmp_path, 'V1','WPO1.1-R-V1').mkdir(parents=True, exist_ok=True)
    
    filepath = Path(tmp_path, 'V1','WPO1.1-R-V1', 'a-L-2019-08-04-040000-g3.h5')
    headpath = Path(tmp_path,  'V1','WPO1.1-R-V1','a-L-2019-08-04-040000-head.txt')

    dummy_header_content = (
        "__xtn03\n2\n0.0\n500.0\n"      # Center X points
        "__ytn03\n2\n0.0\n500.0\n"      # Center Y points
        "__ztn03\n2\n10.0\n50.0\n"      # Center Z points
        "__xmn03\n2\n250.0\n750.0\n"    # Staggered X points
        "__ymn03\n2\n250.0\n750.0\n"    # Staggered Y points
        "__zmn03\n2\n0.0\n30.0\n"       # Staggered Z points
    )
    headpath.write_text(dummy_header_content)
    
    dummy_data = np.zeros((2, 2, 2), dtype=np.float32)
    dummy_lat_lon = np.ones((2, 2), dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "WP": (("phony_dim_3", "phony_dim_1", "phony_dim_2"), dummy_data),
            "UP": (("phony_dim_3", "phony_dim_1", "phony_dim_2"), dummy_data),
            "GLAT": (("phony_dim_1", "phony_dim_2"), dummy_lat_lon),
            "GLON": (("phony_dim_1", "phony_dim_2"), dummy_lat_lon),
        }
    )
    
    ds.to_netcdf(filepath, engine="h5netcdf")
    return filepath


def test_rams_data_reader(dummy_rams_file):

    coords = read_coords(dummy_rams_file)

    ds_processed = read_data(dummy_rams_file, coords=coords, variables=["vertical_velocity",'zonal_velocity'])
    
    # check that WP is staggered correctly
    assert "z_stag" in ds_processed["vertical_velocity"].dims
    assert "z" not in ds_processed["vertical_velocity"].dims

    # check that UP (unstaggered) is not staggered along z
    assert "z_stag" not in ds_processed["zonal_velocity"].dims
    assert "z" in ds_processed["zonal_velocity"].dims
    
    # check lat/lon are present
    assert "latitude" in ds_processed.coords
    assert "longitude" in ds_processed.coords

    # check altitude is present
    assert "altitude_stag" in ds_processed["vertical_velocity"].coords
    
    # check time dimensions appended properly
    assert "time" in ds_processed.dims
    assert ds_processed.sizes["time"] == 1

    # check data is a dask array  but coords are numpy
    assert isinstance(ds_processed["vertical_velocity"].data, da.Array)
    assert isinstance(coords["latitude"][1], np.ndarray)

@pytest.fixture
def dummy_wrf_file(tmp_path):

    Path(tmp_path, 'V1','WPO1.1-WM-V1').mkdir(parents=True, exist_ok=True)
    
    filepath = Path(tmp_path, 'V1','WPO1.1-WM-V1', 'wrfout_d03_2019-08-04_04_00_00')

    # Staggered dimensions have an extra element (+1) along their specific axis
    dummy_data_unstag = np.zeros((1, 2, 2, 2), dtype=np.float32)
    dummy_data_w_stag = np.zeros((1, 3, 2, 2), dtype=np.float32)
    dummy_data_v_stag = np.zeros((1, 2, 3, 2), dtype=np.float32)
    dummy_lat_lon = np.ones((1, 2, 2), dtype=np.float32)
    
    # Geopotential arrays required for dynamic physical height calculations
    dummy_phb = np.ones((1, 3, 2, 2), dtype=np.float32)
    dummy_hgt = np.zeros((1, 2, 2), dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "W": (("Time", "bottom_top_stag", "south_north", "west_east"), dummy_data_w_stag),
            "U": (("Time", "bottom_top", "south_north", "west_east_stag"), dummy_data_unstag),
            "V": (("Time", "bottom_top", "south_north_stag", "west_east"), dummy_data_v_stag),
            "XLAT": (("Time", "south_north", "west_east"), dummy_lat_lon),
            "XLONG": (("Time", "south_north", "west_east"), dummy_lat_lon),
            "PHB": (("Time", "bottom_top_stag", "south_north", "west_east"), dummy_phb),
            "HGT": (("Time", "south_north", "west_east"), dummy_hgt),
        },
        attrs={"DX": 500.0}
    )
    
    ds.to_netcdf(filepath, engine="h5netcdf")
    return filepath


def test_wrf_data_reader(dummy_wrf_file):

    coords = read_coords(dummy_wrf_file)

    ds_processed = read_data(dummy_wrf_file, coords=coords, variables=["vertical_velocity",'zonal_velocity'])
    
    # check that WP is staggered correctly
    assert "z_stag" in ds_processed["vertical_velocity"].dims
    assert "z" not in ds_processed["vertical_velocity"].dims

    # check that UP (unstaggered) is not staggered along z
    assert "z_stag" not in ds_processed["zonal_velocity"].dims
    assert "z" in ds_processed["zonal_velocity"].dims
    
    # check lat/lon are present
    assert "latitude" in ds_processed.coords
    assert "longitude" in ds_processed.coords

    # check altitude is present
    assert "altitude_stag" in ds_processed["vertical_velocity"].coords
    
    # check time dimensions appended properly
    assert "time" in ds_processed.dims
    assert ds_processed.sizes["time"] == 1

    # check data is a dask array  but coords are numpy
    assert isinstance(ds_processed["vertical_velocity"].data, da.Array)
    assert isinstance(coords["latitude"][1], np.ndarray)