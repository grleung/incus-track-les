# incus_track_les/tests/test_readers.py

import pytest
import numpy as np
import xarray as xr
import dask.array as da
from pathlib import Path

from incus_track_les.data_io import read_rams_coords, read_rams_data

@pytest.fixture
def dummy_rams_file(tmp_path):

    filepath = Path(tmp_path, 'a-L-2019-08-04-04000000-g3.h5')
    headpath = Path(tmp_path, 'a-L-2019-08-04-04000000-head.txt')

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

    coords = read_rams_coords(dummy_rams_file)

    ds_processed = read_rams_data(dummy_rams_file, coords=coords, variables=["WP", "GLAT"])
    
    # check that WP is staggered correctly
    assert "z_stag" in ds_processed["WP"].dims
    assert "z" not in ds_processed["WP"].dims
    
    # check unstaggered dims aren't staggered
    assert "y" in ds_processed["GLAT"].dims
    assert "y_stag" not in ds_processed["GLAT"].dims

    # check lat/lon are present
    assert "latitude" in ds_processed.coords
    assert "longitude" in ds_processed.coords

    # check altitude is present
    assert "altitude_stag" in ds_processed["WP"].coords
    
    # check time dimensions appended properly
    assert "time" in ds_processed.dims
    assert ds_processed.sizes["time"] == 1

    # check data is a dask array  but coords are numpy
    assert isinstance(ds_processed["WP"].data, da.Array)
    assert isinstance(coords["latitude"][1], np.ndarray)