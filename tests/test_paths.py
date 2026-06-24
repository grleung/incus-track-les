# incus_track_les/tests/test_paths.py

import pytest
import pandas as pd
from pathlib import Path

from incus_track_les.paths import (find_model_metadata,
    find_run_name_from_file_path
)

def test_rams_path_parsing():
    """tests that RAMS file parsers work correctly for getting run name, grid level, and timestep
    """    

    # fake RAMS path
    rams_path = Path(
        "/monsoon/MODEL/LES_MODEL_DATA/V1/WPO1.1-R-V1/G3/out/a-L-2018-08-28-040000-g3.h5"
    )
    
    run_name = find_run_name_from_file_path(rams_path)
    meta = find_model_metadata(rams_path)

    assert run_name == "WPO1.1-R-V1"
    assert meta['model_type'] == 'RAMS'
    assert meta['grid_level'] == 3
    assert meta['time'] == pd.Timestamp("2018-08-28 04:00:00")


def test_wrf_path_parsing():
    """tests that WRF file parsers work correctly for getting run name, grid level, and timestep
    """    

    wrf_path = Path(
        "/monsoon/MODEL/LES_MODEL_DATA/V1/WPO1.1-WM-V1/G3/wrfout_d03_2018-08-28_04_00_00"
    )
    
    run_name = find_run_name_from_file_path(wrf_path)
    meta = find_model_metadata(wrf_path)
    
    assert run_name == "WPO1.1-WM-V1"
    assert meta['model_type'] == 'WRF'
    assert meta['grid_level'] == 3
    assert meta['time'] == pd.Timestamp("2018-08-28 04:00:00")