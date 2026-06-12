# incus_track_les/paths.py

from pathlib import Path
import pandas as pd

MODEL_DATA_DIR = Path('/monsoon/MODEL/LES_MODEL_DATA/V1')

TRACK_DIR = Path ('/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1.2')

def find_run_name_from_file_path(path: Path) -> str:
    """Returns the run name for given path

    Args:
        path (Path): path to either RAMS or WRF output file

    Returns:
        str: string corresponding to run name
    """
    
    path_parts = Path(path).parts
    # since RAMS and WRF have different file structures, note that the run name is always after the 'V1' in file path
    return(path_parts[path_parts.index("V1") + 1])


def find_grid_level_from_file_path(path: Path) -> int:
    """Returns the grid level for given path

    Args:
        path (Path): path to either RAMS or WRF output file

    Returns:
        int: integer corresponding to grid level
    """

    from . import find_run_name_from_file_path

    run_name = find_run_name_from_file_path(path)

    if "-R-" in run_name:
        return(int(path.name[-4:-3]))
    elif '-WM-' in run_name or '-WT-' in run_name:
        return(int(path.name[9:10]))
    

def find_time_from_file_path(path: Path) -> pd.Timestamp:
    """Returns the UTC time for given path

    Args:
        path (Path): path to either RAMS or WRF output file

    Returns:
        pd.Timestamp: timestamp with file time
    """

    #from . import find_run_name_from_file_path

    run_name = find_run_name_from_file_path(path)

    if "-R-" in run_name:
        return(pd.to_datetime(path.name[4:-6]))
    elif '-WM-' in run_name or '-WT-' in run_name:
        return(pd.to_datetime(path.name[11:],format='%Y-%m-%d_%H_%M_%S'))
    
    