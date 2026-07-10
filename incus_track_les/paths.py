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

def find_model_metadata(path: Path) -> dict:
    """Returns model metadata for given path

    Args:
        path (Path): path to either RAMS or WRF output file

    Returns:
        dict: dictionary containing model type, grid level, and time for specified path
    """
    run_name = find_run_name_from_file_path(path)
    filename = path.name

    if '-R-' in run_name:
        return({'model_type':'RAMS',
                'grid_level': int(filename[-4:-3]),
                'time':pd.to_datetime(filename[4:-6])})
    elif ('-WM-' in run_name) or ('-WT-' in run_name):
        return({'model_type':'WRF',
                'grid_level': int(filename[9:10]),
                'time':pd.to_datetime(filename[11:],format='%Y-%m-%d_%H_%M_%S')})
    else:
        raise ValueError(f"Unknown model type for run name: {run_name}.")

def find_file_path_from_time(parent_dir, time, grid_level=3):
    run_name = find_run_name_from_file_path(parent_dir)

    if '-R-' in run_name:
        return(Path(parent_dir, f"G3/out_30s/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}-g{grid_level}.h5"))
    elif ('-WM-' in run_name) or ('-WT-' in run_name):
        return(Path(parent_dir, f"G3/wrfout_d{str(grid_level).zfill(2)}_{time.strftime('%Y-%m-%d_%H_%M_%S')}"))