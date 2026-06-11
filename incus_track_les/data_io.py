# incus_track_les/data_io.py

from pathlib import Path
from typing import List

def get_filepaths(directory: Path,grid_level: int='3') -> List[Path]:
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
        filepaths= sorted(directory.glob(f"G3/out/a-L-*-{grid_str}.h5"))
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

