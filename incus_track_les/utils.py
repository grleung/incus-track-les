# incus_track_les/utils.py

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import sys
from pathlib import Path
import pandas as pd

import hdf5plugin
plugin_path = hdf5plugin.PLUGIN_PATH

def init_slurm_downdraft(
    jobs: int = 30, 
    job_name: str = "tracking-test", 
    dashboard_port: int = 10101,
    walltime: str = "24:00:00",
    memory: str = "40GB" 
) -> Client:
    """Standard initialization of SLURMCluster for downdraft

    Args:
        jobs (int, optional): number of jobs to initialize. Defaults to 30.
        job_name (str, optional): name of job. Defaults to "tracking-test".
        dashboard_port (int, optional): for tracking daskboard. Defaults to 10101.
        walltime (_type_, optional): Defaults to "24:00:00".
        memory (str, optional): this is the memory per job/worker, not the overall memory! Defaults to "40GB".

    Returns:
        Client:
    """    
    cluster = SLURMCluster(
        cores=1,
        processes=1,
        memory=memory,
        account="incus",
        python=sys.executable,
        walltime=walltime,
        scheduler_options={"dashboard_address": f":{dashboard_port}"},
        job_extra_directives=["--partition=all", f"--job-name={job_name}"],
        job_script_prologue=[f"export HDF5_PLUGIN_PATH={hdf5plugin.PLUGIN_PATH}"],
    )
    client = Client(cluster)
    cluster.scale(jobs=jobs)

    print(f"Requesting {jobs} SLURM workers. Waiting for at least 1 to connect...")
    client.wait_for_workers(n_workers=1, timeout=2700)
    print("Cluster connection established successfully.")
    return client

def save_feature_output(data: dict[str, list[pd.DataFrame]] | list[pd.DataFrame], save_dir: Path, experiment_name: str | None = None):
    
    # If the input is a dictionary, then we are doing a parameter experiment and need to recurse over individual save files
    if isinstance(data, dict):
        for exp_name, list_of_dfs in data.items():
            save_feature_output(list_of_dfs, 
                                  save_dir=save_dir, 
                                  experiment_name=exp_name)
        return

    # If the input is just a list of dataframes, then we can go ahead and save!
    suffix = f"_{experiment_name}" if experiment_name else ""
    outpath = Path(save_dir, f"features{suffix}.pq")
    outpath.parent.mkdir(parents=True,exist_ok=True)

    if outpath.exists():
        return

    # filter out any empty dataframes
    data = [df for df in data if not df.empty]
    
    if not data:
        return # nothing to save!
    
    # make sure features are ordered by time
    sorted_features = sorted(data, key=lambda df: df['time'].iloc[0] if not df.empty else 0)

    # combine dataframes with tobac utils
    all_features = tobac.utils.combine_feature_dataframes(sorted_features)

    # save to parquet
    all_features.to_parquet(outpath)

def save_track_output(data: list[pd.DataFrame] | pd.DataFrame, save_dir: Path, experiment_name: str | list[str] | None = None):
    
    # If the input is a list, then we are doing a parameter experiment and need to recurse over individual save files
    if isinstance(data, list):
        for exp_name, list_of_dfs in zip(experiment_name, data):
            save_track_output(list_of_dfs, 
                                  save_dir=save_dir, 
                                  experiment_name=exp_name)
        return

    # If the input is just dataframes, then we can go ahead and save!
    suffix = f"_{experiment_name}" if experiment_name else ""
    outpath = Path(save_dir, f"tracks{suffix}.pq")
    outpath.parent.mkdir(parents=True,exist_ok=True)

    if outpath.exists():
        return

    data.to_parquet(outpath)
