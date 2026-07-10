# incus_track_les/utils.py

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import sys
from pathlib import Path
import pandas as pd
import tobac
import incus_track_les.tobac_families as tobac_fam

import dask

dask.config.set({
    "distributed.comm.timeouts.connect": "90s",
    "distributed.comm.timeouts.tcp": "90s",
    "distributed.worker.heartbeat": "30s"
})

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

def old_save_segmentation_output(data: dict[str, list[pd.DataFrame]] | list[pd.DataFrame], save_dir: Path, experiment_name: str | None = None):
    
    # If the input is a dictionary, then we are doing a parameter experiment and need to recurse over individual save files
    if isinstance(data, dict):
        for exp_name, list_of_dicts in data.items():
            for name in ['cloud_families','thermal_features','updraft_families']:
                dfs = [d[name] for d in list_of_dicts if name in d]
                save_segmentation_output(dfs, 
                                  save_dir=save_dir, 
                                  experiment_name=f"{name}_{exp_name}")
        return

    # If the input is just a list of dataframes, then we can go ahead and save!
    filename = experiment_name if experiment_name else ""
    outpath = Path(save_dir, f"{filename}.pq")
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
    all_features = tobac.utils.combine_feature_dataframes(sorted_features, renumber_features=False)

    # save to parquet
    all_features.to_parquet(outpath)

def save_segmentation_output(data: dict[str, list[pd.DataFrame]] | list[pd.DataFrame], save_dir: Path, experiment_name: str | None = None):
    import xarray as xr
    
    # If the input is a dictionary, then we are doing a parameter experiment and need to recurse over individual save files
    if isinstance(data, dict):
        for exp_name, list_of_dicts in data.items():
                save_segmentation_output(list_of_dicts, 
                                  save_dir=save_dir, 
                                  experiment_name=exp_name)
        return

    # If the input is just a list of dataframes, then we can go ahead and save!
    filename = f"_{experiment_name}" if experiment_name else ""
    outpath = Path(save_dir, f"segmentation{filename}.pq")
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
    all_features = tobac_fam.combine_feature_families(sorted_features,
                                            old_family_column_name='cloud_feature_id_original',
                                            family_column_name='cloud_feature_id')
    all_features = tobac_fam.combine_feature_families([all_features],
                                            old_family_column_name='updraft_feature_id_original',
                                            family_column_name='updraft_feature_id')
    
    #tobac.utils.combine_feature_dataframes(sorted_features, renumber_features=False)
    
    mask_dir = f"masks-{experiment_name}" if experiment_name else "masks"
    mask_dir = Path(save_dir, mask_dir)

    #rewrite the mask with updated numbering per timestep
    for time in all_features.time.unique():
        features_at_time = all_features[all_features.time==time]

        with xr.open_dataset(mask_dir/f"{time.strftime('%Y-%m-%d-%H%M%S')}.h5") as masks:
            masks.load()

            masks['cloud_mask'] = tobac_fam.renumber_feature_family_masks(features_at_time,
                                                    masks.cloud_mask,
                                                    old_family_column_name="cloud_feature_id_original",
                                                    family_column_name="cloud_feature_id")
            masks['updraft_mask'] = tobac_fam.renumber_feature_family_masks(features_at_time,
                                                    masks.updraft_mask,
                                                    old_family_column_name="updraft_feature_id_original",
                                                    family_column_name="updraft_feature_id")

                                            
                        
        masks.to_netcdf(Path(mask_dir, f"{time.strftime('%Y-%m-%d-%H%M%S')}.h5"),
                                engine='h5netcdf',
                                encoding={"cloud_mask": {"zlib": True, "complevel": 9},
                                        "thermal_mask": {"zlib": True, "complevel": 9},
                                        "updraft_mask": {"zlib": True, "complevel": 9}},)

    # drop the temporary feature IDs and save to parquet
    all_features.drop(['cloud_feature_id_original','updraft_feature_id_original'],axis=1).to_parquet(outpath)

