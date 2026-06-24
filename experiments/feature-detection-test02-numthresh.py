'This script tests the number and spacing of thresholds for feature detection'

import gc
import os
import tobac
from pathlib import Path
import sys
from incus_track_les import MODEL_DATA_DIR, get_filepaths, read_coords, feature_detection_timestep
import numpy as np
import hdf5plugin
plugin_path = hdf5plugin.PLUGIN_PATH
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

batch_size = 30
test_name = 'test02-numthresh'



# spin up SLURM cluster
cluster = SLURMCluster(
    cores=1,  # this is cpus-per-task
    processes=1,# number of  processes for each job; if not specified, this is usually sqrt(cores)
    memory="40GB",  # total memory to be divided among all workers
    account="incus",
    python=sys.executable,
    walltime="08:00:00",  # hours:minutes:seconds
    scheduler_options={
        "dashboard_address": ":10102"
    },  # change this to a port you want to use for monitoring
    job_extra_directives=["--partition=all", "--job-name=tracking-sensitivity-test02"],
    job_script_prologue=[
        f"export HDF5_PLUGIN_PATH={plugin_path}"
    ]
)  # change this to your job name

# set up our dask client and tell it to connect to the SLURM cluster
# this is where all our tasks get submitted to

client = Client(cluster)

cluster.scale(jobs=batch_size)
print("Waiting for Dask workers to spin up via SLURM...")
client.wait_for_workers(n_workers=1, timeout=2700) # 45-minute timeout window so that scheduler doesn't quit right away if worker jobs don't get picked up
print("Workers connected! Proceeding to tracking...")

min_thresh = 1
max_thresh = 49

parameter_experiments = {}

for i, thresh in enumerate([np.arange(min_thresh, max_thresh+2, 2),
               np.arange(min_thresh, max_thresh+3, 3),
               np.arange(min_thresh, max_thresh+6, 6),
               np.geomspace(min_thresh, max_thresh,25),
               np.geomspace(min_thresh, max_thresh,15),
               np.geomspace(min_thresh, max_thresh,10),
               ]):
    experiment_name = f"v{str(i+6).zfill(2)}"

    params = {"position_threshold": "weighted_diff",
              "sigma_threshold": 1,
              "n_erosion_threshold": 0,
              "n_min_threshold": 64,
              "target": "maximum",
              "threshold": thresh.tolist(),
              'vertical_coord':'altitude_stag'
              }

    parameter_experiments[experiment_name] = params

for domain in ['ARG1.1','WPO1.1']:
    for model in ['R','WM','WT']:
        run = f"{domain}-{model}-V1"
        print(f"Starting run: {run}")

        # Check which experiments are missing for this run
        run_parameters = {
            experiment_name: experiment_params
            for experiment_name, experiment_params in parameter_experiments.items()
            if not os.path.exists(
                f"/tempest/gleung/incus-les-track-tests/{test_name}/{run}/features_{experiment_name}.pq"
            )
        }
        if not run_parameters:
            print(f"All experiment features for {run} already exist on disk. Skipping.")
            continue

        filepaths = get_filepaths(Path(MODEL_DATA_DIR,run))
        filepaths = filepaths[60:90] # subset 15 min period to save computation

        print(f"Processing {len(filepaths)} files")

        filepath_batches = [filepaths[i:i + batch_size] for i in range(0, len(filepaths), batch_size)]

        coords = read_coords(filepaths[0])
        coords = client.scatter([coords])[0]

        # initialize output dictionary
        experiment_features = {experiment_name: [] for experiment_name in run_parameters.keys()}

        for filepath_batch in filepath_batches:
            futures = client.map(feature_detection_timestep, filepath_batch, coords=coords, parameters = run_parameters)
                    
            batch_results = client.gather(futures)

            # unpack the results per experiment into the right dictionary element
            for exp_results in batch_results:
                for experiment_name, df in exp_results.items():
                    experiment_features[experiment_name].append(df)

            # clean up memory; maybe this is unnecessary?
            del futures, batch_results
            gc.collect()
            client.run(gc.collect)

        for experiment_name, all_features in experiment_features.items():
            outpath = f"/tempest/gleung/incus-les-track-tests/{test_name}/{run}/features_{experiment_name}.pq"

            if not os.path.exists(outpath):
                # make sure features are ordered by time
                all_features = sorted(all_features, key=lambda df: df['time'].iloc[0] if not df.empty else 0)

                all_features = tobac.utils.combine_feature_dataframes(all_features)

                os.makedirs(os.path.dirname(outpath), exist_ok=True)

                all_features.to_parquet(outpath)
