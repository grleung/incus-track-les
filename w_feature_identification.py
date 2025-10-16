# This is the first script to run. Here, we use tobac to identify updraft
# features that meet the given vertical velocity thresholds.

# inputs: list of model runs, RAMS model output
# output: tobac output dataframe of features

# This script is parallelized with dask distributed from a SLURM Cluster. To run from command line
# run "python w_feature_identification.py [SCHEDULER ADRESS] [LIST OF RUNS]"
# e.g., "python w_feature_identification 10101 DRC1.1-R-V1 ARG1.1-R-V1"
#
# For questions, contact Bee (gabrielle.leung@colostate.edu)

# Import some shared libraries
from shared_functions import (
    get_rams_output,
    get_xy_spacing,
    save_files,
)
import os
import dask.distributed as dd
from dask_jobqueue import SLURMCluster
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob
import sys

# typical vlaues: g1 :: 20 cores, 50GB mem
# g2 :
# g3 10 cores, 200GB mem

# spin up SLURM cluster
cluster = SLURMCluster(
    cores=6,
    processes=6,
    memory="300GB",
    account="incus",
    walltime="56:00:00",
    scheduler_options={"dashboard_address": f":{sys.argv[1]}"},
    job_extra_directives=[
        "--partition=all",
        "--job-name=tobac-feature-detection",
    ],
)
cluster.scale(jobs=1)

# change this address depending on your scheduler address
client = dd.Client(cluster)

client.upload_file("shared_functions.py")

# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"

runs = sys.argv[2:]
print(runs)
grids = ["g3"]

# separately I created a pkl file that contains the min/max lat/lon for each of the simulations
# having this as separate dataframe saves on some computational cost from re-calculating this
# in every script
xybounds = pd.read_parquet(f"/tempest/gleung/incustrack/xybounds.pq")

# tobac feature identification parameters
# see tobac documentation for more detailed description
params = {}
params["position_threshold"] = "weighted_diff"
params["sigma_threshold"] = 1
params["n_erosion_threshold"] = 0
# this is ~4 points in each direction; we had some discussion about changing
# this for different grid spacings, but as of 2024-08-13 this makes more sense to me (Bee)
params["n_min_threshold"] = 64
params["target"] = "maximum"
# threshold in m/s is (1, 2, 4, 6, ..., 50)
params["threshold"] = np.append([1.0], np.arange(2.0, 52.0, 2.0))

# loop through each of the runs
for run in runs:
    dataPath = f"{modelPath}/{run}/G3/out_30s/"

    # list of all timesteps where lite files are found in relevant folder
    all_paths = [
        p.split("/")[-1][:-6]
        for p in sorted(glob.glob(f"{dataPath}/a-L-*-g3.h5"))
    ]

    print(len(all_paths))

    for grid in grids:
        bounds = xybounds.loc[run, grid]

        # For some of the g3 domains, I had some issues with too much data being loaded into memory at once,
        # so I split up feature detection into multiple subsets
        if grid == "g3":
            n_split = len(all_paths) // 12
        else:
            n_split = 1

        for i, paths in enumerate(np.array_split(all_paths, n_split)):
            if grid == "g3":
                savedfPath = (
                    f"{outPath}/{run}/{grid}/w_features_{str(i).zfill(2)}.pq"
                )
            else:
                savedfPath = f"{outPath}/{run}/{grid}/w_features.pq"

            if not os.path.exists(savedfPath):
                print(i)

                # batch size is how may batches to submit the tasks in the list [paths] to scheduler
                if grid == "g3":
                    batch_size = 4
                else:
                    batch_size = 30

                dxy = get_xy_spacing(grid)

                # prep data for feeding to tobac
                ds = client.map(
                    get_rams_output,
                    [f"{dataPath}/{p}-{grid}.h5" for p in paths],
                    variables=["WP"],
                    bounds=bounds,
                    subset=True,
                    subsetxy=True,
                    coords=True,
                )

                ds = client.map(
                    xr.DataArray.expand_dims,
                    ds,
                    [
                        {"time": [pd.to_datetime(p.split("/")[-1][4:])]}
                        for p in paths
                    ],
                )

                # actual tobac run
                feats = client.map(
                    tobac.feature_detection_multithreshold,
                    ds,
                    dxy=dxy,
                    vertical_coord="ztn",
                    **params,
                    batch_size=batch_size,
                )

                # take all the features from tobac run
                all_features = client.gather(feats)

                # once loop is finished, concatenate all the figures
                # then save it to a parquet file
                all_features = tobac.utils.combine_feature_dataframes(
                    all_features,
                    renumber_features=True,
                    sort_features_by="frame",
                )

                # just make sure all directories exist
                if not os.path.isdir(f"{outPath}/{run}"):
                    os.mkdir(f"{outPath}/{run}")

                if not os.path.isdir(f"{outPath}/{run}/{grid}"):
                    os.mkdir(f"{outPath}/{run}/{grid}")

                save_files(all_features, savedfPath)

cluster.close()
client.close()
