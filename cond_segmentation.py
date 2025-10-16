# Import some shared libraries
from shared_functions import (
    get_rams_output,
    get_xy_spacing,
    compute_cond,
    save_files,
)
import os
import sys
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob
from dask_memusage import install

# g3 : 6 cores, 2 proc, 360 GB

# spin up SLURM cluster
cluster = SLURMCluster(
    cores=24,
    processes=24,
    memory="200GB",
    account="incus",
    walltime="56:00:00",
    scheduler_options={"dashboard_address": f":{sys.argv[1]}"},
    job_extra_directives=[
        "--partition=all",
        "--job-name=tobac-cond-segmentation",
    ],
)
client = Client(cluster)
cluster.scale(jobs=5)

install(cluster.scheduler, "/home/gleung/memusage-condseg2-08.csv")

client.upload_file("shared_functions.py")


# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation datasetls /m
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"
runs = sys.argv[2:]
grids = ["g2"]

# parameters for segmentation
params = {}
params["method"] = "watershed"
params["threshold"] = 1.0e-5  # kg/m3 mixing ratio
params["seed_3D_flag"] = "box"
params["vertical_coord"] = "ztn"
params["seed_3D_size"] = (15, 5, 5)


def dask_segmentation(path, run, grid, outPath, params):
    xybounds = pd.read_parquet("/tempest/gleung/incustrack/xybounds.pq").loc[
        run, grid
    ]

    dxy = get_xy_spacing(grid)

    savemaskPath = f"{outPath}/{run}/{grid}/cond_masks/"

    time = pd.to_datetime(path.split("/")[-1][4:])

    if grid == "g3":
        subset = False
        subsetxy = False
    else:
        subset = True
        subsetxy = True

    ds = get_rams_output(
        f"{modelPath}/{run}/G3/out_30s/{path}-{grid}.h5",
        variables=["RCP", "RSP", "RPP", "PI", "THETA", "RV"],
        bounds=xybounds,
        subset=True,
        subsetxy=True,
        coords=True,
    )

    ds = compute_cond(ds)

    ds = ds.expand_dims({"time": [time]})

    tracks = pd.read_parquet(f"{outPath}/{run}/{grid}/w_tracks.pq").reset_index(
        drop=True
    )
    tracks = tracks[tracks.time == time].reset_index(drop=True)

    mask, seg = tobac.segmentation.segmentation(tracks, ds, dxy=dxy, **params)

    mask = mask.to_netcdf(
        f"{savemaskPath}/{path}.h5",
        engine="h5netcdf",
        encoding={"segmentation_mask": {"zlib": True, "complevel": 9}},
    )

    del mask
    del ds

    return seg


# loop through each of the runs
for grid in grids:
    for run in runs:

        # just make sure all directories exist
        if not os.path.isdir(f"{outPath}/{run}"):
            os.mkdir(f"{outPath}/{run}")

        if not os.path.isdir(f"{outPath}/{run}/{grid}"):
            os.mkdir(f"{outPath}/{run}/{grid}")

        if not os.path.isdir(f"{outPath}/{run}/{grid}/cond_masks"):
            os.mkdir(f"{outPath}/{run}/{grid}/cond_masks")

        print(grid, run)

        dataPath = f"{modelPath}/{run}/G3/out_30s"
        # list of all timesteps where lite files are found in relevant folder
        all_paths = [
            p.split("/")[-1][:-6]
            for p in sorted(glob.glob(f"{dataPath}/a-L-*-g3.h5"))
        ]

        if grid == "g3":
            n = len(all_paths) // 8
        elif grid == "g2":
            n = len(all_paths) // 24
        else:
            n = 1

        all_paths = enumerate(np.array_split(all_paths, n))

        for i, paths in all_paths:
            print(i, grid, run)
            times = [pd.to_datetime(p.split("/")[-1][4:]) for p in paths]

            if (grid == "g3") or (grid == "g2"):
                savedfPath = (
                    f"{outPath}/{run}/{grid}/cond_seg_{str(i).zfill(2)}.pq"
                )
            else:
                savedfPath = f"{outPath}/{run}/{grid}/cond_seg.pq"

            if not os.path.exists(savedfPath):
                out = client.map(
                    dask_segmentation,
                    paths,
                    run=run,
                    grid=grid,
                    outPath=outPath,
                    params=params,
                    batch_size=1,
                )

                out = client.gather(out)

                # once loop is finished, concatenate all the figures
                # then save it to a parquet file
                all_segments = tobac.utils.combine_feature_dataframes(
                    out,
                    renumber_features=False,
                    sort_features_by="frame",
                )
                save_files(all_segments, savedfPath)
