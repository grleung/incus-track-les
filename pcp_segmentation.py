# Import some shared libraries
import os
import dask.distributed as dd
import dask
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob

client = dd.Client("updraft:8786")
client.upload_file("shared_functions.py")

from shared_functions import (
    get_rams_output,
    get_xy_spacing,
    subset_data,
    combine_tobac_list,
    rams_dims_lite,
    all_var,
    save_files,
    compute_cond,
    compute_pcp,
    p00,
    rd,
    cp,
)

# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}-temp/"

runs = ['DRC1.1-RCR-V1']
grids = ["g3"]

# parameters for segmentation
params = {}
params["method"] = "watershed"
params["threshold"] = 0.01  # mm/hr mixing ratio

# separately I created a pkl file that contains the min/max lat/lon for each of the simulations
# having this as separate dataframe saves on some computational cost from re-calculating this
# in every script
outbounds = pd.read_pickle(f"/tempest/gleung/incustrack/bounds.pkl")

# loop through each of the runs
for grid in grids:
    for run in runs:
        if 'BRA' in run:
            dataPath = f"{modelPath}/{run}/G3/out_30s/lite/"
        else:
            dataPath = f"{modelPath}/{run}/G3/out_30s/"

        # list of all timesteps where lite files are found in relevant folder
        all_paths = [
            p.split("/")[-1][:-6]
            for p in sorted(glob.glob(f"{dataPath}/a-L-*-g3.h5"))
        ]

        dxy = get_xy_spacing(grid)
        trackPath = (
            f"{outPath}/{run}/{grid}/combined_w_cond_segmented_tracks.pq"
        )
        tracks = pd.read_parquet(trackPath)

        savemaskPath = f"{outPath}/{run}/{grid}/pcp_masks/"

        n = 1
        batch_size = 20
        all_paths = enumerate([all_paths])

        for i, paths in all_paths:
            times = [pd.to_datetime(p.split("/")[-1][4:]) for p in paths]

            latbounds = outbounds.loc[run].values

            savedfPath = f"{outPath}/{run}/{grid}/combined_w_cond_pcp_segmented_tracks.pq"

            ds = client.map(
                get_rams_output,
                [f"{dataPath}/{p}-{grid}.h5" for p in paths],
                variables=[
                    "PCPRR",
                    "PCPRP",
                    "PCPRS",
                    "PCPRA",
                    "PCPRG",
                    "PCPRH",
                    "PCPRD",
                ],
                latbounds=latbounds,
                latlon=True,
                coords=True,
                batch_size=batch_size,
            )

            ds = client.map(compute_pcp, ds, batch_size=batch_size)

            ds = client.map(
                xr.DataArray.expand_dims,
                ds,
                [{"time": [t]} for t in times],
                batch_size=batch_size,
            )

            ds = client.map(
                xr.DataArray.to_iris,
                ds,
                batch_size=batch_size,
            )

            out = client.map(
                tobac.segmentation.segmentation,
                [tracks[tracks.time == t] for t in times],
                ds,
                dxy=dxy,
                **params,
                batch_size=batch_size,
            )

            out = client.gather(out)

            all_segments = [o[1] for o in out]
            all_masks = [o[0] for o in out]

            # just make sure all directories exist
            if not os.path.isdir(f"{outPath}/{run}"):
                os.mkdir(f"{outPath}/{run}")

            if not os.path.isdir(f"{outPath}/{run}/{grid}"):
                os.mkdir(f"{outPath}/{run}/{grid}")

            # once loop is finished, concatenate all the figures
            # then save it to a parquet file
            all_segments = combine_tobac_list(all_segments)
            save_files(all_segments, savedfPath)

            if not os.path.isdir(savemaskPath):
                os.mkdir(savemaskPath)

            for m, p in zip(all_masks, paths):
                ds = xr.DataArray.from_iris(m)
                ds.to_netcdf(
                    f"{savemaskPath}/{p}.h5",
                    engine="h5netcdf",
                    encoding={
                        "segmentation_mask": {"zlib": True, "complevel": 9}
                    },
                )
