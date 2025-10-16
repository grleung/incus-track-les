import pandas as pd
import glob
import xarray as xr
import numpy as np
import os
import tobac
import sys
import dask.distributed as dd
from dask_jobqueue import SLURMCluster
from dask_memusage import install

# spin up SLURM cluster
cluster = SLURMCluster(
    cores=2,
    processes=1,
    memory="100GB",
    account="incus",
    walltime="56:00:00",
    scheduler_options={"dashboard_address": f":{sys.argv[1]}"},
    job_extra_directives=[
        "--partition=all",
        "--job-name=tobac-family-identification",
    ],
)
cluster.scale(jobs=10)
client = dd.Client(cluster)
install(cluster.scheduler, "/home/gleung/memusage-stats-familyID-03.csv")

client.upload_file("shared_functions.py")
from shared_functions import save_files

# This is Sean's latest tobac_family code (version 5)
client.upload_file("tobac_family.py")
from tobac_family import identify_feature_families


def identify_feature_family(time, run, grid):
    trackPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/"

    features = pd.read_parquet(f"{trackPath}/qc_tracks_statistics.pq")
    features = features[features.time == time].reset_index(drop=True)

    w = xr.open_dataarray(
        f"{trackPath}/w_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
        engine="h5netcdf",
        chunks="auto",
    )
    w = w.where(w.isin(features.feature.unique())).fillna(0).astype(int)

    w_fam = identify_feature_families(
        features, w, family_column_name="w_family"
    )

    del w

    w_fam["w_family_count"] = w_fam.groupby("w_family").feature.transform(
        "count"
    )

    cond = xr.open_dataarray(
        f"{trackPath}/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
        engine="h5netcdf",
        chunks="auto",
    )
    cond = (
        cond.where(cond.isin(features.feature.unique())).fillna(0).astype(int)
    )

    cond_fam = identify_feature_families(
        w_fam, cond, family_column_name="cond_family"
    )

    del cond

    cond_fam["cond_family_count"] = cond_fam.groupby(
        "cond_family"
    ).feature.transform("count")

    return cond_fam


runs = sorted(
    [
        "ARG1.1-R-V1",
        "ARG1.2-R-V1",
        "AUS1.1-R-V1",
        "BRA1.1-R-V1",
        "BRA2.1-R-V1",
        "PHI1.1-R-V1",
        "USA1.1-R-V1",
        "WPO1.1-R-V1",
        "WPO1.1-RPR-V1",
        "SIO1.1-R-V1",
        "USA3.1-R-V1",
        "PHI2.1-R-V1",
        "DRC1.1-R-V1",
        "DRC1.1-RCR-V1",
        "SAU1.1-R-V1",
    ]
)

grids = ["g3"]

for grid in grids:
    for run in runs:
        tobacPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}"

        paths = sorted(os.listdir(f"{tobacPath}/cond_masks"))
        times = [pd.to_datetime(p[4:-3]) for p in paths]


        print(len(paths))


        if grid != "g3":

            savePath = f"{tobacPath}/qc_family_statistics.pq"

            if (os.path.exists(f"{tobacPath}/qc_tracks_statistics.pq")) and (
                not os.path.exists(savePath)
            ):

                print(run, grid, len(times))

                ds = client.map(
                    identify_feature_family,
                    times,
                    run=run,
                    grid=grid,
                    batch_size=12,
                )

                ds = client.gather(ds)

                ds = tobac.utils.combine_feature_dataframes(
                    ds,
                    renumber_features=False,
                    sort_features_by="frame",
                )

                save_files(ds, savePath)
        else:
            for n, times_ in enumerate(np.array_split(times, len(times) // 12)):
                savePath = (
                    f"{tobacPath}/qc_family_statistics_{str(n).zfill(2)}.pq"
                )

                if (
                    (os.path.exists(f"{tobacPath}/qc_tracks_statistics.pq"))
                    and (not os.path.exists(savePath)) and (
                not os.path.exists(f"{tobacPath}/qc_family_statistics.pq")
            )
                    
                ):

                    print(run, grid, len(times_), n)

                    ds = client.map(
                        identify_feature_family,
                        times_,
                        run=run,
                        grid=grid,
                        batch_size=2,
                    )

                    ds = client.gather(ds)

                    ds = pd.concat(ds)

                    save_files(ds, savePath)

client.close()
cluster.close()
