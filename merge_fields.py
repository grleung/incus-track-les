import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import glob
import dask.distributed as dd

client = dd.Client('updraft:1428')

runs = [
    "ARG1.1-R-V1",
    "ARG1.2-R-V1",
    "AUS1.1-R-V1",
    "BRA1.1-R-V1",
    "BRA2.1-R-V1",
    "DRC1.1-R-V1",
    "DRC1.1-RCR-V1",
    "PHI1.1-R-V1",
    "PHI2.1-R-V1",
    "SAU1.1-R-V1",
    "SIO1.1-R-V1",
    "USA1.1-R-V1",
    "USA3.1-R-V1",
    "WPO1.1-R-V1",
    "WPO1.1-RPR-V1",
    'BOT1.1-R-V1',
    "BRA1.1-RPR-V1",
    "MED1.1-R-V1"
]
grids = ["g2"]


def get_overlap_features(time, tracks_sub, tobacPath):
    if len(tracks_sub) > 0:

        cond_mask = xr.open_dataset(f"{tobacPath}/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
                                    chunks='auto', engine='h5netcdf')
        w_mask = xr.open_dataset(f"{tobacPath}/w_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
                                 chunks='auto', engine='h5netcdf')

        # mask will have 1s where both updraft and condensate are present
        # This will allow us to make sure updraft and condensate regions are coincident/overlapping in space

        full_mask = (w_mask.segmentation_mask > 0) * \
            (cond_mask.segmentation_mask > 0)

        fts = cond_mask.where(full_mask).segmentation_mask.compute()
        ftlist = np.unique(fts.values)
        ftlist = ftlist[~np.isnan(ftlist)]

        tracks_sub = tracks_sub[tracks_sub.feature.isin(ftlist)]

        return (tracks_sub)


for run in runs:
    for grid in grids:
        print(run, grid)
        dataPath = f"/monsoon/MODEL/LES_MODEL_DATA/V1/{run}/{grid.capitalize()}/out_30s/"
        tobacPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}"

        if (
            (
                not os.path.exists(
                    f"{tobacPath}/combined_w_cond_segmented_tracks.pq"
                )
            )
            and (os.path.exists(f"{tobacPath}/cond_seg.pq"))
            and (os.path.exists(f"{tobacPath}/w_seg.pq"))
        ):
            print("merging")
            tracks = pd.read_parquet(
                f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/w_tracks.pq"
            )
            w = pd.read_parquet(
                f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/w_seg.pq"
            )
            cond = pd.read_parquet(
                f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/cond_seg.pq"
            )

            if (
                sorted(w.feature.values) == sorted(tracks.feature.values)
            ) != True:
                w["feature"] = w.set_index(["frame", "cell"]).index.map(
                    tracks.set_index(["frame", "cell"]).feature
                )

                w.to_parquet(
                    f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/w_seg.pq"
                )

                print("renumbered w features")

            if (
                sorted(cond.feature.values) == sorted(tracks.feature.values)
            ) != True:
                cond["feature"] = cond.set_index(["frame", "cell"]).index.map(
                    tracks.set_index(["frame", "cell"]).feature
                )

                cond.to_parquet(
                    f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/V1/{run}/{grid}/cond_seg.pq"
                )

                print("renumbered cond features")

            w = w.set_index("feature")
            cond = cond.set_index("feature")

            tracks["ncells_w"] = tracks.feature.map(w.ncells)
            tracks["ncells_cond"] = tracks.feature.map(cond.ncells)

            tracks["lifetime"] = tracks.groupby("cell").time_cell.transform(
                "max"
            ) / dt.timedelta(minutes=1)
            tracks["frac_lifetime"] = (
                tracks.time_cell / dt.timedelta(minutes=1)
            ) / tracks.lifetime

            tracks["cellmax_nw"] = tracks.groupby("cell").ncells_w.transform(
                "max"
            )
            tracks["cellmax_ncond"] = tracks.groupby(
                "cell"
            ).ncells_cond.transform("max")
            print(len(tracks))

            thresh = 64
            tracks = tracks[
                (tracks.cellmax_nw >= thresh) & (
                    tracks.cellmax_ncond >= thresh)
            ]

            """times = tracks.time.unique()

            # Now we need to check that condensate and w segments actually overlap in space
            for i, times_ in enumerate(np.array_split(times, len(times)//50)):
                if (not os.path.exists(f'{tobacPath}/cloudy_updrafts_{str(i).zfill(2)}.pq')):
                    tracks_ = client.map(get_overlap_features, times_, [
                                         tracks[tracks.time == t] for t in times_], tobacPath=tobacPath)

                    tracks_ = client.gather(tracks_)

                    tracks_ = pd.concat(tracks_)

                    tracks_.to_parquet(
                        f'{tobacPath}/cloudy_updrafts_{str(i).zfill(2)}.pq')

                    print(f'{tobacPath}/cloudy_updrafts_{str(i).zfill(2)}.pq')

            savePaths = sorted(glob.glob(f"{tobacPath}/cloudy_updrafts_*.pq"))

            # make sure all the saved files are present
            if len(savePaths) == (len(times) // 50):
                # read in all the files, combine, and save

                df = pd.read_parquet(savePaths, engine="pyarrow")
                df.to_parquet(f"{tobacPath}/cloudy_updrafts.pq")"""

            tracks.to_parquet(
                f"{tobacPath}/combined_w_cond_segmented_tracks.pq",
                engine="pyarrow",
            )

        print(run, grid)
