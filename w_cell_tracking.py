# This is the second script to run. Here, we use tobac to connect features in time
# to form cell tracks then subset those tracks which do not go within 9.6km of domain edges
# and which last longer than 2 minutes

# inputs: tobac output dataframe of features (from w_feature_identification)
# output: tobac output dataframe of tracked cells

# To run this script, just do "python w_cell_tracking.py" in command line.

# For questions, contact Bee (gabrielle.leung@colostate.edu)

# Import some shared libraries
import os
import pandas as pd
import tobac

from shared_functions import get_xy_spacing, save_files

# Define the paths to INCUS data and where to save output
ver = "V1"  # version of INCUS simulation dataset
modelPath = f"/monsoon/MODEL/LES_MODEL_DATA/{ver}/"
outPath = f"/monsoon/MODEL/LES_MODEL_DATA/Tracking/{ver}/"
# separately I created a pkl file that contains the number of x,y points for each of the simulations
# having this as separate dataframe saves on some computational cost from re-calculating this
# in every script
outx = pd.read_pickle(f"/tempest/gleung/incustrack/nx.pkl")
outy = pd.read_pickle(f"/tempest/gleung/incustrack/ny.pkl")
runs = outx.index
# tobac tracking parameters
# see tobac documentation for more detailed description of each parameter
params = {}
params["extrapolate"] = 0
params["order"] = 1
params["memory"] = 0
params["time_cell_min"] = 2 * 60  # in seconds
params["method_linking"] = "predict"
params["adaptive_step"] = 0.75
params["adaptive_stop"] = 1.0

# actual loop
for run in runs:
    print(run)
    dataPath = f"{modelPath}/{run}/G3/out_30s/"

    for grid, dmax in zip(
        ["g1", "g2", "g3"], [4800, 2000, 800]
    ):  # based on 0.5th percentile of nearest neighbor distance
        print(grid)

        featPath = f"{outPath}/{run}/{grid}/w_features.pq"
        savePath = f"{outPath}/{run}/{grid}/w_tracks.pq"

        if (os.path.exists(featPath)) and (not os.path.exists(savePath)):

            # assign dxy (horizontal grid spacing) based on grid
            dxy = get_xy_spacing(grid)

            # the d_max comes from the mean 25th percentile nearest neighbor distance (see Bee's ppt documenting this
            # on SharePoint under INCUS_LES_Tracking_ScienceDatabase)
            # these values are different per grid, but the same for all domains
            params["d_max"] = dmax

            # check if there is already a pickle file for w tracks so we don't re-do this
            # and check to make sure pickle file with features already exists
            # if (not os.path.exists(savePath)) & (os.path.exists(featPath)):
            if True:
                features = pd.read_parquet(featPath)

                # Perform linking and save results:
                tracks = tobac.linking_trackpy(
                    features,
                    None,
                    dt=30,  # time in seconds separating each frame
                    dxy=dxy,
                    vertical_coord="ztn",
                    **params,
                )

                # remove any features that don't belong to a track or are less than 2 mins long (in tobac these are labelled -1)
                out_tracks = tracks[tracks.cell > 0]

                # second filter out cells near the edges of the nested grid
                nx = outx.loc[run, grid]
                ny = outy.loc[run, grid]

                # define how near to the boundary features are allowed to be
                # right now, we delete any feature that is within 1600m of the boundaries of the g3 lat/lon bounding box
                # that is 1 pt for g1, 4 pts for g2, 16 pts for g3
                # this is so that we exclude the sponge zone for g3 (which is 8 points)
                min_h1, max_h1 = 1600 / dxy, ny - 1600 / dxy
                min_h2, max_h2 = 1600 / dxy, nx - 1600 / dxy

                # make sure lat lon are float
                out_tracks["lat"] = out_tracks["lat"].astype("float")
                out_tracks["lon"] = out_tracks["lon"].astype("float")

                # select only cells where the cell is never outside the max/min horizontal locations
                cells = out_tracks[["cell", "hdim_1", "hdim_2"]].groupby(
                    "cell")

                out_tracks = out_tracks[
                    (
                        (cells.hdim_1.transform("min") > min_h1)
                        & (cells.hdim_2.transform("min") > min_h2)
                    )
                    & (
                        (cells.hdim_1.transform("max") < max_h1)
                        & (cells.hdim_2.transform("max") < max_h2)
                    )
                ]

                # just make sure all directories exist
                if not os.path.isdir(f"{outPath}/{run}"):
                    os.mkdir(f"{outPath}/{run}")

                if not os.path.isdir(f"{outPath}/{run}/{grid}"):
                    os.mkdir(f"{outPath}/{run}/{grid}")

                save_files(out_tracks, savePath)
