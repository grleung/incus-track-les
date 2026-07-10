import gc
import os
import numpy as np
import pandas as pd
from pathlib import Path
from incus_track_les import MODEL_DATA_DIR, get_filepaths, read_coords, run_feature_detection_timestep, run_tracking, run_segmentation_timestep, calculate_statistics
from incus_track_les.utils import init_slurm_downdraft, save_feature_output,save_track_output, save_segmentation_output

batch_size = 15
grid_level= 3

test_name = 'test08-fullpipeline'

feature_detection_params = {
    'prod':{"position_threshold": "weighted_diff",
              "sigma_threshold": 1,
              "n_erosion_threshold": 0,
              "n_min_threshold": 64,
              "target": "maximum",
              "threshold": np.geomspace(2, 50, 15),
              'vertical_coord':'altitude'
              }}

tracking_params = {"extrapolate": 0,
                   "order": 1,
                   "memory": 0,
                   "time_cell_min": 4*30, # time in s
                   "method_linking": "predict",
                   "d_max": 800,
                   'vertical_coord':'altitude'
}

segmentation_params = {'prod':
            {"cloud_threshold": 1e-3,
              "thermal_segmentation_params": {
                  'method': 'watershed',
                  'threshold': 2, # m/s, lowest vertical velocity threshold
                  'seed_3D_flag': 'box',
                  'vertical_coord': 'altitude',
                  'seed_3D_size': (5,5,5),
              }
              }}


if __name__ == '__main__':
    client = init_slurm_downdraft(jobs=batch_size, job_name=test_name, memory='140GB')
    for domain in ['WPO1.1','ARG1.1','BRA2.1']:
        for model in ['R','WM','WT']:
            run = f"{domain}-{model}-V1"
            print(f"Starting run: {run}")
            output_dir = Path(f'/tempest/gleung/incus-les-track-tests/{test_name}/{run}')
            output_dir.mkdir(exist_ok=True,parents=True)

            
            features_path = Path(output_dir, 'features.pq')
            tracks_path = Path(output_dir, 'tracks.pq')
            segmentation_path = Path(output_dir, "segmentation.pq")
            thermals_path = Path(output_dir, "thermals.pq")
            
            # get files  
            filepaths = get_filepaths(Path(MODEL_DATA_DIR,run),
                                            grid_level=grid_level)
            filepaths = filepaths[60:90] # subset 15 min period to save computation

            print(f"Processing {len(filepaths)} files")

            # batch files to limit memory usage
            filepath_batches = [filepaths[i:i + batch_size] for i in range(0, len(filepaths), batch_size)]

            print('Done making file batches', flush=True)

            # read and scatter coordinates
            coords = read_coords(filepaths[0])
            coords = client.scatter([coords])[0]
            print('Done preparing coordinates', flush=True)

            # Step 1: feature detection
            # check if features already exist
            if not features_path.exists():
                # initialize output dictionary
                features = []

                for filepath_batch in filepath_batches:
                    futures = client.map(run_feature_detection_timestep,
                                        filepath_batch,
                                        coords=coords,
                                        parameters = feature_detection_params)
                            
                    batch_results = client.gather(futures)

                    for exp_results in batch_results:
                        features.append(exp_results['prod'])

                    # clean up memory
                    del futures, batch_results
                    gc.collect()
                    client.run(gc.collect)
                    
                # save feature output from all batches
                save_feature_output(features, 
                            save_dir = output_dir)

            
            print('Done feature detection', flush=True)
            
            # Step 2: tracking

            # check if tracks already exist
            if not tracks_path.exists():
                tracks = run_tracking(features_path,
                                    grid_level=grid_level, 
                                    parameters = tracking_params)

                save_track_output(tracks, 
                                save_dir = output_dir)

            
            print('Done tracking', flush=True)

            # Step 3: Segmentation

            # check if segmentation already exists
            if not segmentation_path.exists():
                features = []

                for filepath_batch in filepath_batches:
                    futures = client.map(run_segmentation_timestep,
                                        filepath_batch,
                                        trackspath = tracks_path,
                                        maskspath = output_dir,
                                        coords=coords,
                                        parameters = segmentation_params)
                            
                    batch_results = client.gather(futures)

                    for exp_results in batch_results:
                        features.append(exp_results['prod'])

                    # clean up memory
                    del futures, batch_results
                    gc.collect()
                    client.run(gc.collect)

                # save segmentation output from all batches 
                save_segmentation_output(features, 
                            save_dir = output_dir)

            print('Done segmentation', flush=True)

            # Step 4: Calculate statistics
            # check if stats already exists
            if not thermals_path.exists():
                features = []

                for filepath_batch in filepath_batches:
                    futures = client.map(calculate_statistics,
                                        filepath_batch,
                                        featurespath = segmentation_path,
                                        maskspath = Path(output_dir, 'masks/'),
                                        coords=coords,
                                        stats_level='thermal')
                            
                    batch_results = client.gather(futures)

                    features.extend(batch_results)

                    # clean up memory
                    del futures, batch_results
                    gc.collect()
                    client.run(gc.collect)

                # save stats output from all batches
                features = pd.concat(features)
                features.to_parquet(Path(output_dir,'thermals.pq'))
                
















                
