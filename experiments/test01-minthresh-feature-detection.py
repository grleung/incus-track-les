'This script tests the minimum vertical velocity threshold for feature detection.'
import gc
import os
from pathlib import Path
from incus_track_les import MODEL_DATA_DIR, get_filepaths, read_coords, run_feature_detection_timestep, run_tracking
from incus_track_les.utils import init_slurm_downdraft, save_feature_output,save_track_output

batch_size = 30
grid_level=3
# initialize experiments for this test
test_name = 'test01-minthresh'

parameter_experiments = {}


for i, thresh in enumerate([1,1.5,2,2.5,3,3.5,4,4.5,5]):
    experiment_name = f"v{str(i).zfill(2)}"

    params = {"position_threshold": "weighted_diff",
              "sigma_threshold": 1,
              "n_erosion_threshold": 0,
              "n_min_threshold": 64,
              "target": "maximum",
              "threshold": [thresh],
              'vertical_coord':'altitude_stag'
              }

    parameter_experiments[experiment_name] = params

# keep default tracking parameters
tracking_params = {"extrapolate": 0,
                   "order": 1,
                   "memory": 0,
                   "time_cell_min": 4*30, # time in s
                   "method_linking": "predict",
                   "d_max": 800,
                   'vertical_coord':'altitude_stag'
}


if __name__ == '__main__':
    client = init_slurm_downdraft(jobs=batch_size, job_name=test_name)
    for domain in ['ARG1.1','WPO1.1','BRA2.1']:
        for model in ['R','WM','WT']:
            run = f"{domain}-{model}-V1"
            print(f"Starting run: {run}")
            experiment_dir = Path(f'/tempest/gleung/incus-les-track-tests/{test_name}/{run}')

            # Check which experiments are missing for this run
            run_parameters = {
                experiment_name: experiment_params
                for experiment_name, experiment_params in parameter_experiments.items()
                if not Path(experiment_dir, f"features_{experiment_name}.pq").exists()
            }

            if run_parameters:
                filepaths = get_filepaths(Path(MODEL_DATA_DIR,run),
                                          grid_level=grid_level)
                filepaths = filepaths[60:90] # subset 15 min period to save computation

                print(f"Processing {len(filepaths)} files")

                # batch files to limit memory usage
                filepath_batches = [filepaths[i:i + batch_size] for i in range(0, len(filepaths), batch_size)]

                coords = read_coords(filepaths[0])
                coords = client.scatter([coords])[0]

                # initialize output dictionary
                experiment_features = {experiment_name: [] for experiment_name in run_parameters.keys()}

                for filepath_batch in filepath_batches:
                    futures = client.map(run_feature_detection_timestep,
                                          filepath_batch,
                                          coords=coords,
                                        parameters = run_parameters)
                            
                    batch_results = client.gather(futures)

                    # unpack the results per experiment into the right dictionary element
                    for exp_results in batch_results:
                        for experiment_name, df in exp_results.items():
                            experiment_features[experiment_name].append(df)

                    # clean up memory
                    del futures, batch_results
                    gc.collect()
                    client.run(gc.collect)
                    
                # save feature output from all batches and experiments!
                save_feature_output(experiment_features, 
                            save_dir = experiment_dir)

                
            # run tracking
            features_paths = list(experiment_dir.glob('features_v*.pq'))
            experiment_names = [p.stem.replace('features_', '') for p in features_paths]
            
            futures = client.map(run_tracking,
                                 features_paths, 
                                 grid_level=grid_level, 
                                 parameters =tracking_params)
            tracks = client.gather(futures)

            save_track_output(tracks, 
                              save_dir = experiment_dir, 
                        experiment_name=experiment_names)

