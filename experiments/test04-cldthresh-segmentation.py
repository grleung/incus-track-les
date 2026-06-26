'This script tests the number and spacing of thresholds for feature detection'

import gc
from pathlib import Path
from incus_track_les import MODEL_DATA_DIR, get_filepaths, read_coords, run_segmentation_timestep
from incus_track_les.utils import init_slurm_downdraft, save_segmentation_output

batch_size = 5

# initialize experiments for this test
test_name = 'test04-cldthresh'
tracks_version = ('test03-dmax','v04') # pick threshold version from test03 to use

grid_level = 3

parameter_experiments = {}

for i, thresh in enumerate([1e-3,1e-4,1e-5]):
    experiment_name = f"v{str(i).zfill(2)}"

    params = {"cloud_threshold": thresh,
              "thermal_segmentation_params": {
                  'method': 'watershed',
                  'threshold': 2, # m/s, lowest vertical velocity threshold
                  'seed_3D_flag': 'box',
                  'vertical_coord': 'altitude_stag',
                  'seed_3D_size': (5,5,5),
              }
              }

    parameter_experiments[experiment_name] = params

if __name__ == '__main__':
    client = init_slurm_downdraft(jobs=batch_size, job_name=test_name, memory='150GB')

    for domain in ['ARG1.1','WPO1.1']:
        for model in ['R']:
            run = f"{domain}-{model}-V1"
            print(f"Starting run: {run}")

            experiment_dir = Path(f'/tempest/gleung/incus-les-track-tests/{test_name}/{run}')
            experiment_dir.mkdir(parents=True, exist_ok=True)

            tracks_path = Path(f'/tempest/gleung/incus-les-track-tests/{tracks_version[0]}/{run}/tracks_{tracks_version[1]}.pq')

            # Check which experiments are missing for this run
            run_parameters = {
                experiment_name: experiment_params
                for experiment_name, experiment_params in parameter_experiments.items()
                if not Path(experiment_dir, f"cloud_families_{experiment_name}.pq").exists()
            }

            if run_parameters:
                filepaths = get_filepaths(Path(MODEL_DATA_DIR,run),
                                          grid_level=grid_level)
                filepaths = filepaths[60:65]#90] # subset 15 min period to save computation

                print(f"Processing {len(filepaths)} files")

                # batch files to limit memory usage
                filepath_batches = [filepaths[i:i + batch_size] for i in range(0, len(filepaths), batch_size)]

                coords = read_coords(filepaths[0])
                coords = client.scatter([coords])[0]

                # initialize output dictionary
                experiment_features = {experiment_name: [] for experiment_name in run_parameters.keys()}

                for filepath_batch in filepath_batches:
                    futures = client.map(run_segmentation_timestep, 
                                        filepath_batch, 
                                        trackspath = tracks_path,
                                        maskspath = experiment_dir,
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

                # save segmentation output from all batches and experiments!
                save_segmentation_output(experiment_features, 
                            save_dir = experiment_dir)



