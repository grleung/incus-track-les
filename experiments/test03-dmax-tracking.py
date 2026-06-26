'This script tests the d_max for tracking.'

from pathlib import Path
from incus_track_les import run_tracking
from incus_track_les.utils import init_slurm_downdraft,save_track_output

batch_size = 11

# initialize experiments for this test
test_name = 'test03-dmax'
feature_version = ('test02-numthresh','v04') # pick threshold version from test02 to use
grid_level = 3

parameter_experiments = {}

for i, dmax in enumerate([100,200,400,600,800,1000,1200,1400,1600,1800,2000]):
    experiment_name = f"v{str(i).zfill(2)}"

    params = {"extrapolate": 0,
                   "order": 1,
                   "memory": 0,
                   "time_cell_min": 4*30, # time in s
                   "method_linking": "predict",
                   "d_max": dmax,
                   "adaptive_step":0.75,
                   "adaptive_stop": 1.0,
                   'vertical_coord':'altitude_stag'
    }

    parameter_experiments[experiment_name] = params

if __name__ == '__main__':
    client = init_slurm_downdraft(jobs=batch_size, job_name=test_name,memory='5GB')

    for domain in ['ARG1.1','WPO1.1','BRA2.1']:
        for model in ['R','WM','WT']:
            run = f"{domain}-{model}-V1"
            print(f"Starting run: {run}")
            
            experiment_dir = Path(f'/tempest/gleung/incus-les-track-tests/{test_name}/{run}')
            experiment_dir.mkdir(parents=True,exist_ok=True)

            features_path = Path(f'/tempest/gleung/incus-les-track-tests/{feature_version[0]}/{run}/features_{feature_version[1]}.pq')

            # Check which experiments are missing for this run
            run_parameters = {
                experiment_name: experiment_params
                for experiment_name, experiment_params in parameter_experiments.items()
                if not Path(experiment_dir, f"tracks_{experiment_name}.pq").exists()
            }

            if run_parameters:
                # run tracking
                futures = client.map(run_tracking,
                                    [features_path]*len(run_parameters),
                                    [grid_level]*len(run_parameters),
                                    list(run_parameters.values()),
                                    )
                tracks = client.gather(futures)

                save_track_output(tracks, 
                                save_dir = experiment_dir, 
                                experiment_name=list(run_parameters.keys()) )
                



