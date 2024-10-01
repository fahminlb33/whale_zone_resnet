import os
import glob
import time
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor

def run_train(run):
    try:
        print("RUN!", run)

        env = {
            **os.environ.copy(),
            "PYTHONHASHSEED": "42", 
            # "TF_DETERMINISTIC_OPS": "1", 
            # "TF_CUDNN_DETERMINISTIC": "1"
        }
        
        result = subprocess.run(["python", "train.py", "-m", run[0], "-l", run[1], "-d", run[2], "-n", run[3]], env=env, text=True, capture_output=True)
        
        with open(f"./runs/{run[0]}_{run[1]}_{run[2]}_{run[3]}.log", "w") as f:
            f.write(result.stdout)
            f.write("\n\n-------------------------------------------------------\n\n")
            f.write(result.stderr)
    except Exception as e:
        print("ERROR!", run, e)

def main():
    valid_models = ["fcn", "resnet", "resnet_bias"]
    valid_losses = ["bce", "fl"]
    valid_depths = ["1", "2", "3", "4", "5", "6"]
    valid_dataset_suffixes = ["sel-10-undersampled", "all-undersampled", "sel-10", "all"]

    with open("start_time_v3.txt", "w") as f:
        f.write(f"{time.time()}")
        f.write("\n")

    already_run = glob.glob("./models/*.keras")
    all_valid_runs = list(itertools.product(valid_models, valid_losses, valid_depths, valid_dataset_suffixes))

    remaining_runs = []
    for run in all_valid_runs:
        run_name = f"{run[3]}_{run[0]}_{run[1]}_{run[2]}"
        if any([run_name in x for x in already_run]):
            print("SKIP:", run_name)
            continue

        remaining_runs.append(run)

    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(run_train, remaining_runs)

    with open("end_time.txt", "w") as f:
        f.write(f"{time.time()}")
        f.write("\n")

    os.system("shutdown.exe -s -t 0")

if __name__ == "__main__":
    main()
