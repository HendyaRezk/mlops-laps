import subprocess

def run_preprocessing():
    result = subprocess.run(["python", "/teamspace/studios/this_studio/mlops/src/data/preprocess_data.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Preprocessing failed:")
        print(result.stderr)
        exit(1)

def run_training():
    result = subprocess.run(["python", "/teamspace/studios/this_studio/mlops/src/model/train_model.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Model training failed:")
        print(result.stderr)
        exit(1)

def run_evaluation():
    result = subprocess.run(["python", "/teamspace/studios/this_studio/mlops/src/evaluate.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Evaluation failed:")
        print(result.stderr)
        exit(1)
    print("Evaluation complete.\n")

if __name__ == "__main__":
    run_preprocessing()
    run_training()
    run_evaluation()
    print("steps completed")
