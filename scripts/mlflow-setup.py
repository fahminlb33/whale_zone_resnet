from mlflow import MlflowClient
from mlflow.server import get_app_client

tracking_uri = "http://10.20.20.102:8009/"
target_user = "paus-dua"

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)


# create experiments and grant permissions
experiments = [
    # "paus_knn",
    # "paus_logistic_regression",
    # "paus_decision_tree",
    # "paus_random_forest",
    # "paus_gradient_boosting",
    # "paus_xgboost",
    # "paus_svm",
    "paus_ann",
    "paus_resnet"
]
for experiment_name in experiments:
    experiment_id = ""
    if experiment := client.get_experiment_by_name(experiment_name):
        print(f"Found experiment: {experiment_name}")
        experiment_id = experiment.experiment_id
    else:
        print(f"Creating experiment: {experiment_name}")
        experiment_id = client.create_experiment(experiment_name)

    try:
        auth_client.create_experiment_permission(
            experiment_id=experiment_id, username=target_user, permission="EDIT"
        )
    except Exception as e:
        print(e)
