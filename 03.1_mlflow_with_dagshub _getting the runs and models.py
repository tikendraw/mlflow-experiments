import os
import random

import mlflow
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from utils import (
    create_confusion_matrix_plot,
    get_classification_metrics,
    load_data,
    split_data,
)

load_dotenv('./.env')

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']

def get_best_model(experiment_id, n=1):
    client = mlflow.tracking.MlflowClient()
    return client.search_runs(experiment_id, order_by=["metrics.f1 DESC"], max_results=n)
    

def print_run_info(runs:list):
    for r in runs:
        print(f"run_id: {r.info.run_id}")
        print(f"lifecycle_stage: {r.info.lifecycle_stage}")
        print(f"metrics: {r.data.metrics}")
        # Exclude mlflow system tags
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        print(f"tags: {tags}")
        
        

import sys
import inspect

from rich import print as rprint

def printt(x, print_variable=True):
    """Prints details about a variable."""
    def get_var_name(var):
        """
        Returns the name of a variable as a string.
        """
        for name, val in globals().items():
            if val is var:
                return name
        return None


    just_print = {
        "name": get_var_name(x),
        "variable": x if print_variable else "print variable is disabled",
        "class": x.__class__.__name__,
    }

    use_try = {
        "type": type,
        "dir": dir,
        "len": len,
        "Memory Usage(B)":sys.getsizeof
    }

    rprint("-" * 60)
    for i, j in just_print.items():
        rprint(f"[bold blue]{i:15} ::[/bold blue] {j}")
        rprint("-" * 60)

    for i, j in use_try.items():
        try:
            rprint(f"[bold magenta]{i:15} ::[/bold magenta] {j(x)}")
            rprint("-" * 60)
        except Exception as e:
            rprint(f"[bold red]{i:15} ::[/bold red] not printed due to {e} !!!")
            rprint("-" * 60)

    rprint()


    
def main():
    logged_model = 'runs:/06ad5c8180414d4d9b40774c5cbbab32/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    import pandas as pd
    data = [
        [6.3, 3.3, 4.7, 1.6],
        [6.5, 3.0, 5.8, 2.2],
        [5.6, 2.5, 3.9, 1.1],
        [5.7, 2.8, 4.5, 1.3],
        [6.4, 2.8, 5.6, 2.2]
        ]
    print("These are actual labels: [1, 2, 1, 1, 2]")
    
    return loaded_model.predict(pd.DataFrame(data))
            
if __name__=='__main__':
    
    client = mlflow.client.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name("dagshub_logistic_regression-iris")
    print("experiment_id:",experiment.experiment_id)
    # printt(experiment)

    out = main()
    printt(out)

    # runs = client.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"])

    # # printt(runs)

    # selected_run = runs[0]
    # # printt(selected_run)
    # print_run_info([selected_run])
    
    # # result = mlflow.register_model(
    # #     f"runs:/{selected_run.info.run_id}/sklearn-model", "sk-learn-random-forest-reg"
    # # )
    # # printt(result)