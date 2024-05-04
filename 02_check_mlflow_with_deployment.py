import random

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from utils import (
    create_confusion_matrix_plot,
    create_roc_auc_plot,
    get_classification_metrics,
    load_data,
    split_data,
)

# mlflow server --backend-store-uri sqlite:///mlruns_backend.db -h 0.0.0.0 -p 5000 

def create_experiment(
        experiment_name, 
        run_name=None, 
        params:dict=None, 
        metrics:dict=None, 
        tags:dict=None, 
        figures:list[plt.Figure] = None, 
        model = None,
        tracking_uri:str=None, 
        **kwargs
    ) -> int | None:
    """ Returns Run id """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        
    mlflow.set_experiment(experiment_name, **kwargs)
    
    with mlflow.start_run(run_name=run_name if run_name else None) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        if figures:
            for fig in figures:
                mlflow.log_artifacts(fig, artifact_path='plots')   
        if model:
            mlflow.sklearn.log_model(model, 'model')

        return run.info.run_id




def main(n_trials=3):
    x,y = load_data('load_iris')
    xtrain, xtest, ytrain,ytest = split_data(x,y, test_size=0.1, random_state=42)
    
    for i in range(n_trials):
        print("Trail  ", i+1,"/", n_trials)
        penalty = random.choice(['l1', 'l2', 'elasticnet', None])
        max_iter = random.choice([i for i in range(50, 200, 30)])
        multi_class = random.choice(['auto', 'ovr', 'multinomial'])
        tol= random.choice([1e-2, 1e-3,5e-3,1e-4,2e-4 ])
        C = random.choice(np.linspace(0, 2, 3))
        
        params =  {
                'penalty': penalty,
                'max_iter': max_iter,
                'multi_class': multi_class,
                'tol': tol,
                'C': C,
                }

        try:
            model = LogisticRegression(**params)
            
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            metrics = get_classification_metrics(ytest, ypred)
            print("params: ", params)
            print("train metrics: ", get_classification_metrics(ytrain, model.predict(xtrain)))
            print("test metrics: ", metrics)
            
            # roc_auc_fig = create_roc_auc_plot(model,xtest, ytest)
            confusion_matrix_fig = create_confusion_matrix_plot(model, xtest, ytest)
            print("confusion matrix: ", confusion_matrix_fig)
            
            experiment_name = 'logistic_regression-iris'
            run_id = create_experiment(
                experiment_name=experiment_name,
                run_name=f"log-zero-{i}",
                params=params,
                metrics=metrics,
                tags={
                    "version":i+1,
                    "class":'test_'+str(1)
                },
                model=model,
                tracking_uri='http://0.0.0.0:5000',
                
            )
            print("Run ID: ", run_id)
            print('\n')
        except Exception as e:
            
            print("Error: ", e)
    
def get_best_model(experiment_id, n=1):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_id, order_by=["metrics.f1 DESC"], max_results=n)
    return runs


def print_run_info(runs):
    for r in runs:
        print(f"run_id: {r.info.run_id}")
        print(f"lifecycle_stage: {r.info.lifecycle_stage}")
        print(f"metrics: {r.data.metrics}")
        # Exclude mlflow system tags
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        print(f"tags: {tags}")
        
        
def printt(x):
    print('+'*22)        
    print(x)
    print('-'*22)    
    print("type: ",type(x))
    print('-'*22)    
    print("dir: ",dir(x))
    try:    
        print('-'*22)
        print("class:", x.__class__.__name__)
    except:
        pass    
    
    try:    
        print('-'*22)
        print("len:", len(x))
    except:
        pass    
    
    print('+'*22)    
    # print('-'*22)    
    # print('-'*22)    
    # print('-'*22)    
    

            
if __name__=='__main__':
    # main()
    
    client = mlflow.client.MlflowClient(tracking_uri="http://0.0.0.0:5000")
    experiment = client.get_experiment_by_name("logistic_regression-iris")
    print("experiment_id:",experiment.experiment_id)
    printt(experiment)

    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"])

    # printt(runs)
    # print_run_info(runs)
    selected_run = runs[0]

    result = mlflow.register_model(
        f"runs:/{selected_run.info.run_id}/sklearn-model", "sk-learn-random-forest-reg"
    )
    printt(result)