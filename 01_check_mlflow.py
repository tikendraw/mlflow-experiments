import random

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression

from utils import get_classification_metrics, split_data


from utils import get_classification_metrics, split_data


def load_wine_data(info=False):
    data = load_wine()
    if info:
        print(data.DESCR)
        print(data.target_names)
    return pd.DataFrame(data.data), pd.array(data.target)


def main():
    x,y = load_wine_data()
    xtrain, xtest, ytrain,ytest = split_data(x,y, test_size=0.1, random_state=42)
    
    penalty = random.choice(['l1', 'l2', 'elasticnet', None])
    max_iter = random.choice([i for i in range(50, 200, 30)])
    multi_class = random.choice(['auto', 'ovr', 'multinomial'])
    tol= random.choice([1e-2, 1e-3,5e-3,1e-4,2e-4,3e-4,1e-5 ])
    C = random.choice(np.linspace(0, 2, 20))
    
    model = LogisticRegression(penalty=penalty, max_iter=max_iter, multi_class=multi_class, tol=tol, C=C)
    
    try:
        with mlflow.start_run() as run:
            
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            metrics = get_classification_metrics(ytest, ypred)
            print("metrics: ", metrics)
            mlflow.log_param('penalty', penalty)
            mlflow.log_param('max_iter', max_iter)
            mlflow.log_param('multi_class', multi_class)
            mlflow.log_param('tol', tol)
            mlflow.log_param('C', C)
            
            mlflow.log_metrics(metrics=metrics)
            
            mlflow.sklearn.log_model(model, 'logistic_reg')
    except Exception as e:
        print("Exception Occured!!", e)
            
            
            
if __name__=='__main__':
    main()
    print("Run `mlflow ui`")