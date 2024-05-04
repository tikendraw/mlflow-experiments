from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn import metrics

import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import datasets
import pandas as pd

def load_data(func_name, info=False):
    data = getattr(datasets, func_name)
    data = data()
    if info:
        print(data.DESCR)
        print(data.target_names)
    return pd.DataFrame(data.data), pd.array(data.target)


def split_data(x,y, test_size=0.1, random_state=42, **kwargs):
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=test_size, random_state=random_state, **kwargs)
    return xtrain, xtest, ytrain, ytest

def get_classification_metrics(y_true, y_pred):
    return {'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average='macro'),'recall': recall_score(y_true=y_true, y_pred=y_pred, average='macro'),'f1': f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    }


def create_roc_auc_plot(clf, X_data, y_data, save_path='roc_auc_curve.png'):
    """
    Create and save a Receiver Operating Characteristic (ROC) curve plot for the given classifier and data.

    Args:
        clf (object): A trained classifier object with a `predict_proba` method.
        X_data (array-like): Input data.
        y_data (array-like): True labels.
        save_path (str, optional): Path to save the ROC curve plot. Default is 'roc_auc_curve.png'.

    Returns:
        str: Absolute URI path of the saved ROC curve plot.
    """
    y_score = clf.predict_proba(X_data)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_data, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight')
    return str(Path(save_path).absolute())

def create_confusion_matrix_plot(clf, X_test, y_test, save_path='confusion_matrix.png'):
    """
    Create and save a confusion matrix plot for the given classifier and test data.

    Args:
        clf (object): A trained classifier object with a `predict` method and `classes_` attribute.
        X_test (array-like): Test input data.
        y_test (array-like): True test labels.
        save_path (str, optional): Path to save the confusion matrix plot. Default is 'confusion_matrix.png'.

    Returns:
        str: Absolute URI path of the saved confusion matrix plot.
    """
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.savefig(save_path, bbox_inches='tight')
    return str(Path(save_path).absolute())