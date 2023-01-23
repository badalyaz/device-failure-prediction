import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling
from datetime import datetime
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

from imblearn.metrics import geometric_mean_score as geo
from imblearn.metrics import make_index_balanced_accuracy as iba
from imblearn.metrics import geometric_mean_score, make_index_balanced_accuracy, classification_report_imbalanced

# import model for imbalanced data set
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import geometric_mean_score, make_index_balanced_accuracy, classification_report_imbalanced
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# change activedays datatype to numerical
def str_to_num(str):
    return str.split(' ')[0]

def cross_val_fit_pred(X_train, y_train, algorithms, names, ros):
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    Geo_score = []
#     Iba_score = []
    Accuracy = []
    F1 = []
    Recall = []
    Prec = []
    for i in range(len(algorithms)):
        j=1
        kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
        geo_score = []
#         iba_score = []
        accuracy = []
        f1 = []
        recall = []
        prec = []
        for train_index,test_index in kf.split(X_train, y_train):
            xtr,xvd=X_train[train_index],X_train[test_index]
            ytr,yvd=y_train[train_index],y_train[test_index]
            xtr_res,ytr_res=ros.fit_resample(xtr, ytr)
            algorithms[i] = algorithms[i].fit(xtr_res,ytr_res)
            y_pred_test = algorithms[i].predict(xvd).round()
            accuracy.append(accuracy_score(yvd, y_pred_test))
            geo_score.append(geo(yvd, y_pred_test))
#             iba_score.append(iba(yvd, y_pred_test))
            f1.append(f1_score(yvd, y_pred_test,average='macro'))
            recall.append(recall_score(yvd, y_pred_test,average='macro'))
            prec.append(precision_score(yvd, y_pred_test))
            j +=1
        mean_ac = np.mean(accuracy)
        mean_geo = np.mean(geo_score)
        mean_f1 = np.mean(f1)
#         mean_iba = np.mean(iba_score)
        mean_recall = np.mean(recall)
        mean_prec = np.mean(prec)
        F1.append(mean_f1)
        Geo_score.append(mean_geo)
#         Iba_score.append(mean_iba)
        Accuracy.append(mean_ac)
        Recall.append(mean_recall)
        Prec.append(mean_prec)

    metrics = pd.DataFrame(columns = ['Accuracy','geo_score','f1','recall','prec'],index=names)
    metrics['Accuracy']=Accuracy
    metrics['geo_score']=Geo_score
#     metrics['iba_score']=Iba_score
    metrics['f1']=F1
    metrics['recall']=Recall
    metrics['prec']=Prec
    return metrics.sort_values('geo_score',ascending=False)