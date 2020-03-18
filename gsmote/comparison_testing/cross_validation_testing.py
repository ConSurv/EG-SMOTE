"""Class to compare performance with different classifiers"""
import sys

from imblearn.over_sampling import SMOTE
from sklearn import model_selection

sys.path.append('../../')
# sys.path.append('/content/Modified-Geometric-Smote/')
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from gsmote import EGSmote
from gsmote.oldgsmote import OldGeometricSMOTE
from gsmote.comparison_testing.Evaluator import evaluate2
import gsmote.comparison_testing.preprocessing as pp
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb


#  Directory
path = '../../data/'

def logistic_training(X_train,y_train,X_test,y_test):

    regressor = LogisticRegression(max_iter=150,C=2,solver='liblinear')
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)
    cm = confusion_matrix(y_test,y_predict)
    print(cm)
    return evaluate2("LR",y_test,y_pred)


def gradient_boosting(X_train,y_train,X_test,y_test):

    gbc = GradientBoostingClassifier(n_estimators=100)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate2("GBC",y_test, y_pred)



def XGBoost(X_train,y_train,X_test,y_test):

    # Fitting X-Gradient boosting

    #
    gbc = xgb.XGBClassifier(scale_pos_weight=99,missing=999999,max_depth=3,colsample_bytree=0.8)
    eval_set=[(X_train,y_train),(X_test,y_test)]
    gbc.fit(X_train, y_train,eval_metric="auc", eval_set=eval_set,verbose=True,early_stopping_rounds=5)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    return evaluate2("XGBoost", y_test, y_pred)


def KNN(X_train,y_train,X_test,y_test):

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    return evaluate2("KNN",y_test, y_pred)



def decision_tree(X_train,y_train,X_test,y_test):

    regressor = DecisionTreeClassifier(criterion="entropy",max_features="auto",min_samples_leaf=0.00005)
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    return evaluate2("DT",y_test, y_pred)


for filename in os.listdir(path):
    print (filename)
    # dataset
    data_file = path+filename.replace('\\', '/')
    # data transformation if necessary.
    X, y = pp.pre_process(data_file)
    # oversample
    print("---------------------------------------------------------")
    print("Dataset: " + filename)
    print("---------------------------------------------------------")

    kfold = model_selection.StratifiedKFold(n_splits=5, random_state=True, shuffle=True)
    scorings = []
    iter = 0
    for train_index, test_index in kfold.split(X, y):
        iter = iter + 1
        print("Itertion: "+ str(iter) + " =>   Processing" )
        X_t, X_test = X[train_index], X[test_index]
        y_t, y_test = y[train_index], y[test_index]
        #
        # GSMOTE = SMOTE()
        # GSMOTE = OldGeometricSMOTE()
        GSMOTE = EGSmote()
        X_train, y_train = GSMOTE.fit_resample(X_t, y_t)
        # X_train,y_train = X_t,y_t
        fold_score = []
        performance1 = logistic_training(X_train,y_train,X_test,y_test)
        # performance2 = gradient_boosting(X_train,y_train,X_test,y_test)
        # performance3 = XGBoost(X_train,y_train,X_test,y_test)
        # performance4 = KNN(X_train,y_train,X_test,y_test)
        # performance5 = decision_tree(X_train,y_train,X_test,y_test)


        fold_score.append(performance1)
        # fold_score.append(performance2)
        # fold_score.append(performance3)
        # fold_score.append(performance4)
        # fold_score.append(performance5)
        scorings.append(fold_score)

    scorings = np.asarray(scorings)

    values = []

    for i in range(scorings.shape[1]):

        LR_scores = scorings[:, i,:]

        classifier = LR_scores[0, 0]
        fscores = (LR_scores[:, 1].astype(np.float)).mean()
        gmean = ((LR_scores[:, 2]).astype(np.float)).mean()
        auc = ((LR_scores[:, 3]).astype(np.float)).mean()
        values.append([classifier,fscores,gmean,auc])

    labels = ["Classifier", "f_score","g_mean", "auc_value"]

    scores = pd.DataFrame(values, columns=labels)
    print(scores)
    # scores.to_csv("output/GSMOTE_scores_"+filename)
    # import applications.main as gsom
    # y_test, y_pred = gsom.run(data_file )
    # gsom.evaluate(y_test, y_pred)


