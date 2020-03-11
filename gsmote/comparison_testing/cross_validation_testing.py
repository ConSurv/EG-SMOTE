"""Class to compare performance with different classifiers"""
import datetime
import sys

from sklearn import model_selection

sys.path.append('../../')
# sys.path.append('/content/Modified-Geometric-Smote/')
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from gsmote import EGSmote
from gsmote.oldgsmote import OldGeometricSMOTE
from gsmote.comparison_testing.Evaluator import evaluate,evaluate2
import gsmote.comparison_testing.preprocessing as pp
from gsmote.comparison_testing.compare_visual import  visualize_data as vs
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score



sys.path.append('../../')

#  Directory
path = '../../data/'


def logistic_training():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    kfold = model_selection.StratifiedKFold(n_splits=2, random_state=True)

    regressor = LogisticRegression()

    # results = model_selection.cross_validate(estimator=regressor,
    #                                          X=X_train,
    #                                          y=y_train,
    #                                          cv=kfold,
    #                                          scoring=scoring)
    #
    # y_predict = model_selection.cross_val_predict(estimator=regressor, X=X_test, y=y_test, cv=kfold)
    # y_pred = np.where(y_predict > 0.5, 1, 0)

    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=True)

    scoreings = []

    for train_index, test_index in kfold.split(X,y):
        print("Train:", train_index, "Validation:", test_index)
        X_t, X_test = X[train_index], X[test_index]
        y_t, y_test = y[train_index], y[test_index]

        GSMOTE = OldGeometricSMOTE()
        X_train, y_train = GSMOTE.fit_resample(X_t, y_t)
        regressor = LogisticRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_predict = regressor.predict(X_test)
        y_pred = np.where(y_predict > 0.5, 1, 0)

        scoreings.append(evaluate2(y_test,y_pred))
    scoreings = np.asarray(scoreings)
    fscores = scoreings[:,0]
    gmean = scoreings[:,1]
    auc = scoreings[:,2]

    return ["LR",fscores.mean(),gmean.mean(),auc.mean()]

    # print("fscores ",fscores.mean(),' ',fscores[fscores.argmax()])
    # print("gmean ",gmean.mean(),' ',gmean[gmean.argmax()])
    # print("auc ",auc.mean(),' ',auc[auc.argmax()])

    # Fitting Simple Linear Regression to the Training set
    # regressor2 = LogisticRegression()
    # regressor2.fit(X_train, y_train)
    #
    # # Predicting the Test set results
    # y_predict2 = regressor2.predict(X_test)
    # y_pred2 = np.where(y_predict2 > 0.5, 1, 0)
    #
    # return evaluate("LR", y_test, y_pred,y_pred2)


def gradient_boosting():
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=True)

    scoreings = []

    for train_index, test_index in kfold.split(X, y):
        print("Train:", train_index, "Validation:", test_index)
        X_t, X_test = X[train_index], X[test_index]
        y_t, y_test = y[train_index], y[test_index]

        GSMOTE = OldGeometricSMOTE()
        X_train, y_train = GSMOTE.fit_resample(X_t, y_t)
        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
        gbc.fit(X_train, y_train)

        # Predicting the Test set results
        y_predict = gbc.predict(X_test)
        y_pred = np.where(y_predict > 0.5, 1, 0)

        scoreings.append(evaluate2(y_test, y_pred))

    scoreings = np.asarray(scoreings)
    fscores = scoreings[:, 0]
    gmean = scoreings[:, 1]
    auc = scoreings[:, 2]

    return ["GBC", fscores.mean(), gmean.mean(), auc.mean()]

    # gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
    #
    # results = model_selection.cross_validate(estimator=gbc,
    #                                          X=X_train,
    #                                          y=y_train,
    #                                          cv=kfold,
    #                                          scoring=scoring)
    #
    # y_predict = model_selection.cross_val_predict(estimator=gbc, X=X_test, y=y_test, cv=kfold)
    # y_pred = np.where(y_predict > 0.5, 1, 0)
    #
    #
    # # Fitting Gradient boosting
    # gbc2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
    # gbc2.fit(X_train, y_train)
    #
    # # Predicting the Test set results
    # y_predict2 = gbc2.predict(X_test)
    # y_pred2 = np.where(y_predict2.astype(int) > 0.5, 1, 0)
    #
    # return evaluate("GBC", y_test, y_pred,y_pred2)


# def XGBoost():
#
#     # Fitting X-Gradient boosting
#     gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
#     gbc.fit(X_train, y_train)
#
#     # Predicting the Test set results
#     y_predict = gbc.predict(X_test)
#     y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)
#
#     return evaluate("XGBoost", y_test, y_pred)


def KNN():

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    kfold = model_selection.KFold(n_splits=10, random_state=True)

    # Fitting Simple Linear Regression to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    results = model_selection.cross_validate(estimator=classifier,
                                              X=X_train,
                                              y=y_train,
                                              cv=kfold,
                                              scoring=scoring)
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=True)

    scoreings = []

    for train_index, test_index in kfold.split(X, y):
        print("Train:", train_index, "Validation:", test_index)
        X_t, X_test = X[train_index], X[test_index]
        y_t, y_test = y[train_index], y[test_index]

        GSMOTE = OldGeometricSMOTE()
        X_train, y_train = GSMOTE.fit_resample(X_t, y_t)
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)


        scoreings.append(evaluate2(y_test, y_pred))

    scoreings = np.asarray(scoreings)
    fscores = scoreings[:, 0]
    gmean = scoreings[:, 1]
    auc = scoreings[:, 2]

    return ["KNN", fscores.mean(), gmean.mean(), auc.mean()]



    # print("mean_acc = ", np.mean(results['test_accuracy']))
    # print("std_acc = ",np.std(results['test_accuracy']))
    #
    # print("mean_precision = ", np.mean(results['test_precision']))
    # print("std_precision = ", np.std(results['test_precision']))
    #
    # print("mean_recall = ", np.mean(results['test_recall']))
    # print("std_recall = ", np.std(results['test_recall']))
    #
    # print("mean_f1 = ", np.mean(results['test_f1_score']))
    # print("std_f1 = ", np.std(results['test_f1_score']))


    # y_pred = model_selection.cross_val_predict(estimator=classifier,X=X_test,y=y_test,cv=kfold)
    #
    #
    #
    #
    # classifier2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    #
    # classifier2.fit(X_train, y_train)
    #
    # # Predicting the Test set results
    # y_pred2 = classifier2.predict(X_test).astype(int)
    #
    # return evaluate("KNN", y_test, y_pred,y_pred2)


def decision_tree():
    # scoring = {'accuracy': make_scorer(accuracy_score),
    #            'precision': make_scorer(precision_score),
    #            'recall': make_scorer(recall_score),
    #            'f1_score': make_scorer(f1_score)}
    #
    # kfold = model_selection.KFold(n_splits=10, random_state=True)
    #
    # regressor = DecisionTreeRegressor()
    #
    # results = model_selection.cross_validate(estimator=regressor,
    #                                          X=X_train,
    #                                          y=y_train,
    #                                          cv=kfold,
    #                                          scoring=scoring)
    #
    # y_predict = model_selection.cross_val_predict(estimator=regressor, X=X_test, y=y_test, cv=kfold)
    # y_pred = np.where(y_predict > 0.5, 1, 0)

    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=True)

    scoreings = []

    for train_index, test_index in kfold.split(X, y):
        print("Train:", train_index, "Validation:", test_index)
        X_t, X_test = X[train_index], X[test_index]
        y_t, y_test = y[train_index], y[test_index]

        GSMOTE = OldGeometricSMOTE()
        X_train, y_train = GSMOTE.fit_resample(X_t, y_t)
        regressor = DecisionTreeRegressor()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_predict = regressor.predict(X_test)
        y_pred = np.where(y_predict > 0.5, 1, 0)

        scoreings.append(evaluate2(y_test, y_pred))

    scoreings = np.asarray(scoreings)
    fscores = scoreings[:, 0]
    gmean = scoreings[:, 1]
    auc = scoreings[:, 2]

    return ["DT", fscores.mean(), gmean.mean(), auc.mean()]

    # # Fitting Simple Linear Regression to the Training set
    # regressor2 = DecisionTreeRegressor()
    # regressor2.fit(X_train, y_train)
    #
    # # Predicting the Test set results
    # y_predict2 = regressor2.predict(X_test)
    # y_pred2 = np.where(y_predict2 > 0.5, 1, 0)
    #
    # return evaluate("DT", y_test, y_pred,y_pred2)

def GaussianMixture_model():
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=1)
    gmm.fit(X_train[y_train==0])

    OKscore = gmm.score_samples(X_train[y_train==0])
    threshold = OKscore.mean() -  1* OKscore.std()


    score = gmm.score_samples(X_test)


    # majority_correct = len(score[(y_test == 1) & (score > thred)])
    y_pred = np.where(score < threshold,1,0)
    return evaluate("GaussianMixture_model",y_test,y_pred)


for filename in os.listdir(path):

    # dataset
    date_file = path+filename.replace('\\', '/')

    # data transformation if necessary.
    X, y = pp.pre_process(date_file)

    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Visualize original data
    # vs(X_t, y_t, "Original data")

    # oversample
    print("---------------------------------------------------------")
    print("Dataset: " + filename)
    print("Oversampling in progress ...")
    GSMOTE = EGSmote()
    X_train, y_train = GSMOTE.fit_resample(X_t, y_t)

    # For SMOTE
    # sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    # X_train, y_train = sm.fit_resample(X_t, y_t)


    # visualize oversampled data.
    print("Oversampling completed.")
    print("Plotting oversampled data...")

    # use this line of code to interpret oversampled data.
    # vs(X_train, y_train, "Oversampled ")

    print("Plotting completed")

    performance1 = logistic_training()
    # performance2 = gradient_boosting()
    # performance3 = XGBoost()
    # performance4 = KNN()
    # performance5 = decision_tree()
    # performance6 = MLPClassifier()
    # performance7 = GaussianMixture_model()

    labels = ["Classifier", "f_score", "f_score2","g_mean","g_mean2", "auc_value","auc_value2"]
    # values = [performance1, performance2, performance4, performance5]
    values = [performance1]

    scores = pd.DataFrame(values, columns=labels)
    # scores.to_csv("../../output/scores_"+datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")+".csv")
    print(scores)

    # import applications.main as gsom
    # y_test, y_pred = gsom.run()
    # gsom.evaluate(y_test, y_pred)





