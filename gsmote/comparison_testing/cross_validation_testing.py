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
from sklearn.tree import DecisionTreeRegressor
from gsmote import EGSmote
from gsmote.oldgsmote import OldGeometricSMOTE
from gsmote.comparison_testing.Evaluator import evaluate,evaluate2
import gsmote.comparison_testing.preprocessing as pp
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

sys.path.append('../../')

#  Directory
path = '../../data/'


def logistic_training(X_train,y_train,X_test,y_test):

    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate2("LR",y_test,y_pred)


def gradient_boosting(X_train,y_train,X_test,y_test):

    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate2("GBC",y_test, y_pred)



def XGBoost(X_train,y_train,X_test,y_test):

    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    return evaluate2("XGBoost", y_test, y_pred)


def KNN(X_train,y_train,X_test,y_test):

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    return evaluate2("KNN",y_test, y_pred)



def decision_tree(X_train,y_train,X_test,y_test):

    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate2("DT",y_test, y_pred)




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

    # X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Visualize original data
    # vs(X_t, y_t, "Original data")

    # oversample
    print("---------------------------------------------------------")
    print("Dataset: " + filename)
    # print("Oversampling in progress ...")
    # GSMOTE = EGSmote()
    # X_train, y_train = GSMOTE.fit_resample(X_t, y_t)

    # For SMOTE
    # sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    # X_train, y_train = sm.fit_resample(X_t, y_t)


    # visualize oversampled data.
    # print("Oversampling completed.")
    # print("Plotting oversampled data...")
    #
    # # use this line of code to interpret oversampled data.
    # # vs(X_train, y_train, "Oversampled ")
    #
    # print("Plotting completed")


    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=True)

    scoreings = []

    for train_index, test_index in kfold.split(X, y):
        # print("Train:", train_index, "Validation:", test_index)
        X_t, X_test = X[train_index], X[test_index]
        y_t, y_test = y[train_index], y[test_index]

        GSMOTE = EGSmote()
        X_train, y_train = GSMOTE.fit_resample(X_t, y_t)

        fold_score = []
        performance1 = logistic_training(X_train,y_train,X_test,y_test)
        performance2 = gradient_boosting(X_train,y_train,X_test,y_test)
        performance3 = XGBoost(X_train,y_train,X_test,y_test)
        performance4 = KNN(X_train,y_train,X_test,y_test)
        performance5 = decision_tree(X_train,y_train,X_test,y_test)
        # performance6 = MLPClassifier()
        # performance7 = GaussianMixture_model()

        fold_score.append(performance1)
        fold_score.append(performance2)
        fold_score.append(performance3)
        fold_score.append(performance4)
        fold_score.append(performance5)
        scoreings.append(fold_score)


    scoreings = np.asarray(scoreings)

    values = []

    for i in range(scoreings.shape[1]):

        LR_scores = scoreings[:, i,:]

        classifier = LR_scores[0,0]
        fscores = (LR_scores[:,1].astype(np.float)).mean()
        gmean = ((LR_scores[:, 2]).astype(np.float)).mean()
        auc = ((LR_scores[:, 3]).astype(np.float)).mean()
        values.append([classifier,fscores,gmean,auc])










    # performance1 = logistic_training()
    # performance2 = gradient_boosting()
    # # performance3 = XGBoost()
    # performance4 = KNN()
    # performance5 = decision_tree()
    # # performance6 = MLPClassifier()
    # # performance7 = GaussianMixture_model()
    #
    # print(performance1)
    # print(performance2)
    # print(performance4)
    # print(performance5)
    # # labels = ["Classifier", "f_score", "f_score2","g_mean","g_mean2", "auc_value","auc_value2"]
    labels = ["Classifier", "f_score","g_mean", "auc_value"]
    #
    # values = [performance1, performance2, performance4, performance5]
    # # values = [performance1]
    #
    scores = pd.DataFrame(values, columns=labels)
    # # scores.to_csv("../../output/scores_"+datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")+".csv")
    print(scores)

    # import applications.main as gsom
    # y_test, y_pred = gsom.run()
    # gsom.evaluate(y_test, y_pred)





