"""Class to compare performance with different classifiers"""
import sys
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
import xgboost as xgb
from gsmote import EGSmote
from gsmote.oldgsmote import OldGeometricSMOTE
from gsmote.comparison_testing.Evaluator import evaluate
import gsmote.comparison_testing.preprocessing as pp
from gsmote.comparison_testing.compare_visual import visualize_data as vs
import pandas as pd
from imblearn.over_sampling import SMOTE

sys.path.append('../../')

#  Directory
path = '../../data/'
# path = '/content/Modified-Geometric-Smote'

def logistic_training():

    # Fitting Simple Linear Regression to the Training set
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate("Logistic Regression", y_test, y_pred)


def gradient_boosting():

    # Fitting Gradient boosting
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    return evaluate("Gradient Boosting", y_test, y_pred)


def XGBoost():

    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)

    return evaluate("XGBoost", y_test, y_pred)


def KNN():

    # Fitting Simple Linear Regression to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test).astype(int)

    return evaluate("KNN", y_test, y_pred)


def decision_tree():

    # Fitting Simple Linear Regression to the Training set
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate("Decision Tree", y_test, y_pred)

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

    # for oldGSMOTE
    # GSMOTE = OldGeometricSMOTE()

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
    performance2 = gradient_boosting()
    performance3 = XGBoost()
    performance4 = KNN()
    performance5 = decision_tree()
    performance6 = GaussianMixture_model()

    labels = ["Classifier", "f_score", "g_mean", "auc_value"]
    values = [performance1, performance2, performance3, performance4, performance5, performance6]
    scores = pd.DataFrame(values, columns=labels)
    # scores.to_csv("../../output/scores_"+datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")+".csv")
    print(scores)

    import applications.main as gsom
    y_test, y_pred = gsom.run()
    gsom.evaluate(y_test, y_pred)





