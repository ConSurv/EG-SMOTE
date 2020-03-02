"""Class to compare performance with different classifiers"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from gsmote import GSMOTE, GeometricSMOTE
from gsmote.comparison_testing.Evaluator import evaluate
import gsmote.comparison_testing.preprocessing as pp
from gsmote.comparison_testing.compare_visual import  visualize_data as vs
import sys
import pandas as pd
import xgboost as xgb
import gsmote.comparison_testing.analysis as analyser

sys.path.append('../../')

# dataset
date_file = "../../data/KDD.csv".replace('\\', '/')

# data transformation if necessary.
X, y = pp.pre_process(date_file)

X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Visualize original data
vs(X_t, y_t, "Original data")

# oversample
print("Oversampling in progress...")
X_train, y_train = GSMOTE.OverSample(X_t, y_t)

# visualize oversampled data.
print("Oversampling completed.")
print("Plotting oversampled data...")

# use this line of code to interpret oversampled data.
# vs(X_train, y_train, "Oversampled ")

print("Plotting completed")

def linear_training():

    # Fitting Simple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_pred = np.where(y_predict > 0.5, 1, 0)

    return evaluate("Linear Regression", y_test, y_pred)


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


def MLPClassifier():

    # Fitting MLPClassifier to the Training set
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, solver='lbfgs', alpha=1e-5,
                        random_state=1)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test).astype(int)

    return evaluate("MLPClassifier", y_test, y_pred)


def Deep_One_Class_Classifier():
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=1)
    gmm.fit(X_train[y_train=='1'])

    OKscore = gmm.score_samples(X_train[y_train=='1'])
    threshold = OKscore.mean() -  1* OKscore.std()

    Trainer=np.column_stack((y_train[y_train=='1'],OKscore))
    Trainer = np.vstack((["y_train","Score"],Trainer))
    Train_Frame=pd.DataFrame(Trainer[1:,:],columns=Trainer[0,:])
    Train_Stat = Train_Frame["Score"]
    asd=pd.Series(OKscore).describe()

    score = gmm.score_samples(X_test)

    Tester = np.column_stack((y_test, score))


    Test_Frame = pd.DataFrame(Tester,columns=["y_test","Score"])
    Test_Stat = Test_Frame["Score"].describe()


    # majority_correct = len(score[(y_test == 1) & (score > thred)])
    y_pred = np.where(score > threshold,1,0)
    return evaluate("Deep_One_Cls_Classifier",y_test,y_pred)




performance1 = linear_training()
performance2 = gradient_boosting()
performance3 = XGBoost()
performance4 = KNN()
performance5 = decision_tree()
performance6 = MLPClassifier()
performance7 = Deep_One_Class_Classifier()


labels = ["Classifier", "f_score","g_mean","auc_value"]
values = [performance1,performance2,performance4,performance5,performance6,performance7]
scores = pd.DataFrame(values,columns=labels)
# scores.to_csv("../../output/scores_"+datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")+".csv")
print(scores)

