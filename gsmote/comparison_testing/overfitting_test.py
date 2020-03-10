import os
import time

from pandas import np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import gsmote.comparison_testing.preprocessing as pp

from gsmote import EGSmote
from gsmote.oldgsmote import OldGeometricSMOTE


def gbc_overfitting(date_file,X_train, y_train,X_test, y_test):
    n_gb = []
    score_gb = []
    time_gb = []
    n_gbes = []
    score_gbes = []
    time_gbes = []

    n_estimators = 100

    # We specify that if the scores don't improve by atleast 0.01 for the last
    # 10 stages, stop fitting additional stages
    gbes = GradientBoostingClassifier(n_estimators=n_estimators,
                                      validation_fraction=0.2,
                                      n_iter_no_change=10, tol=0.01,
                                      random_state=0)
    gb = GradientBoostingClassifier(n_estimators=n_estimators,
                                    random_state=0)
    start = time.time()
    gb.fit(X_train, y_train)
    time_gb.append(time.time() - start)

    start = time.time()
    gbes.fit(X_train, y_train)
    time_gbes.append(time.time() - start)

    score_gb.append(gb.score(X_test, y_test))
    score_gbes.append(gbes.score(X_test, y_test))

    n_gb.append(gb.n_estimators_)
    n_gbes.append(gbes.n_estimators_)

    bar_width = 0.2
    n = 1
    index = np.arange(0, n * bar_width, bar_width) * 2.5
    index = index[0:n]

    plt.figure(figsize=(9, 5))

    bar1 = plt.bar(index, score_gb, bar_width, label='Without early stopping',
                   color='crimson')
    bar2 = plt.bar(index + bar_width, score_gbes, bar_width,
                   label='With early stopping', color='coral')

    plt.xticks(index + bar_width, date_file)
    plt.yticks(np.arange(0, 1.3, 0.1))

    def autolabel(rects, n_estimators):
        """
        Attach a text label above each bar displaying n_estimators of each model
        """
        for i, rect in enumerate(rects):
            plt.text(rect.get_x() + rect.get_width() / 2.,
                     1.05 * rect.get_height(), 'n_est=%d' % n_estimators[i],
                     ha='center', va='bottom')

    autolabel(bar1, n_gb)
    autolabel(bar2, n_gbes)

    plt.ylim([0, 1.3])
    plt.legend(loc='best')
    plt.grid(True)

    plt.xlabel('Datasets')
    plt.ylabel('Test score')

    plt.show()

def decision_tree():
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities



    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()

#  Directory
path = '../../data/'
for filename in os.listdir(path):
    # dataset
    date_file = path + filename.replace('\\', '/')

    # data transformation if necessary.
    X, y = pp.pre_process(date_file)

    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Visualize original data
    # vs(X_t, y_t, "Original data")

    # oversample
    print("------------------------------------------------------")
    print("Dataset: " + filename)
    print("Oversampling in progress ...")

    # for GeometricSMOTE
    GSMOTE = OldGeometricSMOTE()
    X_train, y_train = GSMOTE.fit_resample(X_t, y_t)

    #  for GeometricSMOTE
    # GSMOTE = EGSmote()
    # X_train, y_train = GSMOTE.fit_resample(X_t, y_t)

    # For SMOTE
    # sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    # X_train, y_train = sm.fit_resample(X_t, y_t)

    print("overSampled.")

decision_tree()