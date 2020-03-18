# import sys
# sys.path.append('/content/pygsom/')

from sklearn.base import BaseEstimator, ClassifierMixin
import gsmote.comparison_testing.preprocessing as pp
from gsmote import EGSmote
from gsmote.oldgsmote import OldGeometricSMOTE
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append('../../')


class MeanClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(
            self,
            sampling_rate=0.3,
            min_samples_leaf=0.002,
            criterion="entropy",
            max_features="auto",
            C=0.001,
            solver='saga',
            max_iter=100,
            max_depth=3,
            min_samples_split=0.005

    ):
        """
        Called when initializing the classifier
        """
        self.criterion = criterion
        self.sampling_rate = sampling_rate

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.regressor = GradientBoostingClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,max_features=max_features)

        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        # self.regressor = DecisionTreeClassifier(max_features=max_features,criterion=criterion)

        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        # self.regressor = LogisticRegression(C=C,solver=solver,max_iter=150)

        # self.gsmote = EGSmote(sampling_rate= self.sampling_rate)
        self.gsmote = SMOTE()
        # self.gsmote = OldGeometricSMOTE()

    def fit(self, X, y):
        X_train, y_train = self.gsmote.fit_resample(X, y)
        self.regressor.fit(X_train, y_train)
        return self

    # def _meaning(self, x):
    #     return True

    def predict(self, y):
        return self.regressor.predict(y)

    # def score(self, X, y=None):
    #     # counts number of values bigger than mean
    #     return(sum(self.predict(X)))
    #


from sklearn.model_selection import GridSearchCV

date_file = "../../data/KDDCUP-10000.csv".replace('\\', '/')
# date_file = "content/pygsom/data/ecoli.csv".replace('\\','/')

X, y = pp.pre_process(date_file)
# try different combination of hyper paramenters

# gbc parameters = [{'min_samples_split':[0.005,0.01],'min_samples_leaf':[0.001,0.002],'max_feature':['auto','sqrt',
# 'log2'],'learning_rate':[0.05,0.1,0.2],'max_depth':[3,4,5,6,7]}]
parameters = [{'min_samples_split': [0.005,0.007,0.01], 'max_depth': [3,4,5,6,7,8]}]

# DT
parameters = [{'max_features':['auto','sqrt','log2'],'criterion':["entropy",'gini'],}]

# LR
parameters = [{'C':[1, 10, 100, 1000],'solver':['liblinear', 'saga']}]

gs = GridSearchCV(MeanClassifier(), parameters)
gs.fit(X, y)

params = gs.best_params_
print(params)
#
# 0.990727  0.995339   0.995348
# 0         DT  0.997427  0.999745   0.999745
# DT
# 0.991699
# 0.994965
# 0.994986