

# import sys
# sys.path.append('/content/pygsom/')

from sklearn.base import BaseEstimator, ClassifierMixin
import gsmote.comparison_testing.preprocessing as pp
from gsmote import GeometricSMOTE
from sklearn.ensemble import GradientBoostingClassifier
import sys

sys.path.append('../../')

class MeanClassifier(BaseEstimator,ClassifierMixin):
    """An example of classifier"""

    def __init__(
            self,
            truncation_factor=1.0,
            deformation_factor=0.0,
            k_neighbors=1,
            sampling_rate=0.3,
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3

    ):
        """
        Called when initializing the classifier
        """
        self.truncation_factor = truncation_factor
        self.deformation_factor = deformation_factor
        self.k_neighbors = k_neighbors
        self.sampling_rate = sampling_rate
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.regressor = GradientBoostingClassifier (n_estimators=self.n_estimators, learning_rate = self.learning_rate,
                                                     max_depth = self.max_depth)
        self.gsmote = GeometricSMOTE(random_state=1, truncation_factor=self.truncation_factor,
                                     deformation_factor=self.deformation_factor, k_neighbors=self.k_neighbors,
                                     sampling_rate= self.sampling_rate)

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

date_file = "../../data/adultmini.csv".replace('\\','/')
# date_file = "content/pygsom/data/ecoli.csv".replace('\\','/')

X,y = pp.preProcess(date_file)
#try different combination of hyper paramenters
parameters = [{'truncation_factor':[-1],'deformation_factor':[0],'k_neighbors':[3],
               'sampling_rate':[0.3], 'n_estimators':[1000], 'learning_rate':[0.01],'max_depth':[3]}]
gs = GridSearchCV(MeanClassifier(), parameters)
gs.fit(X,y)

params = gs.best_params_
print (params)
#
# #find performance
# X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
# gsmote = GeometricSMOTE(random_state=1, truncation_factor=params["truncation_factor"],
#                                      deformation_factor=params["deformation_factor"], k_neighbors=params["k_neighbors"],
#                                      sampling_rate=params["sampling_rate"])
# X_train,y_train = gsmote.fit_resample(X_t,y_t)
# # Fitting Gradient boosting
# gbc = GradientBoostingClassifier (n_estimators=params["n_estimators"], learning_rate = params["learning_rate"],
#                                   max_depth = params["max_depth"])
# gbc.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_predict = gbc.predict(X_test)
# y_pred = np.where(y_predict.astype(int)>0.5,1,0)
#
# evaluate("Gradient Boosting",y_test,y_pred)