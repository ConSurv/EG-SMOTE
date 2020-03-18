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
import applications.main as gsom

sys.path.append('../../')


class MeanClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(
            self,
            sampling_rate=0.3,


    ):
        """
        Called when initializing the classifier
        """
        self.data_file = "../../data/KDDCUP-10000.csv".replace('\\', '/')
        self.sampling_rate = sampling_rate
        self.gsom = gsom.run(data_filename=self.data_file)

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
# Enter file name


# try different combination of hyper paramenters
parameters = [{'sampling_rate':[0.1,0.2]}]


gs = GridSearchCV(MeanClassifier(), parameters)
gs.fit(X, y)

params = gs.best_params_
print(params)
