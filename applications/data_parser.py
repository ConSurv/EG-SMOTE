from gsmote import EGSmote
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

class InputParser:

    @staticmethod
    def parse_input_zoo_data(filename):
        gsmote = EGSmote(random_state=1)
        df = pd.read_csv(filename)
        X = np.asarray(df.iloc[:, :-1].values)
        y = np.asarray(df.iloc[:, -1].values)
        X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # X_train, y_train = gsmote.fit_resample(X_t,y_t)
        smt = SMOTE()
        X_train, y_train = smt.fit_sample(X_t, y_t)
        classes = y_train.tolist()
        labels = y_train.tolist()
        input_database = {
            0: X_train
        }

        # input_data = pd.read_csv(filename, header=header)
        #
        # classes = input_data[17].tolist()
        # labels = input_data[0].tolist()
        # input_database = {
        #     0: input_data.as_matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        # }

        return input_database, labels, classes, X_test, y_test
