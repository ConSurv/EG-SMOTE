"""Class to pre-process the input data"""

import sys
import pandas as pd

sys.path.append('../../')

def pre_process(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:,-1].values
    return X,y



