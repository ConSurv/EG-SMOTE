"""Class to pre-process the input data"""

import sys
import pandas as pd
import numpy as np
sys.path.append('../../')

def pre_process(filename):
    df = pd.read_csv(filename)
    X = np.asarray(df.iloc[:, :-1].values)
    y = np.asarray(df.iloc[:,-1].values)
    return X,y



