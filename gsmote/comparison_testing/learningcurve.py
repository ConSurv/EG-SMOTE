from gsmote.comparison_testing import preprocessing as pp
import pandas as pd
from gsmote import EGSmote
date_file = "../../data/KDDCUP0.csv"
X,y = pp.pre_process(date_file)
sm = EGSmote()
# X,y = sm.fit_resample(X,y)

train_sizes = [100,500,600,7000,1000,1500,2000,3000,10000,15000,20000,30000,40000,50000,60000,70000]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
train_sizes, train_scores, validation_scores = learning_curve(
        estimator = LinearRegression(),X = X,y = y, train_sizes = train_sizes, cv = 16, shuffle=True,scoring="f1")
print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))


import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(-1,0.1)
plt.show()