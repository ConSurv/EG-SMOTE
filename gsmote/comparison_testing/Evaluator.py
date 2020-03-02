"""Class to perform evaluation based on metrics"""

from sklearn.metrics import confusion_matrix
import math

def evaluate(classifier,Y_test,y_pred):

    # create a confusion matrix from prediction
    tn, fp, fn, tp = confusion_matrix(Y_test.astype(int), y_pred).ravel()

    # define metrics
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f_score = 2*precision*recall /(precision+recall)
    g_mean = math.sqrt(tp*tn/(tp+fn)*(tn+fp))
    AUC = (tp/(tp+fn)+tn/(tn+fp))/2

    print(classifier," finished executing.")
    return [classifier,f_score,g_mean,AUC]