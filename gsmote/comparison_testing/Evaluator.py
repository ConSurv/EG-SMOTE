"""Class to perform evaluation based on metrics"""

from sklearn.metrics import confusion_matrix
import math

def evaluate(classifier,Y_test,y_pred,y_pred2):

    # create a confusion matrix from prediction
    tn, fp, fn, tp = confusion_matrix(Y_test.astype(int), y_pred).ravel()

    # define metrics
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f_score = 2*precision*recall /(precision+recall)
    g_mean = math.sqrt(tp*tn/((tp+fn)*(tn+fp)))
    AUC = (tp/(tp+fn)+tn/(tn+fp))/2

    tn2, fp2, fn2, tp2 = confusion_matrix(Y_test.astype(int), y_pred2).ravel()

    # define metrics
    precision2 = tp2 / (tp2 + fp2)
    recall2 = tp2 / (tp2 + fn2)
    f_score2 = 2 * precision2 * recall2 / (precision2 + recall2)
    g_mean2 = math.sqrt(tp2 * tn2 / ((tp2 + fn2) * (tn2 + fp2)))
    AUC2 = (tp2 / (tp2 + fn2) + tn2 / (tn2 + fp2)) / 2

    print(classifier, " finished executing")
    print("\nClassifier: " + classifier)
    print("f_score: " + str(f_score)," ",f_score2)
    print("g_mean: " + str(g_mean)," ",g_mean2)
    print("AUC value: " + str(AUC)," ",AUC2 , "\n")

    print(classifier," finished executing.")
    return [classifier,f_score,f_score2,g_mean,g_mean2,AUC,AUC2]


def evaluate_Comparison(classifier,Y_test,y_pred):

    # create a confusion matrix from prediction
    tn, fp, fn, tp = confusion_matrix(Y_test.astype(int), y_pred).ravel()

    # define metrics
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f_score = 2*precision*recall /(precision+recall)
    g_mean = math.sqrt(tp*tn/((tp+fn)*(tn+fp)))
    AUC = (tp/(tp+fn)+tn/(tn+fp))/2

    print("|************************classifier************************|")
    print("\nClassifier: " + classifier)
    print("f_score: " + str(f_score))
    print("g_mean: " + str(g_mean))
    print("AUC value: " + str(AUC))

    print(classifier," finished executing.")
    return [classifier,f_score,g_mean,AUC]