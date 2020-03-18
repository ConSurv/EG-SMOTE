import time
import sys
import os
from os.path import join
from datetime import datetime
sys.path.append('../../')
import numpy as np
import applications.data_parser as Parser
from params import params as Params
from core4 import core_controller as Core
from sklearn.metrics import confusion_matrix
import math


def generate_output_config(dataset, SF, forget_threshold,temporal_contexts, output_save_location):

    # Output data config
    output_save_filename = '{}_data_'.format(dataset)
    filename = output_save_filename + str(SF) + '_T_' + str(temporal_contexts) + '_mage_' + str(
        forget_threshold) + 'itr'
    plot_output_name = join(output_save_location, filename)

    # Generate output plot location
    output_loc = plot_output_name
    output_loc_images = join(output_loc, 'images/')
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    if not os.path.exists(output_loc_images):
        os.makedirs(output_loc_images)

    return output_loc, output_loc_images


def run(data_filename):
    SF = 0.83
    forget_threshold = 60  # To include forgetting, threshold should be < learning iterations.
    temporal_contexts = 1  # If stationary data - keep this at 1
    learning_itr = 100
    smoothing_irt = 50
    plot_for_itr = 4  # Unused parameter - just for visualization. Keep this as it is.

    # File Config
    dataset = 'anomaly'
    # data_filename = "../../data/NSLKDD-3.csv".replace('\\', '/')
    experiment_id = 'Exp-new-gsom-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    output_save_location = join('output/', experiment_id)

    # Init GSOM Parameters
    gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
                                          temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)
    generalise_params = Params.GeneraliseParameters(gsom_params)

    # Process the input files
    input_vector_database, labels, classes,X_test, y_test = Parser.InputParser.parse_input_zoo_data(data_filename)
    output_loc, output_loc_images = generate_output_config(dataset, SF, forget_threshold, temporal_contexts,
                                                           output_save_location)
    # Setup the age threshold based on the input vector length
    generalise_params.setup_age_threshold(input_vector_database[0].shape[0])
    # Process the clustering algorithm algorithm
    controller = Core.Controller(generalise_params)
    controller_start = time.time()
    result_dict, y_pred = controller.run(input_vector_database,X_test, plot_for_itr, classes, output_loc_images)

    return y_test, y_pred

def evaluate(classifier, Y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)
    interVal = (tp + fn) * (tn + fp)
    g_mean = math.sqrt(tp * tn / interVal)
    AUC = (tp / (tp + fn) + tn / (tn + fp)) / 2

    print(classifier, " finished executing")
    print("\nClassifier: " + classifier)
    print("f_score: " + str(f_score))
    print("g_mean: " + str(g_mean))
    print("AUC value: " + str(AUC) + "\n")

    return [classifier, f_score, g_mean, AUC]
#
y_test, y_pred =run('../data/CICID-mini.csv')
evaluate("GSOM_Classifier",y_test, np.array(y_pred).astype(int))
#
# print('Completed.')
