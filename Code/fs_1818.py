import pandas as pd
from dfo_1818 import dfo
from models_1818 import regression_models
import numpy as np
import tikzplotlib

print("RANDOM FOREST ANALYSIS \n")
# print("LINEAR REGRESSION ANALYSIS \n")


def linear_regression_data():
    call = dfo(N, D, delta, maxIterations, lowerB, upperB, X, fitness)
    call.initialize()
    call.loop()
    best_agent = call.eval()
    call.plot(best_agent)
    call.plot_train_test_losses_lr()

    return list(best_agent), call.dataframe


def random_forest_data():
    call = dfo(N, D, delta, maxIterations, lowerB, upperB, X, fitness)
    call.initialize()
    call.loop()
    best_agent = call.eval()
    call.plot(best_agent)
    call.plot_train_test_losses_rf()

    return list(best_agent), call.dataframe


def feature_selection(orginal_data, descending_data, perc, dataset):
    percentage_of_data = perc
    array_data_percentage = int((percentage_of_data / 100) * len(descending_data))

    descending_partial_features = descending_data[0: array_data_percentage]
    other_features = descending_data[array_data_percentage:]

    high_features_index = []
    low_features_index = []

    for ele in descending_partial_features:
        high_features_index.append(orginal_data.index(ele))

    for ele in other_features:
        low_features_index.append(orginal_data.index(ele))

    # features list has the index numbers of classes to be dropped
    target, target_values = dataset.columns[0], dataset.iloc[:, 0].values
    dataset = dataset.drop([str(dataset.columns[0])], axis=1)

    # Updated dataframe READY!
    dataset.drop(dataset.columns[low_features_index], axis=1, inplace=True)
    dataset.insert(0, target, target_values)

    Revised_file = 'Revised_FS.csv'
    dataset.to_csv(Revised_file, index=False)

    call = regression_models(Revised_file)
    x, y, df = call.process_data()
    call.splitting_data(x, y)
    # call.linear_regression()
    call.random_forest()

    # call.lr_train_test_data()
    call.rf_train_test_data()

    print("-------------------------------------------------")
    print("Training Analysis After Feature Selection: \n", call.rf_training_Data,
          "\nTesting Analysis After Feature Selection: \n", call.rf_testing_Data)

    return high_features_index


if __name__ == "__main__":
    file_name = 'AI_app_dataset.csv'
    N = 100  # POPULATION SIZE
    D = 25  # DIMENSIONALITY
    delta = 0.001  # DISTURBANCE THRESHOLD
    maxIterations = 20  # ITERATIONS ALLOWED
    lowerB = [0] * D  # LOWER BOUND (IN ALL DIMENSIONS)
    upperB = [1] * D  # UPPER BOUND (IN ALL DIMENSIONS)

    # INITIALISATION PHASE
    X = np.empty([N, D])  # EMPTY FLIES ARRAY OF SIZE: (N,D)
    fitness = [None] * N  # EMPTY FITNESS ARRAY OF SIZE N

    # best_agent, dataframe = linear_regression_data()
    best_agent, dataframe = random_forest_data()

    best_agent_descending = best_agent.copy()
    best_agent_descending.sort(reverse=True)
    features_percentage = 36
    # hfi = High Feature Index
    hfi = feature_selection(best_agent, best_agent_descending, features_percentage, dataframe,)
    print("-------------------------------------------------")
    dataframe = dataframe.drop([str(dataframe.columns[0])], axis=1)
    print(f"Total Features Considered = {len(hfi)}/25: \n", dataframe.columns[hfi])
