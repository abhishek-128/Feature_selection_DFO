import random
import numpy as np
from statistics import mean
from statistics import median
from statistics import stdev
import matplotlib.pyplot as plt
import math as m
import time

import pandas as pd
import tikzplotlib
from models_1818 import regression_models


# SWITCHING FROM LR TO RF OR VICE VERSA (6 Changes)-
# Change fitness in loop function
# Change fitness in evaluate function
# Change 2 plots titles and color of bar graph
# Invoke respective function in main for dataframe formation


class dfo:
    def __init__(self, N, D, delta, maxIterations, lowerB, upperB, X, fitness):
        seed = 12
        random.seed(seed)
        print("RANDOM SEED UTILISED: ", seed)

        self.N, self.D, self.delta, self.maxIterations, \
        self.lowerB, self.upperB, self.X, self.fitness = N, D, delta, \
                                                         maxIterations, lowerB, upperB, X, fitness

        self.best_cost = np.empty(self.maxIterations)

        self.call = regression_models('AI_app_dataset.csv')
        x, y, self.dataframe = self.call.process_data()
        self.call.splitting_data(x, y)

    def initialize(self):
        for i in range(self.N):
            for d in range(self.D):
                self.X[i, d] = np.random.uniform(self.lowerB[d], self.upperB[d])

    def loop(self):
        for itr in range(self.maxIterations):
            for i in range(self.N):  # EVALUATION
                # self.fitness[i] = self.linear_reg_fitness(self.X[i])
                self.fitness[i] = self.random_forest_fitness(self.X[i])
            s = np.argmax(self.fitness)  # FIND BEST FLY INDEX

            if itr % 1 == 0:  # PRINT BEST FLY EVERY 100 ITERATIONS
                print("Iteration:", itr, "\tBest fly index:", s, "\tFitness value:", self.fitness[s])

            # TAKE EACH FLY INDIVIDUALLY
            for i in range(self.N):
                if i == s:
                    continue  # ELITIST STRATEGY

                # FIND BEST NEIGHBOUR
                left = (i - 1) % self.N
                right = (i + 1) % self.N
                bNeighbour = right if self.fitness[right] > self.fitness[left] else left

                for d in range(self.D):  # UPDATE EACH DIMENSION SEPARATELY
                    if np.random.rand() < self.delta:
                        self.X[i, d] = np.random.uniform(self.lowerB[d], self.upperB[d])
                        continue

                    u = np.random.rand()
                    self.X[i, d] = self.X[bNeighbour, d] + u * (self.X[s, d] - self.X[i, d])

                    # OUT OF BOUND CONTROL
                    if self.X[i, d] < self.lowerB[d] or self.X[i, d] > self.upperB[d]:
                        self.X[i, d] = np.random.uniform(self.lowerB[d], self.upperB[d])

            self.best_cost[itr] = self.fitness[s]
            # print(self.X[s])

    def linear_reg_fitness(self, x):
        self.call.linear_regression()
        pred = self.call.linear_regression_testing([x])
        return pred

    def random_forest_fitness(self, x):
        self.call.random_forest()
        pred = self.call.random_forest_testing([x])
        return pred

    def eval(self):
        for i in range(self.N):
            # self.fitness[i] = self.linear_reg_fitness(self.X[i])  # EVALUATION
            self.fitness[i] = self.random_forest_fitness(self.X[i])
        s = np.argmax(self.fitness)  # FIND BEST FLY

        print("\nFinal best fitness:\t", self.fitness[s])
        print("\nBest fly index:\t", s)
        print("\nBest fly position:\n", self.X[s])
        return self.X[s]

    def plot_train_test_losses_lr(self):
        self.call.lr_train_test_data()

        lr_train_data = pd.DataFrame(self.call.lr_training_Data)
        lr_test_data = pd.DataFrame(self.call.lr_testing_Data)
        print("-------------------------------------------------")
        print("Training Analysis Before Feature Selection: \n", self.call.lr_training_Data,
              "\nTesting Analysis Before Feature Selection: \n", self.call.lr_testing_Data)

        # r2 = [float(self.call.lr_training_Data['R2 Score'][0]), float(self.call.lr_testing_Data['R2 Score'][0])]
        # mse = [float(self.call.lr_training_Data['MSE'][0]), float(self.call.lr_testing_Data['MSE'][0])]
        # mae = [float(self.call.lr_training_Data['MAE'][0]), float(self.call.lr_testing_Data['MAE'][0])]
        #
        # print(r2)
        #
        # # set width of bar
        # barWidth = 0.25
        # fig = plt.subplots(figsize=(12, 8))
        #
        # # Set position of bar on X axis
        # br1 = np.arange(len(r2))
        # br2 = [x + barWidth for x in br1]
        # br3 = [x + barWidth for x in br2]
        #
        # # Make the plot
        # plt.bar(br1, r2, color='r', width=barWidth,
        #         edgecolor='grey', label='R2 Score')
        # plt.bar(br2, mse, color='g', width=barWidth,
        #         edgecolor='grey', label='MSE')
        # plt.bar(br3, mae, color='b', width=barWidth,
        #         edgecolor='grey', label='MAE')
        #
        # # Adding Xticks
        # #plt.xlabel('Branch', fontweight='bold', fontsize=15)
        # #plt.ylabel('Students passed', fontweight='bold', fontsize=15)
        # plt.xticks([r + barWidth for r in range(len(r2))],
        #            ['Training Set', 'Testing Set'])
        #
        # plt.legend()
        # plt.show()

        # plt.plot(train_data['R2 Score'], label='R2 Score', color='green')
        # plt.plot(train_data['MSE'], label='MSE', color='steelblue')
        # plt.plot(train_data['MAE'], label='MAE', color='purple')

        # plt.bar(train_data)
        # plt.legend(title='Loss Functions')
        #
        # plt.ylabel('Scores/Errors', fontsize=14)
        # plt.xlabel('Iterations', fontsize=14)
        # plt.title('Training Data Loss Function Scores/Errors', fontsize=16)
        # plt.show()
        #
        # # -----------------------------------------------------------------------
        #
        # # plt.plot(test_data['R2 Score'], label='R2 Score', color='green')
        # # plt.plot(test_data['MSE'], label='MSE', color='steelblue')
        # # plt.plot(test_data['MAE'], label='MAE', color='purple')
        # plt.bar(test_data)
        # plt.legend(title='Loss Functions')
        #
        # plt.ylabel('Scores/Errors', fontsize=14)
        # plt.xlabel('Iterations', fontsize=14)
        # plt.title('Testing Data Loss Function Scores/Errors', fontsize=16)
        # plt.show()

    def plot_train_test_losses_rf(self):
        self.call.rf_train_test_data()

        rf_train_data = pd.DataFrame(self.call.rf_training_Data)
        rf_test_data = pd.DataFrame(self.call.rf_testing_Data)
        print("-------------------------------------------------")
        print("Training Analysis Before Feature Selection: \n", self.call.rf_training_Data,
              "\nTesting Analysis Before Feature Selection: \n", self.call.rf_testing_Data)

    def plot(self, agent_array):
        # plt.plot(self.best_cost)

        # DFO Fitness Graph LR
        plt.semilogy(self.best_cost)
        plt.xlim(0, self.maxIterations - 1)
        plt.xlabel('Iterations')
        plt.ylabel('Best Cost')
        # plt.title('Dispersive Flies Optimisation - Linear Regression')
        plt.title('Dispersive Flies Optimisation - Random Forest')
        plt.grid(True)
        # tikzplotlib.save("Dispersive Flies Optimisation_Linear Regression.tex")
        tikzplotlib.save("Dispersive Flies Optimisation_Random Forest.tex")
        plt.show()

        # -----------------------------------------------------------------

        dimensions = list(x for x in range(0, len(agent_array)))
        values = list(agent_array)

        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(dimensions, values, color='#E11642',
                width=0.8)
        # RF - E11642
        # LR - 409BBA
        plt.xticks(dimensions)
        plt.xlabel("Dataset Dimension")
        plt.ylabel("Feature Value")
        # plt.title("Feature weights using DFO (Linear Regression)")
        plt.title("Feature weights using DFO (Random Forest)")

        # tikzplotlib.save("Feature weights using DFO (Linear Regression).tex")
        tikzplotlib.save("Feature weights using DFO (Random Forest).tex")
        plt.show()


if __name__ == "__main__":
    pass

