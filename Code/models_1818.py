import random
from math import sqrt
import numpy as np
import pandas as pd
from numpy import asarray, absolute, mean
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# fit_transform for training
# and transform for testing


class regression_models:
    def __init__(self, file):
        self.file = file
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

        self.lr_model = None
        self.rf_model = None

        self.csv_columns = None

        self.lr_training_Data = dict()
        self.lr_training_Data = {'R2 Score': [], 'MSE': [], 'MAE': []}

        self.lr_testing_Data = dict()
        self.lr_testing_Data = {'R2 Score': [], 'MSE': [], 'MAE': []}

        self.rf_training_Data = dict()
        self.rf_training_Data = {'R2 Score': [], 'MSE': [], 'MAE': []}

        self.rf_testing_Data = dict()
        self.rf_testing_Data = {'R2 Score': [], 'MSE': [], 'MAE': []}

        self.losses = ['R2 Score', 'MSE', 'MAE']

        self.lr_train_loss = []
        self.lr_test_loss = []

        self.rf_train_loss = []
        self.rf_test_loss = []

    def process_data(self):
        pd_file = pd.read_csv(self.file)
        dataframe = pd.DataFrame(pd_file)

        dataframe = dataframe.fillna(0)

        X = dataframe.iloc[:, 1:].values
        Y = dataframe.iloc[:, 0].values.reshape(-1, 1)

        return X, Y, dataframe

    def splitting_data(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25,
                                                                                random_state=True, shuffle=True)

        # print(self.y_train)
        scaler = MinMaxScaler()
        scaler.fit(self.X_train)
        scaled_x_train = scaler.fit_transform(self.X_train)
        self.X_train = scaled_x_train

        scaler.fit(self.y_train)
        scaled_y_train = scaler.fit_transform(self.y_train)
        scaled_y_train = scaled_y_train.reshape(1, -1)
        self.y_train = scaled_y_train[0]

        # print(self.X_train, self.y_train)

        scaler.fit(self.X_test)
        scaled_x_test = scaler.transform(self.X_test)
        self.X_test = scaled_x_test
        # print(scaled_x_test)

        # print(self.y_test)
        scaler.fit(self.y_test)
        scaled_y_test = scaler.transform(self.y_test)
        scaled_y_test = scaled_y_test.reshape(1, -1)
        self.y_test = scaled_y_test[0]


    # LINEAR REGRESSION

    def linear_regression(self):
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)

        # _________________________________________________
        lr_train_pred = self.lr_model.predict(self.X_train)

        lr_train_r2score = r2_score(self.y_train, lr_train_pred)
        lr_train_mse = mean_squared_error(self.y_train, lr_train_pred)
        lr_train_mae = mean_absolute_error(self.y_train, lr_train_pred)

        # _________________________________________________
        lr_test_pred = self.lr_model.predict(self.X_test)

        lr_test_r2score = r2_score(self.y_test, lr_test_pred)  # actual and predicted
        lr_test_mse = mean_squared_error(self.y_test, lr_test_pred)
        lr_test_mae = mean_absolute_error(self.y_test, lr_test_pred)

        # losses = 'R2 Score', 'MSE', 'MAE'
        self.lr_train_loss = [lr_train_r2score, lr_train_mse, lr_train_mae]
        self.lr_test_loss = [lr_test_r2score, lr_test_mse, lr_test_mae]

    def linear_regression_testing(self, data):

        pred = self.lr_model.predict(data)

        return pred

    def lr_train_test_data(self):
        for index in range(len(self.lr_train_loss)):
            self.lr_training_Data[self.losses[index]].append(str(self.lr_train_loss[index]))

            self.lr_testing_Data[self.losses[index]].append(str(self.lr_test_loss[index]))

    # RANDOM FOREST

    def random_forest(self):
        self.rf_model = RandomForestRegressor(n_estimators=15, min_samples_split=45, min_samples_leaf=130,
                                              max_features='log2', max_depth=1, bootstrap=True)
        self.rf_model.fit(self.X_train, self.y_train)

        # _________________________________________________
        rf_train_pred = self.rf_model.predict(self.X_train)

        rf_train_r2score = r2_score(self.y_train, rf_train_pred)
        rf_train_mse = mean_squared_error(self.y_train, rf_train_pred)
        rf_train_mae = mean_absolute_error(self.y_train, rf_train_pred)

        # _________________________________________________
        rf_test_pred = self.rf_model.predict(self.X_test)

        rf_test_r2score = r2_score(self.y_test, rf_test_pred)  # actual and predicted
        rf_test_mse = mean_squared_error(self.y_test, rf_test_pred)
        rf_test_mae = mean_absolute_error(self.y_test, rf_test_pred)

        self.rf_train_loss = [rf_train_r2score, rf_train_mse, rf_train_mae]
        self.rf_test_loss = [rf_test_r2score, rf_test_mse, rf_test_mae]

    def random_forest_testing(self, data):
        pred = self.rf_model.predict(data)
        return pred

    def rf_train_test_data(self):
        for index in range(len(self.rf_train_loss)):
            self.rf_training_Data[self.losses[index]].append(str(self.rf_train_loss[index]))

            self.rf_testing_Data[self.losses[index]].append(str(self.rf_test_loss[index]))


if __name__ == "__main__":
    pass
