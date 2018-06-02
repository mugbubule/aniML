import csv
import pandas as pd
from matplotlib import pyplot
import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from pprint import pprint


class Data:
    # Load the csv file into pandas dataset
    def __init__(self, filename):
        self.names = ["id", "title", "type", "source", "episodes", "aired", "duration", "rating", "score", "rank",
                 "scored_by", "popularity","members", "favorites", "related", "genre", "watching", "completed",
                 "on_hold", "dropped", "plan_to_watch", "total"]
        self.dataset = pd.read_csv(filename, index_col=0, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                               skipinitialspace=True, parse_dates=True, header=None, names=self.names)
        self.dataset["score"] = pd.to_numeric(self.dataset["score"], errors='coerce').fillna(0)
        self.dataset["completed"] = pd.to_numeric(self.dataset["completed"], errors='coerce').fillna(0)
        self.dataset["watching"] = pd.to_numeric(self.dataset["watching"], errors='coerce').fillna(0)
        self.dataset["rank"] = pd.to_numeric(self.dataset["rank"], errors='coerce').fillna(0)
        self.dataset["episodes"] = pd.to_numeric(self.dataset["episodes"], errors='coerce').fillna(0)
        # print(self.dataset.head(1))
        # pprint(self.dataset)

    # Print correlations stat data
    def visualize(self):
        print("\n============ DATASET SIZE ============")
        print(self.dataset.shape[0], end="", flush=True), print(" rows")
        print(self.dataset.shape[1], end="", flush=True), print(" colunms")
        pd.set_option('precision', 1)

        print("\n============ DATASET TYPES ============")
        print(self.dataset.dtypes)
        print()

        print("\n============ DATASET GENERAL STATISTICS ============")
        print(self.dataset.describe())

        print("\n============ DATASET CORRELATIONS ============")
        pd.set_option('precision', 3)
        correlations = self.dataset.corr(method='pearson')
        print(correlations)
        print()

    # Plot the data
    def plot(self):
        self.dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
        pyplot.show()
        self.dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
        pyplot.show()
        self.dataset.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, fontsize=8)
        pyplot.show()

    def plot_correlations(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.dataset.corr(), vmin=-1, vmax=1, interpolation='none')
        fig.colorbar(cax)
        ticks = np.arange(0, 14, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.names)
        ax.set_yticklabels(self.names)
        pyplot.show()

    def compute(self):
        # Split-out validation dataset
        array = self.dataset.values
        X = array[:, 11:13]
        Y = array[:, 8]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

        # Test options and evaluation metric
        num_folds = 10
        seed = 7
        scoring = 'neg_mean_squared_error'

        # Spot-Check Algorithms
        models = []
        models.append(('LR', LinearRegression()))
        models.append(('LASSO', Lasso()))
        models.append(('EN', ElasticNet()))
        models.append(('KNN', KNeighborsRegressor()))
        models.append(('CART', DecisionTreeRegressor()))
        models.append(('SVR', SVR()))

        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = KFold(n_splits=num_folds, random_state=seed)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)