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
    def work_on_data(self):
        names = ["producer_name", "anime_id"]
        producer_val = pd.read_csv("../../jikanAPI/jikan/producer.csv", index_col=0, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                            skipinitialspace=True, parse_dates=True, header=None, names=names)
        producer = pd.read_csv("../../jikanAPI/jikan/producer.csv", index_col=0, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                            skipinitialspace=True, parse_dates=True, header=None, names=names)
        pprint("####heeeeeyyyyy")
        working_table = self.dataset
        #working_table.
        pprint(producer_val.groupby("producer_name").size())
        #pprint(producer_val.producer_name.value_counts())
        pprint("####heeeeeyyyyy")
        #pd.merge(self.producer_val, dataset, left_on='id', right_on='anime_id')

    def __init__(self, filename):
        self.names = ["id", "title", "type", "source", "episodes", "aired", "duration", "rating", "score", "rank",
                      "scored_by", "popularity", "members", "favorites", "related", "genre", "watching", "completed",
                      "on_hold", "dropped", "plan_to_watch", "total"]

        self.dataset = pd.read_csv(filename, index_col=['id'], quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL,
                                   skipinitialspace=True, parse_dates=True, header=None,  names=self.names,
                                   dtype={'title': object, 'type': np.int32, 'source': np.int32,
                                          'episodes': np.int32, 'aired': np.int32, 'duration': np.int32,
                                          'rating': np.int32, 'score': np.float64, 'rank': np.int32,
                                          'scored_by': np.int32, 'popularity': np.int32, 'members': np.int32,
                                          'favorites': np.int32, 'related': np.int32, 'genre': object,
                                          'watching': np.int32, 'completed': np.int32, 'on_hold': np.int32,
                                          'dropped': np.int32, 'plan_to_watch': np.int32, 'total': np.int32})

        print("\n============ DATASET TYPES ============")
        print(self.dataset.dtypes)
        print()

    # Print correlations stat data
    def visualize(self):
        print("\n============ DATASET SIZE ============")
        print(self.dataset.shape[0], end="", flush=True), print(" rows")
        print(self.dataset.shape[1], end="", flush=True), print(" colunms")
        pd.set_option('precision', 1)

        print("\n============ DATASET GENERAL STATISTICS ============")
        print(self.dataset.describe())

        print("\n============ DATASET CORRELATIONS ============")
        pd.set_option('precision', 3)
        self.correlations = self.dataset.corr(method='pearson')
        print(self.correlations)
        print()

    def rescale(self, X):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(X)
    def standardize(self, X):
        scaler = StandardScaler().fit(X)
        return scaler.transform(X)
    def normalize(self, X):
        scaler = Normalizer().fit(X)
        return scaler.transform(X)
    def binarize(self, X):
        binarizer = Binarizer(threshold=0.0).fit(X)
        return binarizer.transform(X)
    # Plot the data
    def plot(self):
        self.dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
        pyplot.show()
        self.dataset.plot(kind='density', subplots=True, layout=(5, 5), sharex=False, legend=True, fontsize=1)
        pyplot.show()
        self.dataset.plot(kind='box', subplots=True, layout=(5, 5), sharex=False, sharey=False, fontsize=8)
        pyplot.show()

    def plot_correlations(self):
        label_names = ["type", "source", "episodes", "aired", "duration", "rating", "score", "rank",
                     "scored_by", "popularity", "members", "favorites", "related", "watching", "completed",
                     "on_hold", "dropped", "plan_to_watch", "total"]
        # plot correlation matrix
        fig = pyplot.figure(figsize=(18, 14))
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, 19, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        pyplot.show()

    def preprocess(self):
        print("\n============ DATASET PREPROCESSING ============")
