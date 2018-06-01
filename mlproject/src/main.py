import numpy
from numpy import arange
import csv
from pandas import read_csv, set_option


filename = '../anime.csv'
names = ["id", "title", "type",
         "source", "episodes", "aired",
         "duration", "rating", "score",
         "rank", "scored_by", "popularity",
         "members", "favorites", "related",
         "genre", "watching", "completed",
         "on_hold", "dropped", "plan_to_watch", "total"]
dataset = read_csv(filename, index_col=False, quotechar = '"', delimiter = ',', quoting = csv.QUOTE_ALL, skipinitialspace = True, names=names)
print("The dataset' size :")
print(dataset.shape)
types = dataset.dtypes
description = dataset.describe()
print(description)
correlations = dataset.corr(method='pearson')
print(correlations)
# 2. Summarize Data
# a) Descriptive statistics
# b) Data visualizations
#
#
# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data transforms
#
#
# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms
#
#
# 5. Improve Accuracy
# a) Algorithm tuning
# b) Ensembles

# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use


