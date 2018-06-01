import numpy
from numpy import arange
import csv
import pandas as pd


filename = '../anime.csv'
names = ["id", "title", "type",
         "source", "episodes", "aired",
         "duration", "rating", "score",
         "rank", "scored_by", "popularity",
         "members", "favorites", "related",
         "genre", "watching", "completed",
         "on_hold", "dropped", "plan_to_watch", "total"]

dataset = pd.read_csv(filename, index_col=False, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True, names=names)
dataset["score"] = pd.to_numeric(dataset['score'], errors='coerce').fillna(0)
dataset["completed"] = pd.to_numeric(dataset['completed'], errors='coerce').fillna(0)
dataset["watching"] = pd.to_numeric(dataset['watching'], errors='coerce').fillna(0)
dataset["rank"] = pd.to_numeric(dataset['rank'], errors='coerce').fillna(0)
dataset["episodes"] = pd.to_numeric(dataset['episodes'], errors='coerce').fillna(0)

types = dataset.dtypes
peek = dataset.head(1)
correlations = dataset.corr(method='pearson')

print("The dataset' size :"), print(dataset.shape)
print(types)
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


