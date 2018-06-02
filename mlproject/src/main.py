from data import Data


def main():
    filename = '../../jikanAPI/jikan/anime.csv'
    data = Data(filename)
    #data.visualize()
    # data.plot()
    #data.plot_correlations()
    data.compute()

if __name__ == "__main__":
    main()

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
