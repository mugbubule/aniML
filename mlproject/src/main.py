from data import Data
from algorithm import Algorithm

def main():
    filename = 'helper/clean_anime.csv'
    data = Data(filename)
#   2. Summarize Data
#   a) Descriptive statistics
#   b) Data visualizations
    data.visualize()

    #data.plot()
    data.plot_correlations()


    algorithm = Algorithm()
    #   3. Prepare Data
    # a) Data Cleaning
    # b) Feature Selection
    # c) Data transforms
    #   4. Evaluate Algorithms
    # a) Split-out validation dataset
    # b) Test options and evaluation metric
    # c) Spot Check Algorithms
    # d) Compare Algorithms
    algorithm.evaluate(data)

#   5. Improve Accuracy
#   a) Algorithm tuning
#   b) Ensembles
    #algorithm.tune(data.dataset)

#   6. Finalize Model
#   a) Predictions on validation dataset
#   b) Create standalone model on entire training dataset
#   c) Save model for later use
    #algorithm.finalize(data.dataset)

if __name__ == "__main__":
    main()
