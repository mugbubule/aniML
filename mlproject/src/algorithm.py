import data
from matplotlib import pyplot
from numpy import set_printoptions
import numpy as np
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

class Algorithm:

    def __init__(self):
        self.X_train = list
        self.X_validation = list
        self.Y_train = list
        self.Y_validation = list
        self.num_folds = 10
        self.seed = 7
        self.scoring = 'neg_mean_squared_error'

    def preprocess(self, dataset, coorIndex):
        # Split-out validation dataset

        array = dataset.values
        X = array[:, coorIndex]
        Y = np.asarray(dataset['score'], dtype=int)
        set_printoptions(precision=3)
        X = self.normalize(X)
        return X, Y

    def evaluate(self, data):
        print("\n============ ALGORITHM EVALUATIONS ============")
        validation_size = 0.05
        dataset = data.dataset
        names = data.names
        features_index = []

        names.remove('id')
        for s in data.features_selected:
            if s is not "score":
                features_index.append(names.index(s))
        X, Y = self.preprocess(dataset, features_index)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(X, Y, test_size=validation_size,
                                                                                                  random_state=self.seed)
        # Spot-Check Algorithms
        self.spot_algo(dataset)
        self.standard_spot_algo(dataset)

    def tune(self, data):
        # We can only start to develop this part once we found the most appropriate algo for our problem
        print("\n============ ALGORITHM TUNING ============")

    def finalize(self, data):
        # prepare the model
        scaler = StandardScaler().fit(self.X_train)
        rescaledX = scaler.transform(self.X_train)
        model = GradientBoostingRegressor(random_state=self.seed, n_estimators=400)
        model.fit(rescaledX, self.Y_train)

        # transform the validation dataset
        rescaledValidationX = scaler.transform(self.X_validation)
        predictions = model.predict(rescaledValidationX)
        print(mean_squared_error(self.Y_validation, predictions))

    def spot_algo(self, dataset):
        results = []
        names = []
        # evaluate each model in turn
        print("====== ALGORITHMS ======")
        models = [('LR', LinearRegression()), ('LASSO', Lasso()), ('EN', ElasticNet()), ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()), ('SVR', SVR())]
        for name, model in models:
            kfold = KFold(n_splits=self.num_folds, random_state=self.seed)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=self.scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # Compare Algorithms
        fig = pyplot.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(results)
        ax.set_xticklabels(names)
        pyplot.show()
        print()

    def standard_spot_algo(self, dataset):
        results = []
        names = []
        # Standardize the dataset
        print("====== STANDARDIZING ALGORITHMS ======")
        pipelines = [('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])),
                     ('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])),
                     ('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])),
                     ('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])),
                     ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])),
                     ('ScaledSVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())]))]
        for name, model in pipelines:
            kfold = KFold(n_splits=self.num_folds, random_state=self.seed)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=self.scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # Compare Algorithms
        fig = pyplot.figure()
        fig.suptitle('Scaled Algorithm Comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(results)
        ax.set_xticklabels(names)
        pyplot.show()

    def rescale(self, X):
        scaler = MinMaxScaler(feature_range=(0, 100))
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
