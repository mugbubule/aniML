from numpy import set_printoptions
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from matplotlib import pyplot
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
        Y = np.asarray(dataset['score'], dtype=float)
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
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(X, Y,
                                                                                            test_size=validation_size,
                                                                                            random_state=self.seed)
        # regressor = LinearRegression()
        # regressor.fit(self.X_train, self.Y_train)
        # self.Y_validation = regressor.predict(self.X_train)
        # df = pd.DataFrame({'Actual':  self.Y_train, 'Predicted':  self.Y_validation})
        # print(df)
        # Spot-Check Algorithms
        self.spot_algo(dataset)
        self.standard_spot_algo(dataset)
        self.tune(data)
        self.ensemble_methods(dataset)
        self.tune_ensemble(dataset)

    def ensemble_methods(self, dataset):
        print("\n============ ENSEMBLE EVALUATIONS ============")
        ensembles = []
        ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB',
                                                                                   AdaBoostRegressor())])))
        ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM',
                                                                                    GradientBoostingRegressor())])))
        ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF',
                                                                                   RandomForestRegressor())])))
        ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET',
                                                                                   ExtraTreesRegressor())])))
        results = []
        names = []
        for name, model in ensembles:
            kfold = KFold(n_splits=self.num_folds, random_state=self.seed)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=self.scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        # Compare Algorithms
        fig = pyplot.figure()
        fig.suptitle(' Scaled Ensemble Algorithm Comparison ')
        ax = fig.add_subplot(111)
        pyplot.boxplot(results)
        ax.set_xticklabels(names)
        pyplot.show()

    def tune_ensemble(self, data):
        # We can only start to develop this part once we found the most appropriate algo for our problem
        print("\n============ GBM ALGORITHM TUNING ============")
        # Tune scaled GBM
        scaler = StandardScaler().fit(self.X_train)
        rescaledX = scaler.transform(self.X_train)
        param_grid = dict(n_estimators=np.array([50, 100, 150, 200, 250, 300, 350, 400]))
        model = GradientBoostingRegressor(random_state=self.seed)
        kfold = KFold(n_splits=self.num_folds, random_state=self.seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(rescaledX, self.Y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    def tune(self, data):
        # We can only start to develop this part once we found the most appropriate algo for our problem
        print("\n============ KNN TUNING ============")
        # KNN Algorithm tuning
        scaler = StandardScaler().fit(self.X_train)
        rescaledX = scaler.transform(self.X_train)
        k_values = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
        param_grid = dict(n_neighbors=k_values)
        model = KNeighborsRegressor()
        kfold = KFold(n_splits=self.num_folds, random_state=self.seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(rescaledX, self.Y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    def finalize(self, data):
        print("\n====== FINALIZATION ======")
        # prepare the model
        scaler = StandardScaler().fit(self.X_train)
        rescaledX = scaler.transform(self.X_train)
        model = GradientBoostingRegressor(random_state=self.seed, n_estimators=250)
        model.fit(rescaledX, self.Y_train)

        # transform the validation dataset
        rescaledValidationX = scaler.transform(self.X_validation)
        predictions = model.predict(rescaledValidationX)
        d = pd.DataFrame({'Actual value': self.Y_validation, 'Predicted value': predictions})
        print(d)
        print("\nFinal MSE: ", end="", flush=True), print(mean_squared_error(self.Y_validation, predictions))

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
