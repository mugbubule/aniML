import data
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
from pprint import pprint


class Algorithm:
    def __init__(self):
        self.X_train = list
        self.X_validation = list
        self.Y_train = list
        self.Y_validation = list
        self.num_folds = 10
        self.seed = 7
        self.scoring = ''

    def evaluate(self, dataset):
        # Split-out validation dataset
        array = dataset.values
        X = array[:, [3, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]]
        Y = array[:, 7]
        validation_size = 0.20
        seed = 7
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(X, Y,
                                                                                            test_size=validation_size,
                                                                                            random_state=seed)
        # Test options and evaluation metric
        self.num_folds = 10
        self.seed = 7
        self.scoring = 'neg_mean_squared_error'

        # Spot-Check Algorithms
        self.spot_algo(dataset)
        self.standard_spot_algo(dataset)

    def tune(self, dataset):
        # We can only start to develop this part once we found the most appropriate algo for our problem
        print("\n============ ALGORITHM TUNING ============")

    def finalize(self, dataset):
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
        models = [('LR', LinearRegression()), ('LASSO', Lasso()), ('EN', ElasticNet()), ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()), ('SVR', SVR())]

        # evaluate each model in turn
        results = []
        names = []
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

    def standard_spot_algo(self, dataset):
        # Standardize the dataset
        pipelines = [('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])),
                     ('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])),
                     ('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])),
                     ('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])),
                     ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])),
                     ('ScaledSVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())]))]
        results = []
        names = []
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
