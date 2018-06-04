import data
from matplotlib import pyplot
from numpy import set_printoptions
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

    def evaluate(self, dataset):
        print("\n============ ALGORITHM EVALUATIONS ============")
        names = ["title", "type", "source", "episodes", "aired", "duration", "rating", "score", "rank",
                 "scored_by", "popularity", "members", "favorites", "related", "genre", "watching", "completed",
                 "on_hold", "dropped", "plan_to_watch", "total"]

        # Split-out validation dataset
        array = dataset.values
        X = array[:, [names.index("type"), names.index("source"),  names.index("episodes"),  names.index("aired"),
                      names.index("duration"), names.index("rating"), names.index("scored_by"), names.index("popularity"),
                      names.index("members"), names.index("favorites"), names.index("related")
                      ]]
        Y = array[:, names.index("score")]
        validation_size = 0.20
        print("it broke")
        set_printoptions(precision=3)
        print(X)
        X = self.rescale(X)
        print("fixed it")
        print(X)
        # Test options and evaluation metric
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(X, Y,
                                                                                            test_size=validation_size,
                                                                                            random_state=self.seed)
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
