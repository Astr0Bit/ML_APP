import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

class Regressor():
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame):
        
        self.X = X
        self.y = y
        self.estimator_params = {
                            "LiR" : {},
                            "RCV" : {},
                            "LCV" : {},
                            "ECV" : {},
                            "SVR" : {}}
        
        self.estimator_scores = {
                            "LiR" : {},
                            "RCV" : {},
                            "LCV" : {},
                            "ECV" : {},
                            "SVR" : {}}
        
        self.best_estimator_ = {
            "estimator" : {},
            "scores" : {},
            "params" : {},
        }
        
        self._final_model = None

        # choose test_size based on the size of X
        if 10000 <= len(self.X) < 20000:
            self.test_size = 0.2

        elif len(self.X) >= 20000:
            self.test_size = 0.1

        elif len(self.X) <= 5000:
            self.test_size = 0.3

        else:
            self.test_size = 0.4

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=101)

    def create_model(self, estimator):
        # here, the user can enter which estimator they desire to use,
        # the user will not have this functionality when using the app

        # to create a model, a pipeline is used to ensure every step is
        # executed as desired and also to ensure no miss-typo. 
        if issubclass(estimator, LinearRegression):
            estimator = LinearRegression()
            column = "LiR"
            param_grid = {
                "poly__include_bias" : [False],
                "poly__degree" : list(range(1, 3)), # degree 1 to 2
                "estimator__fit_intercept" : [True]
            }

        elif issubclass(estimator, RidgeCV):
            estimator = RidgeCV()
            column = "RCV"
            param_grid = {
                "poly__include_bias" : [False],
                "poly__degree" : list(range(1, 3)),
                "estimator__alphas" : list(np.arange(0.1, 10, 0.1))
                }

        elif issubclass(estimator, LassoCV):
            estimator = LassoCV()
            column = "LCV"
            param_grid = {
                "poly__include_bias" : [False],
                "poly__degree" : list(range(1, 3)),
                "estimator__eps" : [0.001, 0.01, 0.1, 1],
                "estimator__n_alphas" : [10, 100],
                "estimator__max_iter" : [10_000, 100_000]
            }

        elif issubclass(estimator, ElasticNetCV):
            estimator = ElasticNetCV()
            column = "ECV"
            param_grid = {
                "poly__include_bias" : [False],
                "poly__degree" : list(range(1, 3)),
                "estimator__l1_ratio" : [.1, .5, .7, .9, .95, .99, 1],
                "estimator__eps" : [0.001, 0.01, 0.1, 1],
                "estimator__n_alphas" : [10, 100],
                "estimator__max_iter" : [10_000, 100_000]
            }

        elif issubclass(estimator, SVR):
            estimator = SVR()
            column = "SVR"
            param_grid = {
                "poly__include_bias" : [False],
                "poly__degree" : list(range(1, 3)),
                "estimator__kernel" : ["linear", "poly", "rbf", "sigmoid"],
                "estimator__degree" : np.arange(1, 11),
                "estimator__C" : np.logspace(0, 1, 10),
                "estimator__gamma" : ["scale", "auto"],
                "estimator__epsilon" : [0, 0.001, 0.01, 0.1, 0.5, 1, 2]
            }

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures()),
            ("estimator", estimator)
        ])

        # gridsearchcv is used to find the best parameters for the entered estimator
        gridsearch = GridSearchCV(n_jobs=-1, estimator=self.pipeline,
                                  param_grid=param_grid, cv=10, 
                                  scoring="neg_mean_squared_error")
        
        gridsearch.fit(self.X_train, self.y_train);

        best_model = gridsearch.best_estimator_
        best_model_pred = best_model.predict(self.X_test);

        # when going through the code, these do not appear to ever be used,
        # maybe they are not needed and can be removed?
        X_train = best_model.named_steps["scaler"].transform(self.X_train)
        X_train = best_model.named_steps["poly"].transform(X_train)

        X_test = best_model.named_steps["scaler"].transform(self.X_test)
        X_test = best_model.named_steps["poly"].transform(X_test)

        MAE = mean_absolute_error(self.y_test, best_model_pred)
        MSE = mean_squared_error(self.y_test, best_model_pred)
        RMSE = np.sqrt(MSE)
        R2_score = r2_score(self.y_test, best_model_pred)
        self.estimator_scores[column]["MAE"] = MAE
        self.estimator_scores[column]["RMSE"] = RMSE
        self.estimator_scores[column]["R2_Score"] = R2_score
        self.estimator_params[column]["params"] = {param: [value] for param, value in gridsearch.best_params_.items()}

    def create_models(self):
        # method to create all models using a for-loop
        estimators = [LinearRegression, RidgeCV, LassoCV, ElasticNetCV, SVR]

        # for loop to pass and create an estimator to the create_model-method
        for estimator in tqdm(estimators): # tqdm is just for output --> gives a nice progress bar
            self.create_model(estimator=estimator)
    
    def get_best_estimator(self):
        # Find the model with the highest R2 score
        best_score = -float('inf')
        for model_name, scores in self.estimator_scores.items():
            if len(scores) == 0:
                continue

            r2_score = scores.get("R2_score", 0)
            if r2_score > best_score:
                best_score = r2_score
                self.best_estimator_["estimator"] = model_name
                self.best_estimator_["scores"] = scores
                self.best_estimator_["params"] = self.estimator_params[model_name]["params"]

        return f"Best estimator for this data is : {self.best_estimator_['estimator']}"

    def get_scores(self):
        # method to get scores for each model that was created
        for model_name, scores in self.estimator_scores.items():
            if len(scores) == 0:
                continue
            
            print(model_name)
            for metric, score in self.estimator_scores[model_name].items():
                print(f"{metric} : {score:.3f}")
            print()

    def create_final_model(self):
        # method to create the final model using the best found estimator
        # along with the best found parameters
        best_estimator = self.best_estimator_["estimator"]
        best_params = self.best_estimator_["params"]
        
        # simple if-statement to check what the best_estimator was
        if best_estimator == "LiR":
            estimator = LinearRegression()

        elif best_estimator == "RCV":
            estimator = RidgeCV()

        elif best_estimator == "LCV":
            estimator = LassoCV()

        elif best_estimator == "ECV":
            estimator = ElasticNetCV()
            
        elif best_estimator == "SVR":
            estimator = SVR()
        
        # we use a pipeline to ensure every step is executed correctly
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures()),
            ("estimator", estimator)
        ])
        
        gridsearch = GridSearchCV(n_jobs=-1, estimator=self.pipeline,
                                  param_grid=best_params, cv=10,
                                  scoring="neg_mean_squared_error")

        gridsearch.fit(self.X, self.y)

        self._final_model = gridsearch.best_estimator_
      
        return self._final_model