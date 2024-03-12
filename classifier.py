from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class Classifier():
    def __init__(self, X, y):
        # Initialization method for the Classifier class
        self.X = X
        self.y = y

        # Parameters and scores for each estimator
        self.estimator_params = {
            "LoR": {},
            "KNN": {},
            "SVC": {}
        }
        self.estimator_scores = {
            "LoR": {},
            "KNN": {},
            "SVC": {}
        }
        self.best_estimator_ = {
            "estimator": {},
            "scores": {},
            "params": {},
        }
        self._final_model = None
        self.confusion_matrix_ = None

        # Splitting data into train and test sets based on the size of X
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
        # Method to create a model for the given estimator
        if issubclass(estimator, LogisticRegression):
            # Logistic Regression estimator
            estimator = LogisticRegression()
            column = "LoR"  # Column name for estimator
            param_grid = {
                "estimator__C": [0.1, 1, 10],  # Regularization parameter
                "estimator__solver": ['liblinear', 'lbfgs', 'sag', 'saga'],  # Solver for optimization
                "estimator__max_iter": [100, 200, 300]  # Maximum number of iterations
            }

        elif issubclass(estimator, KNeighborsClassifier):
            # K-Nearest Neighbors estimator
            estimator = KNeighborsClassifier()
            column = "KNN"  # Column name for estimator
            param_grid = {
                "estimator__n_neighbors": [3, 5, 7],  # Number of neighbors
                "estimator__weights": ['uniform', 'distance'],  # Weight function
                "estimator__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']  # Algorithm for computing nearest neighbors
            }

        elif issubclass(estimator, SVC):
            # Support Vector Classifier estimator
            estimator = SVC()
            column = "SVC"  # Column name for estimator
            param_grid = {
                "estimator__C": [0.1, 1, 10],  # Regularization parameter
                "estimator__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
                "estimator__gamma": ['scale', 'auto']  # Kernel coefficient
            }

        # Pipeline for preprocessing and building the model
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),  # Standardization
            ("estimator", estimator)  # Estimator
        ])

        # Grid search to find the best parameters for the estimator
        gridsearch = GridSearchCV(n_jobs=-1, estimator=self.pipeline,
                                  param_grid=param_grid, cv=10,
                                  scoring="accuracy")

        gridsearch.fit(self.X_train, self.y_train)

        best_model = gridsearch.best_estimator_
        best_model_pred = best_model.predict(self.X_test)

        # Calculating evaluation metrics for the model
        accuracy = accuracy_score(self.y_test, best_model_pred)
        precision = precision_score(self.y_test, best_model_pred, average='weighted')
        recall = recall_score(self.y_test, best_model_pred, average='weighted')
        f1 = f1_score(self.y_test, best_model_pred, average='weighted')

        # Storing scores and parameters for the estimator
        self.estimator_scores[column]["accuracy"] = accuracy
        self.estimator_scores[column]["precision"] = precision
        self.estimator_scores[column]["recall"] = recall
        self.estimator_scores[column]["f1"] = f1
        self.estimator_params[column]["params"] = {param: [value] for param, value in gridsearch.best_params_.items()}

    def create_models(self):
        # Method to create models for all estimators
        estimators = [LogisticRegression, KNeighborsClassifier, SVC]

        for estimator in tqdm(estimators):
            self.create_model(estimator=estimator)

    def get_best_estimator(self):
        # Method to find the best estimator based on accuracy score
        best_accuracy = -float('inf')  # Initialize with negative infinity
        for model_name, scores in self.estimator_scores.items():
            if len(scores) == 0:
                continue

            accuracy = scores.get("accuracy", 0)  # Get the accuracy score, defaulting to 0 if not present
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_estimator_["estimator"] = model_name
                self.best_estimator_["scores"] = scores
                self.best_estimator_["params"] = self.estimator_params[model_name]["params"]

        return f"Best estimator for this data is : {self.best_estimator_['estimator']}"

    def get_scores(self):
        # Method to display scores for all models
        for model_name, scores in self.estimator_scores.items():
            if len(scores) == 0:
                continue

            print(model_name)
            for metric, score in scores.items():
                print(f"{metric.capitalize()}: {score:.3f}")
            print()

    def create_final_model(self):
        # Method to create the final model using the best estimator and parameters
        best_estimator = self.best_estimator_["estimator"]
        best_params = self.best_estimator_["params"]
        
        if best_estimator == "LoR":
            estimator = LogisticRegression()

        elif best_estimator == "KNN":
            estimator = KNeighborsClassifier()
            
        elif best_estimator == "SVC":
            estimator = SVC()
        
        self._final_model = Pipeline([
            ("scaler", StandardScaler()),  # Standardization
            ("estimator", estimator)  # Estimator
        ])

        # Grid search to find the best parameters for the final model
        gridsearch = GridSearchCV(n_jobs=-1, estimator=self._final_model,
                                  param_grid=best_params, cv=10,
                                  scoring="accuracy")

        gridsearch.fit(self.X, self.y)

        self._final_model = gridsearch.best_estimator_

        return self._final_model
