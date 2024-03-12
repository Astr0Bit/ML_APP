import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer

class AutoEDA():
    # the __init__ for AutoEDA --> requires a df and a target column (must be in df)
    def __init__(self, df:pd.DataFrame | np.ndarray, target:str):
        if not isinstance(df, pd.DataFrame | np.ndarray):
            raise ValueError("'df' must be pd.Dataframe or np.ndarray")

        if target not in df.columns and not isinstance(target, str):
            raise ValueError("'target' must be in df, and 'target' must be str")
        
        self.df = df
        self.target_ = target
        self.target = self.df[target]
        self.features = self.df.drop(target, axis=1)

    def get_na(self):
        # method to retrieve missing values in df
        self.missing_values_ = None
        missing_values = self.df.isna().sum().sum()
        if missing_values > 0:
            # print(f"Missing values found : {missing_values}")
            self.missing_values_ = True
            self.missing_values = missing_values
        
        else:
            self.missing_values_ = False

    def fill_na(self, how:str):
        # various methods of handling missing data
        if self.missing_values_:
            # using interpolate
            if how == "interpolate":
                self.df = self.df.interpolate()
                self.target = self.df[self.target_]
                self.features = self.df.drop(self.target_, axis=1)

            # using IterativeImputer --> ML algorithm to guess the missing data
            elif how == "iterative_imputer":
                imputer = IterativeImputer(random_state=101)
                imputer.fit(self.df)
                self.df = pd.DataFrame(imputer.transform(self.df), columns=self.df.columns)
                self.target = self.df[self.target_]
                self.features = self.df.drop(self.target_, axis=1)
            
            # using SimpleImputer --> fill with mean values
            elif how == "simple_imputer":
                imputer = SimpleImputer(strategy="median")
                imputer.fit(self.df)
                self.df = pd.DataFrame(imputer.transform(self.df), columns=self.df.columns)
                self.target = self.df[self.target_]
                self.features = self.df.drop(self.target_, axis=1)

            # using KNNImputer --> works like traditional KNN,
            # but instead of "guessing" for our label,
            # it tries to guess the missing values
            elif how == "knn_imputer":
                imputer = KNNImputer(n_neighbors=5)
                imputer.fit(self.df)
                self.df = pd.DataFrame(imputer.transform(self.df), columns=self.df.columns)
                self.target = self.df[self.target_]
                self.features = self.df.drop(self.target_, axis=1)

            # If the user so desires, they can simply remove the
            # the missing / NaN - data.
            elif how == "drop_na":
                self.df = self.df.dropna()
                self.target = self.df[self.target_]
                self.features = self.df.drop(self.target_, axis=1)

        else:
            print("Missing values not filled.")
    
    def regressor_classifier(self):
        # method to determine whether regressor or classifier should be used
        # on the given data.
        self.regressor_classifier_ = None
        self.dummies_ = False
        self.dummies_list_ = []

        # Check if the target variable is continuous
        if np.issubdtype(self.target.dtype, np.number):
            self.regressor_classifier_ = True  # If true --> use regressor
        else:
            self.regressor_classifier_ = False  # If false --> use classifier

        # Check other columns if they are continuous or categorical
        for col in self.features.columns:
            if not np.issubdtype(self.features[col].dtype, np.number):
                self.dummies_ = True
                self.dummies_list_.append(col)

    def binary_features(self, create:bool=False):
        # method to convert the features to binary using get_dummies()

        if self.dummies_:
            self.df = pd.get_dummies(data=self.df, columns=self.dummies_list_, drop_first=True, dtype="uint8")
            self.target = self.df[self.target_]
            self.features = self.df.drop(self.target_, axis=1)
            
        elif not self.regressor_classifier_ and create:
            self.features = pd.get_dummies(self.features, drop_first=True, dtype="uint8")
        
        else:
            print(f"'{self.target.name}' is not categorical --> regressor will be used.")

        return self.features

    def get_outliers_iqr(self, thresh:float=1.5): # as this is not reliable at the moment, it will most likely not be used in the app
        # inspiration found on https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/
        # and https://saturncloud.io/blog/how-to-detect-and-exclude-outliers-in-a-pandas-dataframe/

        # method to retrieve outliers found in data
        # at the moment, it is not that reliable

        outlier_rows = pd.DataFrame()

        for col in self.features.columns:
            temp_df = self.features.copy()
            temp_df = pd.get_dummies(temp_df, drop_first=True)

            Q1 = temp_df[col].quantile(0.01)
            Q3 = temp_df[col].quantile(0.99)
            iqr = Q3 - Q1
            lower_bound = Q1 - thresh * iqr
            upper_bound = Q3 + thresh * iqr

            outliers = temp_df[(temp_df[col] < lower_bound) | (temp_df[col] > upper_bound)]
            if not outliers.empty:
                print(f"Outliers found in column '{col}'")
                outlier_rows = pd.concat([outlier_rows, outliers])

        if len(outlier_rows) == 0:
            print("No outliers found in any column.")

        else:
            print(f"Total outliers found: {len(outlier_rows)}")
            return outlier_rows