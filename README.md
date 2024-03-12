# ML Application README

This application automates the process of Exploratory Data Analysis (EDA), feature engineering, model selection, and model training for both regression and classification tasks. The user can simply input a CSV file, specify the target column, and the application will handle the rest, including handling missing values, encoding categorical features, selecting the appropriate model, and hyperparameter tuning.

## Files

### 1. app.py

```python
import os
from joblib import dump
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from auto_eda import AutoEDA
from regressor import Regressor
from classifier import Classifier
from termcolor import colored

# Implementation details can be found in the app.py file.
```

### 2. auto_eda.py

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer

# Implementation details can be found in the auto_eda.py file.
```

### 3. classifier.py

```python
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Implementation details can be found in the classifier.py file.
```

### 4. regressor.py

```python
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

# Implementation details can be found in the regressor.py file.
```

## Usage

1. **Run the Application**: Execute `app.py` using Python to launch the application.
2. **Select CSV File**: Use the file explorer window to select the CSV file containing your dataset.
3. **Specify Target Column**: Enter the name of the target column when prompted.
4. **View DataFrame**: Preview the loaded DataFrame in the terminal.
5. **Handle Missing Values**: Choose how to handle missing values (interpolation, imputation, or removal).
6. **Choose Model Type**: Based on the target column type, the application automatically selects either regression or classification.
7. **Model Training**: Models are trained and optimized using cross-validation and hyperparameter tuning.
8. **Evaluate Models**: View evaluation metrics for each model, including accuracy, precision, recall, F1-score, etc.
9. **Dump Final Model**: Optionally, dump the final trained model as a `.joblib` file for later use.

## Dependencies

- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `tkinter`
- `joblib`
- `termcolor`
