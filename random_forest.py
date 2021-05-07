import pandas as pd
import matplotlib.pyplot as plt
import logging


from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_mat
from sklearn.metrics import confusion_matrix

import itertools
from datetime import datetime
from scipy import interp
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

import logging
import logging


print('Creating the Random Forest Model.') 

def data_loader(path):
    print('loading file from' + path)
    df = pd.read_csv(path)
    print(df.head(5))
    return df

cleaned_train_data = data_loader('cleaned_train_data.csv')
cleaned_test_data = data_loader('cleaned_test_data.csv')

df_lgb_ = cleaned_train_data.copy()
target = cleaned_train_data['inadimplente']
df_lgb = cleaned_train_data.drop(['inadimplente'], axis=1)
train_df = cleaned_train_data.copy()
x = cleaned_train_data.copy()
features = list(cleaned_train_data.columns)

X_train, X_val, y_train, y_val = train_test_split(cleaned_train_data, target,
                                                  test_size=0.30, 
                                                  random_state=2020, 
                                                  stratify=target)

def random_forest_model():
    random_forest = RandomForestClassifier(n_estimators = 100, random_state = 2020, verbose = 1, n_jobs = -1)
    print('Training the training data')
    random_forest.fit(X_train,y_train)
    print('Extract feature importances')
    feature_importance_values = random_forest.feature_importances_
    feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
    print('Get score on training set and validation set for random forest')
    train_preds = random_forest.predict_proba(X_train)[:, 1]
    val_preds = random_forest.predict_proba(X_val)[:, 1]
    train_score = auc_score(y_train, train_preds)
    val_score = auc_score(y_val, val_preds)

random_forest_model = random_forest_model()