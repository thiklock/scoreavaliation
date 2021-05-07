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

def data_loader(path):
    logging.info('loading file from' + path)
    df = pd.read_csv(path)
    logging.info(df.head(5))
    return df

cleaned_train_data = data_loader('cleaned_train_data.csv')
cleaned_test_data = data_loader('cleaned_test_data.csv')


def data_processing(df_train, df_test):
    logging.info('Starting Processment:')
    train = df_train.drop(columns = ['inadimplente'])
    test = df_test
    # # Feature names
    features = list(train.columns)
    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')
    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))
    logging.info('Processment Finished.')
    # # Fit on the training data
    imputer.fit(train)
    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(test)
    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    logging.info('Training data shape: ', train.shape)
    logging.info('Testing data shape: ', test.shape)

data_processed = data_processing(cleaned_train_data, cleaned_test_data)

def plot_confusion_matrix(cm, classes,normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def auc_score(y_true, y_pred):
    """
    Calculates the Area Under ROC Curve (AUC)
    """
    return roc_auc_score(y_true, y_pred)
def plot_curve(y_true_train, y_pred_train, y_true_val, y_pred_val, model_name):
    """
    Plots the ROC Curve given predictions and labels
    """
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_pred_train, pos_label=1)
    fpr_val, tpr_val, _ = roc_curve(y_true_val, y_pred_val, pos_label=1)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_train, tpr_train, color='black',
             lw=2, label=f"ROC train curve (AUC = {round(roc_auc_score(y_true_train, y_pred_train), 4)})")
    plt.plot(fpr_val, tpr_val, color='darkorange',
             lw=2, label=f"ROC validation curve (AUC = {round(roc_auc_score(y_true_val, y_pred_val), 4)})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xticks(fontsize=14)


def plot_pre_curve(y_test,probs):
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.title("precision recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the plot
    plt.show()

train = cleaned_train_data
target = train['inadimplente']

X_train, X_val, y_train, y_val = train_test_split(train, target,
                                                  test_size=0.30, 
                                                  random_state=2020, 
                                                  stratify=target)


# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)
# Train on the training data
log_reg.fit(X_train, y_train)

logging.info('Get score on training set and validation set for Logistic Regression')
# Get score on training set and validation set for Logistic Regression
train_preds = log_reg.predict_proba(X_train)[:, 1]
val_preds = log_reg.predict_proba(X_val)[:, 1]
train_score = auc_score(y_train, train_preds)
val_score = auc_score(y_val, val_preds)

# Plot ROC curve
logging.info('Ploting Logistic Regression Baseline')

plot_curve(y_train, train_preds, y_val, val_preds, "Logistic Regression Baseline")


logging.info('Script Ended')