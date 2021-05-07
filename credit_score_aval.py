import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Suppress warnings 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import itertools
from datetime import datetime
from scipy import interp
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


print('Starting Model.')
print('Data Exploration by Graph Genaration.')

raw_train_data_path = 'source/treino.csv'
raw_test_data_path = 'source/teste.csv'

def data_loader(path):
    print('loading file from' + path)
    df = pd.read_csv(path)
    print(df.head(5))
    return df

df_raw_train = data_loader(raw_train_data_path) # Loading the raw file into a df variable
df_raw_test = data_loader(raw_test_data_path)

def data_sorting(df):
    global df_lgb_
    global target 
    global df_lgb
    global train_df
    global x
    df_lgb_ = df_raw_train.copy()
    target = df_raw_train['inadimplente']
    df_lgb = df_raw_train.drop(['inadimplente'], axis=1)
    train_df = df_raw_train.copy()
    x = df_raw_train.copy()

data_sorted = data_sorting(df_raw_train)

def target_examination():
    print("There are {}% target values with 1".format(100 * df_raw_train['inadimplente'].value_counts()[1]/df_raw_train.shape[0]))
    print('There is unbalancing in the data about the target value.')

target_examination = target_examination()

def missing_data_evaluation(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_train_data.head(10))
    print('Columns salario_mensal and numero_de_dependentes have many Null values, around salario_mensal 19.78% and numero_de_dependentes 2.61%')

missing_data_evaluation(df_raw_train)

def data_explore(df_train, df_test):
    print('Checking the Data Types')
    print('The data Types are:')
    print(df_train, df_test.dtypes.value_counts())
    print('The numeric variables are of 7 of the int64 type and 4 of the float64 (which can be either discrete or continuous).')
    print('Now Checking for duplicates:')
    features = df_train.columns.values[1:11]
    unique_max_train = []
    unique_max_test = []
    for feature in features:
        values = df_train[feature].value_counts()
        unique_max_train.append([feature, values.max(), values.idxmax()])
        values = df_test[feature].value_counts()
        unique_max_test.append([feature, values.max(), values.idxmax()])
    print(unique_max_train)
    print(unique_max_test)
    
    duplicate_train = np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).\
        sort_values(by = 'Max duplicates', ascending=False).head(10))
    
    duplicate_test = np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value'])).\
            sort_values(by = 'Max duplicates', ascending=False).head(10))
    print(duplicate_train)
    print(duplicate_test)
    print('Same columns in train and test set have very close number of duplicates of same or very close values. This is an interesting pattern that we might be able to use in the future ')

data_explore(df_raw_train, df_raw_test)


def data_correlation(df_train):
    correlations = df_train.corr()['inadimplente'].sort_values()
    print('Most Positive Correlations:\n', correlations.tail(5))
    print('\nMost Negative Correlations:\n', correlations.head(5))
    print('It is clear that idade and vezes_passou_de_30_59_dias have the most relevant correlation. ')
    corr_train = df_train.corr()
    plt.figure(figsize = (14, 10))
    print('Generating Correlation heatmap:')
    sns.heatmap(corr_train, cmap = "YlGnBu", vmin = -0.3, annot = True, vmax = 0.7)
    plt.title('Correlation Heatmap')
    plt.savefig('Correlation_Heatmap1.png')

data_correlation(df_raw_train)

def age_exploration(df):
    sns.displot(df, x=df['idade'], binwidth=5)
    # plt.hist(df['idade'], edgecolor = 'k')
    plt.title('Age of Clients'); plt.xlabel('Age (years)'); plt.ylabel('Count')
    plt.savefig('age_of_clients.png')
    print('The minimum age is {} and the maximum age is {}'.format(df['idade'].min(), df['idade'].max()))
    print('The are {} clientes with the age lesser then 20 and {} greater then 99'.format(len(df[df['idade']>99]),len(df[df['idade']<20])))
    print('Ploting distribution density of Inadimplents by age:')
    plt.figure(figsize = (16, 12))
    sns.kdeplot(df.loc[df['inadimplente'] == 0, 'idade'],linestyle="--", label = 'target == 0')
    # sns.kdeplot(data=df, x="idade", hue="inadimplente")
    sns.kdeplot(df.loc[df['inadimplente'] == 1, 'idade'], label = 'target == 1')
    plt.xlabel('Age (years)'); plt.ylabel('Density')
    plt.legend(['Inadimplente = 0','Inadimplente = 1' ])
    plt.title('Distribution of Ages')
    plt.savefig('Distribution_of_Ages.png')
    print('Density distribution ploted.')
    print('Grouping Inadimplente by dage group:')
    age_data = df[['inadimplente', 'idade']]
    age_data['age_binned'] = pd.cut(age_data['idade'], bins = np.linspace(20, 80, num = 6))
    print(age_data.head(10))
    age_groups  = age_data.groupby('age_binned').mean()
    print(age_groups)
    plt.figure(figsize = (10, 10))
    plt.bar(age_groups.index.astype(str), 100 * age_groups['inadimplente'])
    plt.xticks(rotation = 73); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by Age Group')
    plt.savefig('Failure_to_Repay_by_Age_Group.png')
    print('AGE FACTOR')
age_exploration(df_raw_train)


def open_cred_lines_explorer(df):
    print('Ploting Distribution of number of open credit lines')
    plt.figure(figsize = (12, 10))
    sns.kdeplot(df.loc[df['inadimplente'] == 0, 'numero_linhas_crdto_aberto'], label = 'target == 0')
    sns.kdeplot(df.loc[df['inadimplente'] == 1, 'numero_linhas_crdto_aberto'], label = 'target == 1')
    plt.xlabel('number of open credit lines'); plt.ylabel('Density')
    plt.legend(['Inadimplente = 0','Inadimplente = 1' ])
    plt.title('Distribution of number of open credit lines')
    plt.savefig('Distribution_of_number_of_open_credit_lines.png')
    print('Distribution of number of open credit lines ploted')
# print(df_raw_train)

open_cred_lines_explorer(df_raw_train)

def plot_dist_col(column, train, test):
    print('Ploting comparision of train and test by {}'.format(column))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.distplot(train[column].dropna(), color='green', ax=ax).set_title(column, fontsize=10)
    sns.distplot(test[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=10)
    plt.xlabel(column, fontsize=12)
    plt.legend(['train', 'test'])
    plt.savefig('{}.png'.format(column))

plot_dist_col('util_linhas_inseguras',df_raw_train,df_raw_test)

plot_dist_col('idade',df_raw_train,df_raw_test)

plot_dist_col('vezes_passou_de_30_59_dias',df_raw_train,df_raw_test)

plot_dist_col('salario_mensal',df_raw_train,df_raw_test)

plot_dist_col('numero_linhas_crdto_aberto',df_raw_train,df_raw_test)

plot_dist_col('numero_vezes_passou_90_dias',df_raw_train,df_raw_test)

plot_dist_col('numero_emprestimos_imobiliarios',df_raw_train,df_raw_test)

plot_dist_col('numero_de_vezes_que_passou_60_89_dias',df_raw_train,df_raw_test)

plot_dist_col('numero_de_dependentes',df_raw_train,df_raw_test)

print('with all of these we can have some visualization of the data e with')

def baseline_model(df_train, df_test):
    df_train = df_train.drop(columns = ['inadimplente'])
    # Feature names
    features = list(df_train.columns)
    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')
    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Fit on the df_training data
    imputer.fit(df_train)
    # Transform both df_training and testing data
    df_train = imputer.transform(df_train)
    df_test = imputer.transform(df_test)
    # Repeat with the scaler
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)

    print('df_Training data shape: ', df_train.shape)
    print('Testing data shape: ', df_test.shape)

baseline_model(df_raw_train,df_raw_test)


def df_drop_col(df):
    cols_to_drop = ['salario_mensal', 'salario_mensal', 'numero_de_dependentes']
    print('Droping Columns:')
    print(cols_to_drop)
    df_with_droped_cols = df.drop(columns=cols_to_drop)
    print('Columns Droped.')
    return df_with_droped_cols

df_train_droped_cols = df_drop_col(df_raw_train)
df_test_droped_cols = df_drop_col(df_raw_test)

def auc_score(y_true, y_pred):
    """
    Calculates the Area Under ROC Curve (AUC)
    """
    return roc_auc_score(y_true, y_pred)

# droped_null_columns(df_raw_train,df_raw_test)

print('Random Forest Modelling')

def random_forest_model(df_train, target, df_test):
    X_train, X_val, y_train, y_val = train_test_split(df_train, target,
                                                  test_size=0.30, 
                                                  random_state=2020, 
                                                  stratify=target)
    # Make the random forest classifier
    random_forest = RandomForestClassifier(n_estimators = 100, random_state = 2020, verbose = 1, n_jobs = -1)
    # Train on the training data
    random_forest.fit(X_train,y_train)
        # Extract feature importances
    
    features = list(df_train.columns)
    feature_importance_values = random_forest.feature_importances_
    feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

    # Get score on training set and validation set for random forest
    train_preds = random_forest.predict_proba(X_train, )[:, 1]
    # probability_class_1 = model.predict_proba(X)[:, 1]
    a = train_preds
    print(a)
    np.savetxt("foo.csv", a, delimiter=",", fmt='%f')
    val_preds = random_forest.predict_proba(X_val)[:, 1]
    train_score = roc_auc_score(y_train, train_preds)
    val_score = roc_auc_score(y_val, val_preds)
    aval_test = val_preds = random_forest.predict_proba(df_test)[:, 1]
    # Plot ROC curve
    # plot_curve(y_train, train_preds, y_val, val_preds, "Random Forest Baseline")

random_forest_model(df_train_droped_cols, target, df_test_droped_cols)

print('Logistic Regression')

