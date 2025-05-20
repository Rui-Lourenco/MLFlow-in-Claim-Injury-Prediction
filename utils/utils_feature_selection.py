# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Sklearn packages
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# embedded methods
from sklearn.linear_model import LassoCV
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import RFE

import warnings
warnings.filterwarnings('ignore')

from utils import *


def TestIndependence(X,y,var,alpha=0.05, verbose=True):
    dfObserved = pd.crosstab(y,X)
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(var)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)
        if not verbose:
            print(result)
    if verbose:
        print(result)


def feature_selection_RFE(X,y,n_features,model=None):
    best_score = 0
    best_features = []

    results = {}
    
    results = {}
    
    for feature in range(1,n_features):
        
        rfe = RFE(estimator=model, n_features_to_select=feature)
        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_]
        
        y_pred = rfe.predict(X)
        
        macro_f1 = f1_score(y, y_pred, average='macro')
        
        results[feature] = selected_features
        
        if macro_f1 > best_score:
            best_score = macro_f1
            best_features = selected_features.tolist()  
    
    return best_features


def feature_selection_Lasso(X,y):
    reg = LassoCV()
    reg.fit(X, y)
    coef = pd.Series(reg.coef_, index = X.columns)
    coef.sort_values()
    plot_importance(coef,'Lasso')
    return coef

def plot_importance(coef,name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(8,10))
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using " + name + " Model")
    plt.show()


def check_performace(model_copy,X,y,features_to_scale,feature_selection,n_folds = 5, random_state = 68+1):

    K_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold = 1

    avg_train = []
    avg_val = []
    model = model_copy
    for train_index, val_index in K_fold.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        remove_outliers(X_train)
        X_train, X_val = apply_frequency_encoding(X_train, X_val)

        NA_imputer(X_train,X_val)
        create_new_features(X_train,X_val)

        scaler = StandardScaler().fit(X_train[features_to_scale])
        X_train[features_to_scale]  = scaler.transform(X_train[features_to_scale])
        X_val[features_to_scale]  = scaler.transform(X_val[features_to_scale])  

        drop_list = []
        if feature_selection != []:
            for col in X_train.columns:
                if col not in feature_selection:
                    drop_list.append(col)
        X_train = X_train.drop(drop_list, axis=1)
        X_val = X_val.drop(drop_list, axis=1)

        model.fit(X_train, y_train)
    
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        f1_train = f1_score(y_train, y_train_pred, average='macro')
        f1_val = f1_score(y_val, y_val_pred, average='macro')
        
        avg_train.append(f1_train)
        avg_val.append(f1_val)

        print(f"Fold {fold} train F1 score: {f1_train:.4f}")
        print(f"Fold {fold} validation F1 score: {f1_val:.4f}")
        print(f"------------------------------")

        fold += 1

    print(f"Average Train F1 score: {sum(avg_train)/len(avg_train)}")
    print(f"Average Validation F1 score: {sum(avg_val)/len(avg_val)}")

