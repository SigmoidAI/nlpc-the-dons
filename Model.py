#Importing all needed libraries and modules
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import catboost as cat
import pandas as pd
import numpy as np
import pickle



class Models:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, xgboost: bool = False, logistic_regression: bool = False, random_forest: bool = False,
     KNN: bool = False, gaussian: bool = False, cat: bool = False) -> None:
        '''
            The constructor of the customer satisfaction models.
        :param X: 'DataFrame'
            The features values of the dataset
        :param y: 'DataFrame'
            The target values of the dataset
        :param xgboost: 'Bool'
            The trigger of the XGBoost model
        :param logistic_regression: 'Bool'
            The trigger of the Logistic Regression model
        :param random_forest: 'Bool'
            The trigger of the RandomForestClassifier model
        :param KNN: 'Bool'
            The trigger of the KNeighborsClassifier model
        :param gaussian: 'Bool'
            The trigger of the GaussianNB model
        :param cat: 'Bool'
            The trigger of the CatBoostClassifier model
        '''
        self.X = X
        self.y = y
        self.xgboost = xgboost
        self.logistic_regression = logistic_regression
        self.random_forest = random_forest
        self.KNN = KNN
        self.gaussian = gaussian
        self.cat = cat

    def train(self):
        """
            The function is responsible for running different classification models
        """
        skf = StratifiedKFold(n_splits=5, shuffle = True)
        score = []
        if self.logistic_regression:
                for train_index, test_index in skf.split(self.X, self.y):
                  X_train = self.X.loc[train_index,:]
                  X_test = self.X.loc[test_index,:] 
                  y_train = self.y.loc[train_index, :]
                  y_test = self.y.loc[test_index, :]


                  lr = LogisticRegression()
                  lr.fit(X_train, y_train)
                  y_pred_lr = lr.predict(X_test)
                  score.append(accuracy_score(y_test, y_pred_lr))
                print("Accuracy (Logistic Regression): {}".format(sum(score)/len(score)))
                
        if self.cat:
            for train_index, test_index in skf.split(self.X, self.y):
                X_train = self.X.loc[train_index,:]
                X_test = self.X.loc[test_index,:] 
                y_train = self.y.loc[train_index, :]
                y_test = self.y.loc[test_index, :]


                reg_cat = cat.CatBoostClassifier(iterations=3)
                reg_cat.fit(X_train, y_train)
                y_pred_cat = reg_cat.predict(X_test)
                score.append(accuracy_score(y_test, y_pred_cat))
            print("Accuracy (Cat Regression): {}".format(sum(score)/len(score)))
            pickle.dump(reg_cat, open("../CatBoost.pkl", "wb"))

        if self.KNN:
            for train_index, test_index in skf.split(self.X, self.y):
                X_train = self.X.loc[train_index,:]
                X_test = self.X.loc[test_index,:] 
                y_train = self.y.loc[train_index, :]
                y_test = self.y.loc[test_index, :]

                knn = KNeighborsClassifier()
                knn.fit(X_train, y_train)
                y_pred_knn = knn.predict(X_test)
                score.append(accuracy_score(y_test, y_pred_knn))
            print("Accuracy (KNN): {}".format(sum(score)/len(score)))
            pickle.dump(knn, open("../KNN.pkl", "wb"))
        
        if self.random_forest:
            for train_index, test_index in skf.split(self.X, self.y):
                X_train = self.X.loc[train_index,:]
                X_test = self.X.loc[test_index,:] 
                y_train = self.y.loc[train_index, :]
                y_test = self.y.loc[test_index, :]

                forest = RandomForestClassifier()
                forest.fit(X_train, y_train)
                y_pred_forest = forest.predict(X_test)
                score.append(accuracy_score(y_test, y_pred_forest))
            print("Accuracy (Random Forest): {}".format(sum(score)/len(score)))
            pickle.dump(forest, open("../RandomForest.pkl", "wb"))

        if self.gaussian:
            for train_index, test_index in skf.split(self.X, self.y):
                X_train = self.X.loc[train_index,:]
                X_test = self.X.loc[test_index,:] 
                y_train = self.y.loc[train_index, :]
                y_test = self.y.loc[test_index, :]

                gauss = GaussianNB()
                gauss.fit(X_train, y_train)
                y_pred_gauss = gauss.predict(X_test)
                score.append(accuracy_score(y_test, y_pred_gauss))
            print("Accuracy (Gaussian): {}".format(sum(score)/len(score)))
            pickle.dump(gauss, open("../Gauss.pkl", "wb"))

        if self.xgboost:
            for train_index, test_index in skf.split(self.X, self.y):
                X_train = self.X.loc[train_index,:]
                X_test = self.X.loc[test_index,:] 
                y_train = self.y.loc[train_index, :]
                y_test = self.y.loc[test_index, :]

                xgboost = XGBClassifier()
                xgboost.fit(X_train, y_train)
                y_pred_xgb = xgboost.predict(X_test)
                score.append(accuracy_score(y_test, y_pred_xgb))
            print("Accuracy (XGBoost): {}".format(sum(score)/len(score)))
            pickle.dump(xgboost, open("../XGBoost.pkl", "wb"))
    
    def predict(X):
        """
            This function is used to predict future data, with a pretrained model, the default model is RandomForest.
        """
        model = pickle.load(open("RandomForest(2).pkl", 'rb'))
        y = model.predict(X)
        if y == 0:
            print('trash')
        elif y == 1:
            print('bug')
        else:
            print('feature')
