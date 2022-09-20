# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


class RF:
    def __init__(self):
        self.p = 0

    def fit(self, data, lable):
        rf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [3, 5, 7, 9, 11, 13],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }
        self.grid = GridSearchCV(rf, param_grid, cv=5,scoring='accuracy', n_jobs= -1)

        # 拟合网格搜索模型
        self.grid.fit(data, lable)
        # 打印调整后的最佳参数
        #print(self.grid.best_params_)

        self.p = self.grid.best_score_
        #print(self.grid.best_score_)

    def predict(self, x_test):
        y_pred = self.grid.predict(x_test)
        return y_pred
