# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np


class SVM:
    def __init__(self):
        self.p = 0

    def fit(self, data, lable):
        svc = svm.SVC()
        param_grid = {
            'C': [0.1, 1, 10, 100],  # 正则化参数
            'gamma': [0.001, 0.01, 1, 10, 100],  # 核系数
            #'kernel': ['linear', 'poly', 'sigmoid']
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # 内核函数(线性核'linear'、多项式核'poly'、高斯核'rbf'、核函数'sigmoid')
        }
        self.grid = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs= -1)

        # 拟合网格搜索模型
        self.grid.fit(data, lable)
        # 打印调整后的最佳参数
        #print(self.grid.best_params_)
        self.p = self.grid.best_score_
        #print(self.grid.best_score_)

    def predict(self, x_test):
        y_pred = self.grid.predict(x_test)
        return y_pred
