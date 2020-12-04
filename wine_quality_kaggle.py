# https://www.kaggle.com/danielj6/red-wine-quality-prediction-regression/notebook?scriptVersionId=44585586
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from typing import Dict
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV
from typing import List


def evaluate_model(estimator: BaseEstimator, cv: int = 10) -> Dict[str, float]:
    """Print and return cross validation of model
    """
    scoring = 'neg_mean_squared_error'
    scores = cross_validate(estimator, X_train, y_train, return_train_score=True, cv=cv, scoring=scoring)
    train_mean, train_std = -1 * scores['train_score'].mean(), scores['train_score'].std()
    print(f'Train MSE: {train_mean} ({train_std})')
    val_mean, val_std = -1 * scores['test_score'].mean(), scores['test_score'].std()
    print(f'Validation MSE: {val_mean} ({val_std})')
    fit_mean, fit_std = scores['fit_time'].mean(), scores['fit_time'].std()
    print(f'Fit time: {fit_mean} ({fit_std})')
    score_mean, score_std = scores['score_time'].mean(), scores['score_time'].std()
    print(f'Score time: {score_mean} ({score_std}')
    result = {
        'Train MSE': train_mean,
        'Train std': train_std,
        'Validation MSE': val_mean,
        'Validation std': val_std,
        'Fit time (s)': fit_mean,
        'Score time (s)': score_mean,
    }
    return result


if __name__ == '__main__':
    # load data
    wine = pd.read_csv('Wine_train.csv', sep=',')
    wine = wine.drop('X', axis=1)
    features = wine.drop('quality', axis='columns').columns.tolist()    # 特征列表
    X = wine.drop('quality', axis='columns')    # 特征
    y = wine['quality']     # 标签

    # split off some data for testing 分割训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ExtraTrees Regressor 极端随机森林回归
    best_model = ExtraTreesRegressor(n_estimators=100, min_samples_split=5)
    best_model.fit(X_train, y_train)
    print('Mean Squared Error of Tuned Model on Test Set:')
    prediction = best_model.predict(X_test)
    print(mean_squared_error(prediction, y_test))
    prediction = np.around(prediction, )
    print('ExtraTrees Regressor 模型准确率：', accuracy_score(prediction, y_test), '\n')

    # Dummy Regressor 简单规则回归，用于其他模型比较的基准
    best_model = DummyRegressor()
    best_model.fit(X_train, y_train)
    print('Mean Squared Error of Tuned Model on Test Set:')
    prediction = best_model.predict(X_test)
    print(mean_squared_error(prediction, y_test))
    prediction = np.around(prediction, )
    print('Dummy Regressor 模型准确率：', accuracy_score(prediction, y_test), '\n')

    # Linear Regression 线性回归
    best_model = LinearRegression()
    best_model.fit(X_train, y_train)
    print('Mean Squared Error of Tuned Model on Test Set:')
    prediction = best_model.predict(X_test)
    print(mean_squared_error(prediction, y_test))
    prediction = np.around(prediction, )
    print('Linear Regression 模型准确率：', accuracy_score(prediction, y_test), '\n')

    # ridge Regression 岭回归
    best_model = Pipeline([('scale', StandardScaler()), ('ridge', RidgeCV())])
    best_model.fit(X_train, y_train)
    print('Mean Squared Error of Tuned Model on Test Set:')
    prediction = best_model.predict(X_test)
    print(mean_squared_error(prediction, y_test))
    prediction = np.around(prediction, )
    print('ridge 模型准确率：', accuracy_score(prediction, y_test), '\n')

    # knn k近邻
    best_model = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=50)), ])
    # knn = KNeighborsRegressor()
    best_model.fit(X_train, y_train)
    print('Mean Squared Error of Tuned Model on Test Set:')
    prediction = best_model.predict(X_test)
    print(mean_squared_error(prediction, y_test))
    prediction = np.around(prediction, )
    print('knn 模型准确率：', accuracy_score(prediction, y_test), '\n')

    # Decision Tree Regressor 决策树
    best_model = DecisionTreeRegressor()
    best_model.fit(X_train, y_train)
    print('Mean Squared Error of Tuned Model on Test Set:')
    prediction = best_model.predict(X_test)
    print(mean_squared_error(prediction, y_test))
    prediction = np.around(prediction, )
    print('Decision Tree Regressor 模型准确率：', accuracy_score(prediction, y_test), '\n')

    # LGBM Regressor 基于学习算法的决策树
    best_model = lgbm.LGBMRegressor()
    best_model.fit(X_train, y_train)
    print('Mean Squared Error of Tuned Model on Test Set:')
    prediction = best_model.predict(X_test)
    print(mean_squared_error(prediction, y_test))
    prediction = np.around(prediction, )
    print('LGBM Regressor 模型准确率：', accuracy_score(prediction, y_test), '\n')

    # Mean Squared Error of Tuned Model on Test Set:
    # 0.36693415598290596
    # ExtraTrees Regressor 模型准确率： 0.6820512820512821
    #
    # Mean Squared Error of Tuned Model on Test Set:
    # 0.742681845850519
    # Dummy Regressor 模型准确率： 0.4705128205128205
    #
    # Mean Squared Error of Tuned Model on Test Set:
    # 0.5386362327422761
    # Linear Regression 模型准确率： 0.5294871794871795
    #
    # Mean Squared Error of Tuned Model on Test Set:
    # 0.5390997020119666
    # ridge 模型准确率： 0.5294871794871795
    #
    # Mean Squared Error of Tuned Model on Test Set:
    # 0.5153271794871794
    # knn 模型准确率： 0.558974358974359
    #
    # Mean Squared Error of Tuned Model on Test Set:
    # 0.6717948717948717
    # Decision Tree Regressor 模型准确率： 0.6166666666666667
    #
    # Mean Squared Error of Tuned Model on Test Set:
    # 0.41262605958560605
    # LGBM Regressor 模型准确率： 0.6294871794871795
