# https://www.kaggle.com/ellis773/wine-rating
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE    #SMOTE Technique
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as knn


if __name__ == '__main__':
    # 读取训练集数据
    df = pd.read_csv('Wine_train.csv', sep=',')
    # 处理数据（丢弃序列号）
    df = df.drop('X', axis=1)

    # 生成数据的特征和标签
    labels = df['quality']
    features = df.drop('quality', axis=1)

    X = df.drop(columns=['quality'])
    y = df['quality']

    # csdn 数据不平衡问题——SMOTE算法赏析
    # transform the dataset
    oversample = SMOTE(k_neighbors=2)
    X, y = oversample.fit_resample(X, y)

    # 对原始数据集进行切分
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    ks = []
    for i in range(1, 300):
        knn_regressor = knn(n_neighbors=i, weights='distance')
        knn_regressor.fit(X_train, y_train)
        y_pred = knn_regressor.predict(X_test).round(0).astype(int)

        ks.append(accuracy_score(y_test, y_pred))

    max_percent = max(ks)
    index = ks.index(max_percent) + 1
    print(max_percent, index)







