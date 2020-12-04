# https://www.kaggle.com/srinivasat16/red-wine-rating-knn-classifier
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE    #SMOTE Technique
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # 读取训练集数据
    df = pd.read_csv('Wine_train.csv', sep=',')
    # 处理数据（丢弃序列号）
    df = df.drop('X', axis=1)

    X = df.drop(columns=['quality'])
    y = df['quality']

    # # 生成数据的特征和标签
    # labels = df['quality']
    # features = df.drop('quality', axis=1)

    # # 对原始数据集进行切分
    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
    #                                                                             random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # csdn 数据不平衡问题——SMOTE算法赏析
    # transform the dataset
    oversample = SMOTE(k_neighbors=2)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)



    # 可以优化
    k = 1

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # print(X_test)
    print(y_pred)
    print('模型准确率：', accuracy_score(y_pred, y_test), '\n')

    # test_features = scaler.fit_transform(test_features)
    # y_pred = knn.predict(test_features)
    # prediction = np.around(y_pred, )
    # print('实际模型准确率：', accuracy_score(y_pred, test_labels), '\n')

# k=1
# 模型准确率： 0.8788252714708786
# 实际模型准确率： 0.8760683760683761





