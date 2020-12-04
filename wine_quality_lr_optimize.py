# 寻找红酒预测方法
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from itertools import combinations, permutations


def get_train_data_set():
    """
    获取训练数据集的特征和标签
    :return:
    """
    # 读取训练集数据
    df = pd.read_csv('Wine_train.csv', sep=',')
    # 处理数据（丢弃序列号）
    df = df.drop('X', axis=1)

    # 生成数据的特征和标签
    labels = df['quality']
    features = df.drop('quality', axis=1)

    return features, labels


def data_set_train(features, labels):
    """
    训练模型
    :param features: 数据集特征
    :param labels: 数据集标签
    :return:
    """
    # copy一个特征集
    features_copy = features
    # 提取数据集列特征
    columns = features.columns.tolist()

    # print(list(combinations(columns, 2)))

    # 使用组合数对模型进行择优
    acc_score_max = 0
    for i in range(1, len(columns) + 1):
        for lst in combinations(columns, i):
            features = features_copy
            lit = list(lst)
            features = features[lit]
            acc_score, column= linear_regression(features, labels)
            if acc_score > acc_score_max:
                acc_score_max = acc_score
                columns_max = column
    print(acc_score_max, columns_max)


    # # 线性回归 训练1：依次减少模型特征进行训练
    # for i in range(len(columns)):
    #     # 训练线性回归模型
    #     linear_regression(features, labels)
    #     features = features.drop(columns[-1], axis=1)
    #     columns.pop()


def linear_regression(features, labels):
    """
    线性回归模型
    :param features:
    :param labels:
    :return:
    """
    # 对原始数据集进行切分
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=0)
    # 训练线性回归模型
    LR = LinearRegression()
    LR.fit(train_features, train_labels)
    # 用线性回归模型对测试集进行预测
    prediction = LR.predict(test_features)

    # 对模型进行评估
    RMSE = np.sqrt(mean_squared_error(test_labels, prediction))
    print(train_features.columns.tolist())
    print('线性回归模型的预测误差:', RMSE)
    prediction = np.around(prediction, )
    print('模型准确率：', accuracy_score(prediction, test_labels), '\n')

    return accuracy_score(prediction, test_labels), train_features.columns.tolist()


def data_test():
    pass


if __name__ == '__main__':
    # 获取数据集
    data_features, data_labels = get_train_data_set()
    # 数据集训练
    data_set_train(data_features, data_labels)
    # 数据预测
    data_test()


# 模型最高准确率： 0.523931623931624 ['volatile.acidity', 'citric.acid', 'residual.sugar', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'pH', 'alcohol']
