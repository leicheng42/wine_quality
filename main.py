# 寻找红酒预测方法
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


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


def ridge(features, labels):
    """
    L2正则的线性回归模型
    :param features:
    :param labels:
    :return:
    """
    # 对原始数据集进行切分
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=0)
    # 训练L2正则的线性回归模型
    RR = Ridge()
    RR.fit(train_features, train_labels)
    # 用线性回归模型对测试集进行预测
    prediction = RR.predict(test_features)

    # 对模型进行评估
    RMSE = np.sqrt(mean_squared_error(test_labels, prediction))
    print(train_features.columns.tolist())
    print('L2正则的线性回归模型的预测误差:', RMSE)
    prediction = np.around(prediction, )
    print('模型准确率：', accuracy_score(prediction, test_labels), '\n')


def data_set_train(features, labels):
    """
    训练模型
    :param features: 数据集特征
    :param labels: 数据集标签
    :return:
    """
    # copy一个特征集
    features_copy = features.copy()
    # 提取数据集列特征
    columns = features.columns.tolist()

    # 线性回归 训练1：依次减少模型特征进行训练
    for i in range(len(columns)):
        # 训练线性回归模型
        linear_regression(features, labels)
        features = features.drop(columns[-1], axis=1)
        columns.pop()

    # 线性回归 训练2：删去密度特征
    features = features_copy
    features = features.drop(['density'], axis=1)
    linear_regression(features, labels)

    # 线性回归 训练3：删去密度、总二氧化硫特征
    features = features_copy
    features = features.drop(['density', 'total.sulfur.dioxide'], axis=1)
    linear_regression(features, labels)

    # 随机森林 训练1：依次减少模型特征进行训练
    features = features_copy
    columns = features.columns.tolist()
    for i in range(len(columns)):
        # 训练随机森林模型
        random_forest_regressor(features, labels)
        features = features.drop(columns[-1], axis=1)
        columns.pop()

    # GBDT模型 训练1：依次减少模型特征进行训练
    features = features_copy
    columns = features.columns.tolist()
    for i in range(len(columns)):
        # 训练GBDT模型模型
        gradient_boosting(features, labels)
        features = features.drop(columns[-1], axis=1)
        columns.pop()

    # L2正则的线性回归 训练1：依次减少模型特征进行训练
    features = features_copy
    columns = features.columns.tolist()
    for i in range(len(columns)):
        # 训练L2正则的线性回归模型
        ridge(features, labels)
        features = features.drop(columns[-1], axis=1)
        columns.pop()


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


def random_forest_regressor(features, labels):
    """
    随机森林模型
    :param features:
    :param labels:
    :return:
    """
    # 对原始数据集进行切分
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=0)
    # 训练随机森林模型
    RF = RandomForestRegressor()
    RF.fit(train_features, train_labels)
    # 用随机森林模型对测试集进行预测
    prediction = RF.predict(test_features)

    # 对模型进行评估
    RMSE = np.sqrt(mean_squared_error(test_labels, prediction))
    print(train_features.columns.tolist())
    print('随机森林模型的预测误差:', RMSE)
    prediction = np.around(prediction, )
    print('模型准确率：', accuracy_score(prediction, test_labels), '\n')


def gradient_boosting(features, labels):
    """
    GBDT模型
    :param features:
    :param labels:
    :return:
    """
    # 对原始数据集进行切分
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=0)
    # 训练GBDT模型
    GBDT = GradientBoostingRegressor()
    GBDT.fit(train_features, train_labels)
    # 用GBDT模型对测试集进行预测
    prediction = GBDT.predict(test_features)

    # 对GBDT模型进行评估
    RMSE = np.sqrt(mean_squared_error(prediction, test_labels))
    print(train_features.columns.tolist())
    print('GBDT模型的预测误差:', RMSE)
    prediction = np.around(prediction, )
    print('模型准确率：', accuracy_score(prediction, test_labels), '\n')


def data_test():
    pass


if __name__ == '__main__':
    # 获取数据集
    data_features, data_labels = get_train_data_set()
    # 数据集训练
    data_set_train(data_features, data_labels)
    # 数据预测
    data_test()


# output
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol']
# 线性回归模型的预测误差: 0.744198378276254
# 模型准确率： 0.5017094017094017
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates']
# 线性回归模型的预测误差: 0.7443160278392718
# 模型准确率： 0.5017094017094017
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH']
# 线性回归模型的预测误差: 0.7473105529599884
# 模型准确率： 0.4982905982905983
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density']
# 线性回归模型的预测误差: 0.7649862952201036
# 模型准确率： 0.4863247863247863
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide']
# 线性回归模型的预测误差: 0.818296017320836
# 模型准确率： 0.4658119658119658
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide']
# 线性回归模型的预测误差: 0.8300947155072071
# 模型准确率： 0.44871794871794873
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides']
# 线性回归模型的预测误差: 0.8278031160530699
# 模型准确率： 0.45042735042735044
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar']
# 线性回归模型的预测误差: 0.8469678600431805
# 模型准确率： 0.44358974358974357
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid']
# 线性回归模型的预测误差: 0.8502093306551073
# 模型准确率： 0.44358974358974357
#
# ['fixed.acidity', 'volatile.acidity']
# 线性回归模型的预测误差: 0.8501579791931787
# 模型准确率： 0.44358974358974357
#
# ['fixed.acidity']
# 线性回归模型的预测误差: 0.8689248145845151
# 模型准确率： 0.43675213675213675
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'pH', 'sulphates', 'alcohol']
# 线性回归模型的预测误差: 0.7495162916848362
# 模型准确率： 0.505982905982906
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'pH', 'sulphates', 'alcohol']
# 线性回归模型的预测误差: 0.7506241454439284
# 模型准确率： 0.5025641025641026
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol']
# 随机森林模型的预测误差: 0.6287375765539548
# 模型准确率： 0.6170940170940171
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates']
# 随机森林模型的预测误差: 0.6219601630687004
# 模型准确率： 0.629059829059829
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH']
# 随机森林模型的预测误差: 0.6235835230796355
# 模型准确率： 0.6230769230769231
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density']
# 随机森林模型的预测误差: 0.6329851944637065
# 模型准确率： 0.6205128205128205
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide']
# 随机森林模型的预测误差: 0.6596823892608129
# 模型准确率： 0.6042735042735042
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide']
# 随机森林模型的预测误差: 0.6704771231366307
# 模型准确率： 0.6017094017094017
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides']
# 随机森林模型的预测误差: 0.7019275049620312
# 模型准确率： 0.6
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar']
# 随机森林模型的预测误差: 0.7436432083944982
# 模型准确率： 0.5811965811965812
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid']
# 随机森林模型的预测误差: 0.8024259866322592
# 模型准确率： 0.541025641025641
#
# ['fixed.acidity', 'volatile.acidity']
# 随机森林模型的预测误差: 0.9130285133226601
# 模型准确率： 0.4512820512820513
#
# ['fixed.acidity']
# 随机森林模型的预测误差: 0.8781049531051002
# 模型准确率： 0.4256410256410256
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol']
# GBDT模型的预测误差: 0.6743501881230263
# 模型准确率： 0.5504273504273505
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates']
# GBDT模型的预测误差: 0.6711510217261337
# 模型准确率： 0.5504273504273505
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH']
# GBDT模型的预测误差: 0.6745845026941475
# 模型准确率： 0.5512820512820513
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density']
# GBDT模型的预测误差: 0.6857366828988531
# 模型准确率： 0.5487179487179488
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide']
# GBDT模型的预测误差: 0.7174351928693724
# 模型准确率： 0.5290598290598291
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide']
# GBDT模型的预测误差: 0.7234929565089966
# 模型准确率： 0.5205128205128206
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides']
# GBDT模型的预测误差: 0.7419185320916454
# 模型准确率： 0.5042735042735043
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar']
# GBDT模型的预测误差: 0.7835052295565126
# 模型准确率： 0.4846153846153846
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid']
# GBDT模型的预测误差: 0.8037436508063202
# 模型准确率： 0.4752136752136752
#
# ['fixed.acidity', 'volatile.acidity']
# GBDT模型的预测误差: 0.8408662877407554
# 模型准确率： 0.441025641025641
#
# ['fixed.acidity']
# GBDT模型的预测误差: 0.8790618834599532
# 模型准确率： 0.4282051282051282
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol']
# L2正则的线性回归模型的预测误差: 0.7500220556684208
# 模型准确率： 0.505982905982906
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates']
# L2正则的线性回归模型的预测误差: 0.8187504631151274
# 模型准确率： 0.46153846153846156
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH']
# L2正则的线性回归模型的预测误差: 0.8196362835520895
# 模型准确率： 0.4658119658119658
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density']
# L2正则的线性回归模型的预测误差: 0.8215539203266258
# 模型准确率： 0.4623931623931624
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide']
# L2正则的线性回归模型的预测误差: 0.8221137430760641
# 模型准确率： 0.4623931623931624
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide']
# L2正则的线性回归模型的预测误差: 0.8343718176528248
# 模型准确率： 0.44700854700854703
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides']
# L2正则的线性回归模型的预测误差: 0.8321686645376551
# 模型准确率： 0.44700854700854703
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar']
# L2正则的线性回归模型的预测误差: 0.8470813192306911
# 模型准确率： 0.4444444444444444
#
# ['fixed.acidity', 'volatile.acidity', 'citric.acid']
# L2正则的线性回归模型的预测误差: 0.8503044124082261
# 模型准确率： 0.4444444444444444
#
# ['fixed.acidity', 'volatile.acidity']
# L2正则的线性回归模型的预测误差: 0.850263562638994
# 模型准确率： 0.4427350427350427
#
# ['fixed.acidity']
# L2正则的线性回归模型的预测误差: 0.8689246873496626
# 模型准确率： 0.43675213675213675 s
