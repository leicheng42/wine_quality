# cr：晒月亮的孩子     url：https://zhuanlan.zhihu.com/p/63854175
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# 把特征数据进行标准化为均匀分布
def uniform_norm(X_in):
    X_max = X_in.max(axis=0)
    X_min = X_in.min(axis=0)
    X = (X_in - X_min) / (X_max - X_min)
    return X


# 主函数
if __name__ == "__main__":
    # 读取样本数据
    redwine_data = pd.read_csv("Wine_train.csv", sep=",")
    # 处理数据（丢弃序列号）
    redwine_data = redwine_data.drop('X', axis=1)
    # 生成数据集的特征集和标签集
    X = redwine_data.drop('quality', axis=1)
    Y = redwine_data['quality']
    # 对X矩阵进行归一化
    unif_X = uniform_norm(X)
    # 对样本数据进行训练集和测试集的划分
    unif_trainX, unif_testX, train_Y, test_Y = train_test_split(unif_X, Y, test_size=0.3, random_state=0)

    # 模型训练
    model = Ridge()  # L2正则的线性回归
    model.fit(unif_trainX, train_Y)

    # 模型评估
    print("训练集上效果评估 >>")
    r2 = model.score(unif_trainX, train_Y)
    print("R^2系数 ", r2)
    train_pred = model.predict(unif_trainX)
    mse = mean_squared_error(train_Y, train_pred)
    print("均方误差 ", mse)

    print("\n测试集上效果评估 >>")
    r2 = model.score(unif_testX, test_Y)
    print("R^2系数 ", r2)
    test_pred = model.predict(unif_testX)
    mse = mean_squared_error(test_Y, test_pred)
    # 等价于 mse = sum((test_pred-test_Y)**2) / test_Y.shape[0]
    print("均方误差", mse)

# output
# 训练集上效果评估 >>
# R^2系数  0.28394847065211826
# 均方误差  0.5614221687394368
#
# 测试集上效果评估 >>
# R^2系数  0.26814293239794884
# 均方误差 0.5593336193684975