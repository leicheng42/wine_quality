# 导入所需包   cr:mmggqq    https://zhuanlan.zhihu.com/p/33266192
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib


# 读取样本数据
data = pd.read_csv("Wine_train.csv", sep=",")
# 处理数据（丢弃序列号）
data = data.drop('X', axis=1)

# 将数据集分割为训练集和测试集
Y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)

# 创建管道
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# 声明模型需要关注的超参
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

# 用K折交叉检验和网格搜索对模型进行训练和调参
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, Y_train)

# 利用得到的模型进行预测并用预测结果对模型进行性能评估
pred = clf.predict(X_test)
r2_score(Y_test, pred)
print(mean_squared_error(Y_test, pred))

pred = np.around(pred, )
print('模型准确率：', accuracy_score(pred, Y_test), '\n')

# 保存模型以便将来使用
# joblib.dump(clf, 'rf_regressor.pkl')

# 取模型来使用
# clf2 = joblib.load('rf_regressor.pkl')

# output:
# 0.4017266666666666                        0.3982020512820513
# 模型准确率： 0.6461538461538462             0.6551282051282051