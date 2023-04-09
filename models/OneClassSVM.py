# OneClassSVM
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv('x_train.csv')
    Y = pd.read_csv('train_with_label/train_label.csv')['label']
    X['label'] = Y

    x_train, x_test, _, y_test = train_test_split(X, Y, test_size=0.2)

    x_train = x_train[x_train['label'] == 0]
    x_train = x_train.drop('label', axis=1)
    x_test = x_test.drop('label', axis=1)
    y_test = y_test.apply(lambda x: -1 if x == 1 else 1)

    return x_train, x_test, y_test


def train_model(x_train):
    # 训练模型
    OneClassSVM_model = OneClassSVM(
        kernel='rbf',  # 核函数
        degree=3,  # 多项式维度
        gamma='auto',
        coef0=0.0,  # 核函数常数项
        shrinking=True,  # 采用shrinking heuristic方法
        tol=0.001,  # 停止训练的误差值
        cache_size=200,  # 核函数cache缓存
        verbose=False,  # 允许冗余输出
        max_iter=-1,  # 最大迭代次数无约束
    )
    OneClassSVM_model = OneClassSVM_model.fit(x_train)

    # 保存模型
    with open('OneClassSVM_model.pkl', 'wb') as f:
        pickle.dump(OneClassSVM_model, f)


# 本地测试
x_train, x_test, y_test = load_data()
train_model(x_train)

# 读取模型
with open('OneClassSVM_model.pkl', 'rb') as f:
    model = pickle.load(f)

# accuracy
y_score = model.predict(x_test)
y_test = np.array(y_test.values.tolist())
accuracy = 0
for i in range(len(y_test)):
    if y_test[i] == y_score[i]:
        accuracy += 1
accuracy = accuracy / len(y_test) * 100
print('accuracy:', accuracy, '%')

# auc
from sklearn.metrics import roc_auc_score

score = roc_auc_score(y_test, y_score) * 100
print('auc:', score, '%')
