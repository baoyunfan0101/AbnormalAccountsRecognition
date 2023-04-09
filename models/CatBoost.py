# CatBoost
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv('x_train.csv')
    Y = pd.read_csv('train_with_label/train_label.csv')['label']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    # 训练模型
    CatBoost_model = CatBoostClassifier(
        iterations=1000,  # 迭代次数，解决机器学习问题能够构建的最大树的数目，default=1000
        learning_rate=0.03,  # 学习率，default=0.03
        depth=6,  # 树的深度，default=6
        l2_leaf_reg=20.0,  # L2正则化数，default=3.0
        model_size_reg=None,  # 模型大小正则化系数，数值越大模型越小，仅在有类别型变量时有效，取值范围从0到inf，GPU计算时不可用，default=None
        loss_function='Logloss',  # 损失函数，分类任务：default='Logloss'，回归任务：default='RMSE'
        logging_level=None,  # 打印的日志级别，default=None
        class_weights=None  # 类别权重，default=None
    )
    CatBoost_model = CatBoost_model.fit(x_train, y_train)

    # 保存模型
    with open('CatBoost_model.pkl', 'wb') as f:
        pickle.dump(CatBoost_model, f)


# 本地测试
x_train, x_test, y_train, y_test = load_data()
train_model(x_train, y_train)

# 读取模型
with open('CatBoost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# accuracy
accuracy = model.score(x_test, y_test) * 100
print('accuracy:', accuracy, '%')

# auc
from sklearn.metrics import roc_auc_score

y_score = model.predict_proba(x_test)[:, 1]
score = roc_auc_score(y_test, y_score) * 100
print('auc:', score, '%')

# precision & recall
from sklearn.metrics import precision_score, recall_score, f1_score

y_score = model.predict(x_test)
precision = precision_score(y_test, y_score, average='binary') * 100
recall = recall_score(y_test, y_score, average='binary') * 100
f1 = f1_score(y_test, y_score, average='binary') * 100
print('precision:', precision, '%')
print('recall:', recall, '%')
print('f1:', f1, '%')
