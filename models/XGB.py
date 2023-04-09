# XGB
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv('x_train.csv')
    Y = pd.read_csv('train_with_label/train_label.csv')['label']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    # 训练模型
    XGB_model = XGBClassifier(
        max_depth=6,  # 树的深度
        learning_rate=0.1,  # 学习率
        n_estimators=100,  # 迭代次数(决策树个数)
        silent=False,  # 是否输出中间过程
        objective='binary:logitraw',  # 目标函数
        booster='gbtree',  # 基分类器
        nthread=-1,  # 使用全部CPU进行并行运算
        gamma=10,  # 惩罚项系数
        min_child_weight=1,  # 最小叶子节点样本权重和
        max_delta_step=0,  # 对权重改变的最大步长无约束
        subsample=1,  # 每棵树训练集占全部训练集的比例
        colsample_bytree=1,  # 每棵树特征占全部特征的比例
        colsample_bylevel=1,
        eta=0.1,
        reg_alpha=0,  # L1正则化系数
        reg_lambda=1,  # L2正则化系数
        scale_pos_weight=None,  # 正样本的权重
        base_score=0.5,
        random_state=0,
        seed=None,  # 随机种子
        missing=None,
        use_label_encoder=False
    )
    XGB_model = XGB_model.fit(x_train, y_train)

    # 保存模型
    with open('XGB_model.pkl', 'wb') as f:
        pickle.dump(XGB_model, f)


# 本地测试
x_train, x_test, y_train, y_test = load_data()
train_model(x_train, y_train)

# 读取模型
with open('XGB_model.pkl', 'rb') as f:
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
print(y_score)
precision = precision_score(y_test, y_score, average='binary') * 100
recall = recall_score(y_test, y_score, average='binary') * 100
f1 = f1_score(y_test, y_score, average='binary') * 100
print('precision:', precision, '%')
print('recall:', recall, '%')
print('f1:', f1, '%')
