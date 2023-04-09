# LR
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv('x_train.csv')
    Y = pd.read_csv('train_with_label/train_label.csv')['label']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    # 训练模型
    LR_model = LR(
        penalty='l2',  # 惩罚项
        dual=False,  # 对偶(或原始)方法
        tol=0.0001,  # 停止求解的标准
        C=1,  # 正则化系数λ的倒数
        fit_intercept=True,  # 存在截距或偏差
        intercept_scaling=1,
        class_weight=None,  # 分类模型中各类型的权重
        random_state=None,  # 随机种子
        solver='lbfgs',  # 优化算法选择参数
        max_iter=100,  # 算法收敛最大迭代次数
        multi_class='auto',  # 分类方式选择参数
        verbose=0,  # 日志冗长度
        warm_start=False,  # 热启动参数
        n_jobs=None,  # 并行数
        l1_ratio=None
    )
    LR_model = LR_model.fit(x_train, y_train)

    # 保存模型
    with open('LR_model.pkl', 'wb') as f:
        pickle.dump(LR_model, f)


# 本地测试
x_train, x_test, y_train, y_test = load_data()
train_model(x_train, y_train)

# 读取模型
with open('LR_model.pkl', 'rb') as f:
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
