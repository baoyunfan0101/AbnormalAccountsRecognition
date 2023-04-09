# SVM
import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv('x_train.csv')
    Y = pd.read_csv('train_with_label/train_label.csv')['label']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    # 训练模型
    SVM_model = svm.SVC(
        C=1.0,  # 惩罚参数
        kernel='rbf',  # 核函数
        degree=3,  # 多项式维度
        gamma='auto',
        coef0=0.0,  # 核函数常数项
        shrinking=True,  # 采用shrinking heuristic方法
        probability=True,  # 采用概率估计
        tol=0.001,  # 停止训练的误差值
        cache_size=200,  # 核函数cache缓存
        class_weight=None,  # 类别权重
        verbose=False,  # 允许冗余输出
        max_iter=-1,  # 最大迭代次数无约束
        decision_function_shape='ovo',  # 多分类策略
        random_state=None,  # 随机种子
    )
    SVM_model = SVM_model.fit(x_train, y_train)

    # 保存模型
    with open('SVM_model.pkl', 'wb') as f:
        pickle.dump(SVM_model, f)


# 本地测试
x_train, x_test, y_train, y_test = load_data()
train_model(x_train, y_train)

# 读取模型
with open('SVM_model.pkl', 'rb') as f:
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
