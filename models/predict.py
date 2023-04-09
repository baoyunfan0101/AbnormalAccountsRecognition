# predict
import pandas as pd
import pickle


X = pd.read_csv('x_test.csv')
res = pd.read_csv('test_without_label/submit_example.csv')

# 读取模型
with open('CatBoost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 预测
y_score = model.predict_proba(X)[:, 1]
print(y_score)
res['label'] = y_score
res.to_csv('res.csv', index=None)
