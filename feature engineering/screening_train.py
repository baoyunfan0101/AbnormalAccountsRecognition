# screening_train
import pandas as pd
from sklearn.preprocessing import StandardScaler


X = pd.DataFrame()

df = pd.read_csv('preprocessing_train.csv')
X['op_recent_tm'] = df['op_recent_tm']
X['op_frequency'] = df['op_frequency']
X['op_interval'] = df['op_interval']
X['op_min_interval'] = df['op_min_interval']
X['trans_recent_tm'] = df['trans_recent_tm']
X['trans_frequency'] = df['trans_frequency']
X['trans_amount'] = df['trans_amount']
X['trans_interval'] = df['trans_interval']
X['trans_min_interval'] = df['trans_min_interval']

df = pd.read_csv('train_with_label/train_base.csv')
X['age'] = df['age']
X['using_time'] = df['using_time']
X['card_a_cnt'] = df['card_a_cnt']
X['card_b_cnt'] = df['card_b_cnt']
X['card_c_cnt'] = df['card_c_cnt']
X['card_d_cnt'] = df['card_d_cnt']
X['op1_cnt'] = df['op1_cnt']
X['op2_cnt'] = df['op2_cnt']
X['service1_cnt'] = df['service1_cnt']
X['service1_amt'] = df['service1_amt']
X['service2_cnt'] = df['service2_cnt']
X['agreement_total'] = df['agreement_total']
X['acc_count'] = df['acc_count']
X['login_cnt_period1'] = df['login_cnt_period1']
X['login_cnt_period2'] = df['login_cnt_period2']
X['ip_cnt'] = df['ip_cnt']
X['login_cnt_avg'] = df['login_cnt_avg']
X['login_days_cnt'] = df['login_days_cnt']
X['product7_cnt'] = df['product7_cnt']
X['product7_fail_cnt'] = df['product7_fail_cnt']

# 标准化
for col in ['op_interval', 'op_min_interval', 'trans_interval', 'trans_min_interval']:
    m = X[X[col] != float('inf')][col].max()
    X[col] = X[col].replace(float('inf'), m)
scalar = StandardScaler().fit(X)
X = scalar.transform(X)
X = pd.DataFrame(X)

df = pd.read_csv('preprocessing_train.csv')
X['sex'] = df['sex']
X['provider'] = df['provider']
X['level'] = df['level']
X['verified'] = df['verified']
X['provider'] = df['provider']
X['level'] = df['level']
X['verified'] = df['verified']
X['regist_type'] = df['regist_type']
X['agreement1'] = df['agreement1']
X['agreement2'] = df['agreement2']
X['agreement3'] = df['agreement3']
X['agreement4'] = df['agreement4']
# X['province'] = df['province']
# X['city'] = df['city']
X['balance'] = df['balance']
X['balance_avg'] = df['balance_avg']
X['balance1'] = df['balance1']
X['balance1_avg'] = df['balance1_avg']
X['balance2'] = df['balance2']
X['balance2_avg'] = df['balance2_avg']
X['service3'] = df['service3']
X['product1_amount'] = df['product1_amount']
X['product2_amount'] = df['product2_amount']
X['product3_amount'] = df['product3_amount']
X['product4_amount'] = df['product4_amount']
X['product5_amount'] = df['product5_amount']
X['product6_amount'] = df['product6_amount']

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
print(X)

X.fillna(0, inplace=True)
X.to_csv('x_train.csv', index=None)
