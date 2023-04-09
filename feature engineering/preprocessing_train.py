# preprocessing_train
import pandas as pd

# 读取文件
train_base = pd.read_csv('train_with_label/train_base.csv')
train_op = pd.read_csv('train_with_label/train_op.csv')
train_trans = pd.read_csv('train_with_label/train_trans.csv')
train_label = pd.read_csv('train_with_label/train_label.csv')

print('读取文件完成')

# 训练样本
X = pd.DataFrame()

# 属性数值化
for col in ['sex', 'provider', 'level', 'verified', 'regist_type', 'agreement1', 'agreement2', 'agreement3',
            'agreement4', 'service3', 'service3_level']:
    train_base[col] = train_base[col].apply(lambda x: x if pd.isnull(x) else x.replace('category ', ''))

for col in ['balance', 'balance_avg', 'balance1', 'balance1_avg', 'balance2', 'balance2_avg', 'product1_amount',
            'product2_amount', 'product3_amount', 'product4_amount', 'product5_amount', 'product6_amount']:
    train_base[col] = train_base[col].apply(lambda x: x if pd.isnull(x) else x.replace('level ', ''))

train_op['tm_diff'] = train_op['tm_diff'].apply(lambda x: pd.Timedelta(x, format='%d days %H:%M:%S').total_seconds())

train_trans['tm_diff'] = train_trans['tm_diff'].apply(
    lambda x: pd.Timedelta(x, format='%d days %H:%M:%S').total_seconds())

print('属性数值化完成')

# 样本编号user
X['user'] = train_base['user'].drop_duplicates()

# 标签label
X['label'] = train_label[train_label['user'] == X['user']]['label']

print('user和label插入成功')

# 基础base:
for col in ['sex', 'provider', 'level', 'verified', 'regist_type', 'agreement1', 'agreement2', 'agreement3',
            'agreement4', 'service3', 'service3_level', 'balance', 'balance_avg', 'balance1', 'balance1_avg',
            'balance2', 'balance2_avg', 'product1_amount', 'product2_amount', 'product3_amount', 'product4_amount',
            'product5_amount', 'product6_amount']:
    X[col] = train_base[train_base['user'] == X['user']][col]

print('基础信息插入成功')

# 操作op:
print('正在插入操作信息.. [', end='')
op_recent_tm = []
op_frequency = []
op_interval = []
op_min_interval = []
for idx, row in X.iterrows():
    tmp = train_op[train_op['user'] == row['user']]
    op_frequency.append(tmp['user'].count())

    if op_frequency[-1] == 0:
        op_recent_tm.append(0)
        op_interval.append(float('inf'))
        op_min_interval.append(float('inf'))

    elif op_frequency[-1] == 1:  # type(tmp) = <class 'pandas.core.series.Series'>
        op_recent_tm.append(tmp['tm_diff'].values[0])
        op_interval.append(float('inf'))
        op_min_interval.append(float('inf'))

    else:
        tm_diff = tmp['tm_diff'].sort_values().values.tolist()
        op_recent_tm.append(tm_diff[-1])

        op_interval.append((tm_diff[-1] - tm_diff[0]) / (op_frequency[-1] - 1))

        interval = []
        for i in range(1, op_frequency[-1]):
            interval.append(tm_diff[i] - tm_diff[i - 1])
        op_min_interval.append(min(interval))

    if idx % 300 == 0:
        print('#', end='')

print(']')

# 最近操作时间op_recent_tm
X['op_recent_tm'] = op_recent_tm

# 操作频次op_frequency
X['op_frequency'] = op_frequency

# 操作平均间隔op_interval
X['op_interval'] = op_interval

# 操作最小间隔op_min_interval
X['op_min_interval'] = op_min_interval

print('操作信息插入成功')

# 交易trans:
print('正在插入交易信息.. [', end='')
trans_recent_tm = []
trans_frequency = []
trans_amount = []
trans_interval = []
trans_min_interval = []
for idx, row in X.iterrows():
    tmp = train_trans[train_trans['user'] == row['user']]
    trans_frequency.append(tmp['user'].count())

    if trans_frequency[-1] == 0:
        trans_recent_tm.append(0)
        trans_amount.append(0)
        trans_interval.append(float('inf'))
        trans_min_interval.append(float('inf'))

    elif trans_frequency[-1] == 1:  # type(tmp) = <class 'pandas.core.series.Series'>
        trans_recent_tm.append(tmp['tm_diff'].values[0])
        trans_amount.append(tmp['amount'].values[0])
        trans_interval.append(float('inf'))
        trans_min_interval.append(float('inf'))

    else:
        trans_amount.append(tmp['amount'].sum() / trans_frequency[-1])

        tm_diff = tmp['tm_diff'].sort_values().values.tolist()
        trans_recent_tm.append(tm_diff[-1])

        trans_interval.append((tm_diff[-1] - tm_diff[0]) / (trans_frequency[-1] - 1))

        interval = []
        for i in range(1, trans_frequency[-1]):
            interval.append(tm_diff[i] - tm_diff[i - 1])
        trans_min_interval.append(min(interval))

    if idx % 300 == 0:
        print('#', end='')

print(']')

# 最近交易时间trans_recent_tm
X['trans_recent_tm'] = trans_recent_tm

# 交易频次trans_frequency
X['trans_frequency'] = trans_frequency

# 交易金额trans_amount
X['trans_amount'] = trans_amount

# 交易平均间隔trans_interval
X['trans_interval'] = trans_interval

# 交易最小间隔trans_min_interval
X['trans_min_interval'] = trans_min_interval

print('交易信息插入成功')

# 写入文件
X.to_csv('preprocessing_train.csv', index=None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
print(X)
