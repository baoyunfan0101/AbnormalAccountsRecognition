# Abnormal Accounts Recognition

## 文件说明
datasets // 数据集（训练集、测试集）  
feature engineering // 特征工程  
models // 风险模型  
references // 参考文献

## 测试环境
Python3.8 & MATLAB R2018a

## 任务描述
从数据集给出的基础信息、操作信息和交易信息中，提取出有效特征，建立账户特征模型，即账户特征与账户风险“label”之间的关系模型，从而实现风险账户识别。

## 特征工程
### 数据预处理
原始数据未发现重复记录，多数属性分布较为合理，不存在与常识不符的记录。部分属性缺失值过多（如服务3等级“service3_level”），在后续特征筛选过程中会删除这部分属性；部分属性有少量缺失，后续特征提取完成后会以0（即均值）进行填充。

另外，原始数据中的离散型类别属性均以类别编码（字符串形式）存在。对于部分有偏序关系的离散型属性，将其类别编码转换为数字编码；对于部分并列关系的离散型属性，将其类别编码转换为独热编码（one-hot编码）。

由于不同特征数据的量纲不一致，存在超出取值范围的离群数据，因此需进行数据标准化。这里基于原始数据的均值和标准差进行z-score标准化，以满足下列模型训练的需要，公式为

$$ (x^2 + x^y )^{x^y}+ x_1^2= y_1 - y_2^{x_1-y_1^2} $$

## 参考文献
[1] Bernhard H Schölkopf, Robert C Williamson, Alexander Smola, John C Shawe-Taylor, John C Platt. Support vector method for novelty detection[C]. NIPS'99: Proceedings of the 12th International Conference on Neural Information Processing Systems, 1999.  
[2] David M.J. Tax, Robert P.W. Duin. Support Vector Data Description[J]. Machine Learning, 2004, 54: 45-66.  
[3] Markus M. Breunig, Hans-Peter Kriegel, Raymond Tak Yan Ng, Jörg Sander. LOF: identifying density-based local outliers[C]. Proc. ACM SIGMOD 2000 Int. Conf. On Management of Data, 2000.  
[4] Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. Isolation Forest[C]. IEEE International Conference on Data Mining, 2008.  
[5] 朱佳俊, 陈功, 施勇, 薛质. 基于用户画像的异常行为检测[J]. 通信技术, 2017, 50(10): 2310-2315.  
[6] 崔景洋, 陈振国, 田立勤, 张光华. 基于机器学习的用户与实体行为分析技术综述[J/OL]. 计算机工程. https://doi.org/10.19678/j.issn.1000-3428.0062623.  
[7] 爱丽丝·郑, 阿曼达·卡萨丽. 精通特征工程[M]. 北京: 人民邮电出版社, 2019.
