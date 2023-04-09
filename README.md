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

$$
{X'}_{i} = \frac{X_{i} - {\overset{-}{X}}_{i}}{S}

$$

其中${X'}_{i}$为数据标准化后的特征；$X_{i}$原数据的特征；${\overset{-}{X}}_{i}$为原数据特征的均值；S为原数据特征的标准差，其计算公式为$\sqrt{\frac{\sum\limits_{i = 1}^{n}\left( {x_{i} - \overset{-}{x}} \right)^{2}}{n - 1}}$。由于后续可能还会进行特征衍生，实际操作中数据标准化可以在特征工程结束后进行。

*测试集和训练集数据预处理的Python脚本分别在“preprocessing_train.py”和“preprocessing_test.py”中。*

### 特征的衍生和筛选

同一账户的操作和交易信息显然是账户特征模型的重点，而其操作和交易的时间信息（对应属性“tm_diff”）更是建立模型的重中之重。为此，我们参考RFM分析方法，对相关时间信息进行特征衍生。

RFM分析方法中的“RFM”分别指的是Recency（距离最近一次交易）、Frequency（交易频率）和Monetary（交易金额）。参考此方法的基本思想，我们从账户操作信息中提取出四个特征，分别为最近操作时间“op_recent_tm”、操作频次“op_frequency”、操作平均间隔“op_interval”和操作最小间隔“op_min_interval”；从账户交易信息中提取出五个特征，分别为最近交易时间“trans_recent_tm”、交易频次“trans_frequency”、交易金额“trans_amount”、交易平均间隔“trans_interval”和交易最小间隔“trans_min_interval”。

其中，同时在特征中保留平均间隔与最小间隔有特别的考虑。一方面，从专业角度来说，操作和交易的最小间隔是判断账户是否为人工处理的重要标准，对账户异常的识别有着特殊的价值；另一方面，平均间隔仅与账户的最早和最晚一次的操作或交易有关，而加入最小间隔能够更有效地利用数据，更完整地反映RFM分析方法中Frequency的概念。

特征的筛选过程中，除删除在上述“数据质量分析及数据预处理”部分提及的缺失值过多的属性外，还依据下面特征分析的结果进行了进一步地筛选，下面将会详细阐述。

*测试集和训练集的特征衍生也在“preprocessing_train.py”和“preprocessing_test.py”中，与数据预处理同步进行。测试集和训练集特征筛选的Python脚本分别在“screening_train.py”和“screening_test.py”中。*

### 特征分析

#### 特征重要性评估

**WOE**（Weight of Evidence，证据权重），是对原始自变量的一种编码形式，在对某个评价指标进行分组、离散化处理后，由下面公式计算

$$
{WOE}_{i} = ln\left( \frac{{py}_{i}}{{pn}_{i}} \right) = ln\left( \frac{\frac{\# y_{i}}{\# y_{T}}}{\frac{\# n_{i}}{\# n_{T}}} \right)

$$

其中${WOE}_{i}$为第i组的WOE；${py}_{i}$为第i组响应客户（即该问题中的风险账户）占所有样本中响应客户的比例；${pn}_{i}$为第i组未响应客户占所有样本中未响应客户的比例；$\# y_{i}$为第i组响应客户的数量；$\# y_{T}$为第i组未响应客户的数量；$\# n_{i}$为所有样本中响应客户的数量；$\# n_{T}$为所有样本中未响应客户的数量。

**IV**（Information Value，信息价值），综合考虑了每组样本的WOE以及其在总体样本中所占的比例，可以看作WOE的加权和，在该问题中能够反映某一特征对账户风险的贡献率。某一组IV的具体计算公式为

$$
{IV}_{i} = \left( {py}_{i} - {pn}_{i} \right) \times {WOE}_{i} = \left( \frac{\# y_{i}}{\# y_{T}} - \frac{\# n_{i}}{\# n_{T}} \right) \times ln\left( \frac{\frac{\# y_{i}}{\# y_{T}}}{\frac{\# n_{i}}{\# n_{T}}} \right)

$$

其中${IV}_{i}$为第i组的IV。某个特征IV的计算公式为

$$
IV = {\sum\limits_{i = 1}^{n}{IV}_{i}}

$$

其中n为组数。

考察上述数据预处理和特征衍生后得到的特征，除仅有0和1两种取值的布尔型离散特征外，分别以其它的每个特征为标准，对数据进行卡方分箱，将所有数据划分为5组，再计算其IV，结果如下图所示。

![image](https://github.com/baoyunfan0101/AbnormalAccountsRecognition/blob/main/static/iv.jpg)

从图中不难发现，多数特征与账户风险“label”的关联性均在合理范围内。对于部分IV值极小的特征，可以将其舍去。

*数据分析的Python脚本在“features.py”和“iv.m”中。*

#### 特征相关性分析

计算上述各评价指标的相关系数矩阵，并绘制热力图，如下图所示。

![image](https://github.com/baoyunfan0101/AbnormalAccountsRecognition/blob/main/static/correlation.png)

从热力图中可知，大部分特征之间的相关性都在合理范围内。对于相关性过强的特征，可以采取合并特征的措施。

*数据分析的Python脚本也在“features.py”中。*

## 模型训练与优化

### 逻辑回归

**逻辑回归**（Logistic Regression，LR）是一种广义的线性回归分析模型，常用于解决二分类问题。

在账户风险模型中，设因变量账户风险“label”为y，其仅有1和0两个取值，可以看作二分类问题。若在自变量x=X的条件下因变量y=1的概率为p，记作$p = P\left( y = 1 \middle| X \right)$，则y=0的概率为$1 - p$，将因变量取1和0的概率比值$\frac{p}{1 - p}$记为优势比，对优势比取自然对数，即可得到Sigmoid函数

$$
Sigmoid(p) = ln\left( \frac{p}{1 - p} \right)

$$

令$Sigmoid(p) = z$，则有

$$
p = \frac{1}{1 + e^{- z}}

$$

设各特征的向量为X，系数向量为β，代入上式的z中，即得到回归模型的表达式

$$
h(x) = \frac{1}{1 + e^{- X\beta^{T}}}

$$

其中，h(x)的取值范围为[0,1]，可以表示题目所需的账户风险“label”的预测值。又h(x)≥0.5时令y=1，h(x)<0.5时令y=0，即可实现二分类。

*逻辑回归模型相关的Python脚本在“LR.py”中。*

### 支持向量机

**支持向量机**（Support Vector Machine，SVM）是一类按监督学习方式对数据进行二元分类的广义线性分类器，对于小样本、复杂模型的学习表现出较好的效果。

支持向量机通过最大边距超平面实现类别的划分，即将上述特征视为高维空间上的点，并求解与两类点的边距最大的超平面，设为$wx + b = 0$。称同一类别点中到超平面距离最近的点为支持向量，记为$z_{0}$，则需要使支持向量到超平面的距离尽可能远。首先，任意一点x到超平面的距离为

$$
d = \frac{\left| {wx + b} \right|}{\left\| w \right\|}

$$

其中$\left\| w \right\|$选取w的2-范数，即$\left\| w \right\| = \sqrt{\sum\limits_{i}w_{i}^{2}}$。又由支持向量的定义，有

$$
\frac{\left| {wx + b} \right|}{\left\| w \right\|} \geq \frac{\left| {wz_{0} + b} \right|}{\left\| w \right\|} = d_{0}

$$

化简可得

$$
\left| \frac{wx + b}{\left\| w \right\| d_{0}} \right| \geq 1

$$

为便于进一步推导与优化，由$\left\| w \right\| d_{0}$为正数，可令其为1，则有

$$
\left| {wx + b} \right| \geq 1

$$

又因为要想使$d_{0}$尽可能大，应使$\frac{1}{\left\| w \right\|}$尽可能大，由此得出支持向量机模型

$$
\max\limits_{}\frac{1}{\left\| w \right\|} \quad s.t.\left| {wx + b} \right| \geq 1

$$

*支持向量机模型相关的Python脚本在“SVM.py”中。*

### XGBoost

**XGBoost**（eXtreme Gradient Boosting，XGB）是梯度提升决策树（Gradient Boosting Decision Tree，GBDT）的一种，由集成的CART回归树构成，在很多情景下都表现出了出色的效率与较高的预测准确度。

XGBoost采用前向分布算法，学习包含K棵树的加法模型

$$
{\hat{y}}_{i} = {\sum\limits_{k = 1}^{K}{f_{k}\left( x_{i} \right)}}, \quad f \in F

$$

其中$f_{k}$为第k棵回归树模型；F对应回归树组成的函数空间。其目标函数定义为

$$
Obj(\Theta) = {\sum\limits_{i = 1}^{N}{l\left( {y_{i},{\hat{y}}_{i}} \right)}} + {\sum\limits_{j = 1}^{t}{\Omega\left( f_{j} \right)}}, \quad f_{j} \in F

$$

其中l为损失函数；Ω为正则化函数，与模型的复杂程度相关。正则项的加入能够有效防止模型过度拟合。

*XGBoost模型相关的Python脚本在“XGB.py”中。*

### CatBoost

**CatBoost**得名于“Category”和“Boosting”，是俄罗斯的搜索巨头Yandex在2017年开源的机器学习库，是Boosting族算法的一种。CatBoost和XGBoost、LightGBM并称为GBDT的三大主流神器，都是在GBDT算法框架下的一种改进实现。

CatBoost模型与上述XGBoost模型相比，具有以下特点：

- 采用创新算法，将类别特征处理为数值型特征；
- 使用组合类别特征，利用特征与特征之间的联系，极大地丰富了特征维度；
- 采用排序提升的方法对抗训练集中的噪声点，从而避免梯度估计的偏差，进而解决预测偏移的问题；
- 采用了完全对称树作为基模型。

此模型在独立模型中对此问题的效果最好。

*CatBoost模型相关的Python脚本在“CatBoost.py”中。*

### 一类支持向量机

经测试，上面几种模型在此问题上的表现并不理想。

一种合理的猜想是，用户的正常行为都比较类似，但异常行为的特征各异（亦或用户的异常行为都比较类似，但正常行为的特征各异），导致传统的监督学习难以区分正常和异常行为。

另一种猜想是，上面的模型仅适用于检测**离群点**（outlier detection），即存在于训练集中的异常点，而不适用于检测**奇异点**（novelty detection），即未在训练集中出现的新类型的样本。

由此，将已经完成各项处理的数据按照账户风险“label”划分为正常行为集和异常行为集，引入下面模型。

**一类支持向量机**（One Class Support Vector Machine，One Class SVM）是一类典型的单分类模型，常用于奇异点检测。

One Class SVM模型的训练集中应只包含一类行为。One Class SVM的定义方式有很多，比较常见的有以下两种。

在参考文献[1]中提出的One Class SVM方法（可简称为OCSVM）实质是将所有数据点与零点在特征空间F分离，并且最大化分离超平面到零点的距离。其优化目标与经典的SVM有所不同，要求

$$
{\min\limits_{w,\zeta_{i},\rho}{\frac{1}{2}\left\| w \right\|^{2}}} + \frac{1}{\nu n}{\sum\limits_{i = 1}^{n}\zeta_{i}} - \rho \quad s.t.\left( {w^{T}\phi\left( x_{i} \right)} \right) > \rho - \zeta_{i}, \quad i = 1,..,n

$$

其中$\zeta_{i}$为松弛变量且满足$\zeta_{i} > 0$，ν可以调整训练集中可信样本的比例。

在参考文献[2]中提出的One Class SVM方法（可简称为SVDD）实质是在特征空间中获得数据周围的球形边界，这个超球体的体积是最小化的，从而最小化异常点的影响。产生的超球体中心为a、半径为R，体积$R^{2}$被最小化，中心a是支持向量的线性组合。与经典的SVM方法相似，要求每个数据点x_i到中心的距离严格小于R，但同时构造一个惩罚系数为C的松弛变量$\zeta_{i}$满足$\zeta_{i} > 0$，优化问题为

$$
{\min\limits_{R,a}R^{2}} + C{\sum\limits_{i = 1}^{n}\zeta_{i}} \quad s.t.\left\| {x_{i} - a} \right\|^{2} \leq R^{2} + \zeta_{i}, \quad i = 1,..,n

$$

*一类支持向量机模型相关的Python脚本在“OneClassSVM.py”中。*

## 参考文献

[1] Bernhard H Schölkopf, Robert C Williamson, Alexander Smola, John C Shawe-Taylor, John C Platt. Support vector method for novelty detection[C]. NIPS'99: Proceedings of the 12th International Conference on Neural Information Processing Systems, 1999.
[2] David M.J. Tax, Robert P.W. Duin. Support Vector Data Description[J]. Machine Learning, 2004, 54: 45-66.
[3] Markus M. Breunig, Hans-Peter Kriegel, Raymond Tak Yan Ng, Jörg Sander. LOF: identifying density-based local outliers[C]. Proc. ACM SIGMOD 2000 Int. Conf. On Management of Data, 2000.
[4] Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. Isolation Forest[C]. IEEE International Conference on Data Mining, 2008.
[5] 朱佳俊, 陈功, 施勇, 薛质. 基于用户画像的异常行为检测[J]. 通信技术, 2017, 50(10): 2310-2315.
[6] 崔景洋, 陈振国, 田立勤, 张光华. 基于机器学习的用户与实体行为分析技术综述[J/OL]. 计算机工程. https://doi.org/10.19678/j.issn.1000-3428.0062623.
[7] 爱丽丝·郑, 阿曼达·卡萨丽. 精通特征工程[M]. 北京: 人民邮电出版社, 2019.
