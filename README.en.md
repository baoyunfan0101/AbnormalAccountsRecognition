# Abnormal Accounts Recognition

<div align="right">
	[<a href="README.md">中文</a> | English(Current)</a>]
</div>

## Related Project

[Certification Risk Prediction](https://github.com/baoyunfan0101/CertificationRiskPrediction)

## File Description

datasets // datasets (training set, test set)  
feature engineering // feature engineering  
models // risk models  
references // references

## Test Environment

Python3.8 & MATLAB R2018a

## Task Description

From the basic information, operation information, and transaction information provided in the dataset, extract effective features to build an account feature model—that is, a model describing the relationship between account features and account risk labels—thereby achieving risk account identification.

## Feature Engineering

### Data Preprocessing

The raw data has no duplicate records. Most attribute distributions are reasonable, with no records conflicting with common sense. Some attributes have too many missing values (e.g., service level 3 `service3_level`), which will be removed during feature selection. Some attributes have few missing values, which will be filled with 0 (the mean) after feature extraction.

In addition, categorical attributes in the raw data are stored as string codes. For discrete attributes with ordinal relationships, we convert the categories to numerical codes. For those with nominal relationships, we apply one-hot encoding.

Because feature scales are inconsistent and there are outliers outside the normal range, data standardization is required. Here we perform z-score standardization using the mean and standard deviation of the original data to meet the needs of subsequent model training. The formula is

$$
{X'}_{i} = \frac{X_{i} - {\overset{-}{X}}_{i}}{S}
$$

where
${X'}_{i}$
is the standardized feature;
$X_{i}$
is the original feature;
${\overset{-}{X}}_{i}$
is the mean of the original feature; and \(S\) is the standard deviation of the original feature, computed as
\(\sqrt{\frac{\sum\limits_{i = 1}^{n}\left( {x_{i} - \overset{-}{x}} \right)^{2}}{n - 1}}\).
Because further feature derivation may follow, in practice standardization can be performed after completing feature engineering.

*The Python scripts for preprocessing the training and test sets are in `preprocessing_train.py` and `preprocessing_test.py`.*

### Feature Derivation and Selection

Operations and transaction information of the same account are naturally the focus of an account feature model, and their timing information (attribute `tm_diff`) is the most critical part of modeling. Inspired by the RFM analysis method, we derive features from the relevant timing information.

RFM stands for Recency (time since the last transaction), Frequency (transaction frequency), and Monetary (transaction amount). Based on this idea, we extract four features from operation information: latest operation time `op_recent_tm`, operation frequency `op_frequency`, average operation interval `op_interval`, and minimum operation interval `op_min_interval`; and five features from transaction information: latest transaction time `trans_recent_tm`, transaction frequency `trans_frequency`, trans_amount, average transaction interval `trans_interval`, and minimum transaction interval `trans_min_interval`.

There is a particular reason to keep both the average and minimum intervals. On the one hand, from a professional perspective, the minimum interval of operations/transactions is an important standard for determining whether an account is manually operated, which is valuable for abnormal account detection. On the other hand, the average interval only depends on the earliest and latest operations/transactions of an account. Adding the minimum interval enables more effective use of data and better reflects the Frequency dimension of RFM.

During feature selection, in addition to removing attributes with too many missing values as mentioned in “Data Preprocessing,” further selection was made based on the following feature analysis results, explained in detail below.

*Feature derivation for training and test sets is in `preprocessing_train.py` and `preprocessing_test.py`, performed together with preprocessing. Feature selection scripts for training and test sets are in `screening_train.py` and `screening_test.py`.*

### Feature Analysis

#### Feature Importance Evaluation

**WOE** (Weight of Evidence) encodes an original independent variable. After grouping/discretizing a feature with respect to an evaluation criterion, WOE is computed as

$$
{WOE}_{i} = ln\left( \frac{{py}_{i}}{{pn}_{i}} \right) = ln\left( \frac{\frac{y_{i}}{y_{T}}}{\frac{n_{i}}{n_{T}}} \right)
$$

where
${WOE}_{i}$
is the WOE of group \(i\);
${py}_{i}$
is the proportion of responding customers (risky accounts) in group \(i\);
${pn}_{i}$
is the proportion of non-responding customers in group \(i\);
$y_{i}$
is the number of responding customers in group \(i\);
$y_{T}$
is the total number of responding customers;
$n_{i}$
is the number of non-responding customers in group \(i\);
$n_{T}$
is the total number of non-responding customers.

**IV** (Information Value) takes into account both WOE and group proportions. It can be seen as a weighted sum of WOE and reflects a feature’s contribution to account risk. The IV of group \(i\) is computed as

$$
{IV}_{i} = \left( {py}_{i} - {pn}_{i} \right) \times {WOE}_{i} = \left( \frac{y_{i}}{y_{T}} - \frac{n_{i}}{n_{T}} \right) \times ln\left( \frac{\frac{y_{i}}{y_{T}}}{\frac{n_{i}}{n_{T}}} \right)
$$

and the IV of a feature is

$$
IV = {\sum\limits_{i = 1}^{n}{IV}_{i}}
$$

where \(n\) is the number of groups.

After preprocessing and feature derivation, except for boolean features with only 0/1 values, we applied chi-square binning to each feature, dividing the data into 5 groups, then calculated IV. The results are shown below.

![image](https://github.com/baoyunfan0101/AbnormalAccountsRecognition/blob/main/static/iv.jpg)

It can be seen that most features have reasonable correlation with account risk `label`. Features with very small IV values can be discarded.

*The Python scripts for feature importance evaluation are in `features.py` and `iv.m`.*

#### Feature Correlation Analysis

We compute the correlation matrix of the evaluation metrics and plot a heatmap, as shown below.

![image](https://github.com/baoyunfan0101/AbnormalAccountsRecognition/blob/main/static/correlation.png)

From the heatmap, most features have correlations within reasonable ranges. For highly correlated features, feature merging may be applied.

*The Python script for correlation analysis is also in `features.py`.*

## Model Training and Optimization

### Logistic Regression

**Logistic Regression (LR)** is a generalized linear regression model commonly used for binary classification.

In the account risk model, let the dependent variable account risk label be \(y \in \{0,1\}\). This is a binary classification problem. If under independent variables \(x=X\), the probability that \(y=1\) is \(p\), written as
\(p = P\left( y = 1 \middle| X \right)\),
then the probability of \(y=0\) is
\(1 - p\).
The odds ratio is
\(\frac{p}{1 - p}\),
and its natural logarithm yields the Sigmoid function

$$
Sigmoid(p) = ln\left( \frac{p}{1 - p} \right)
$$

Let
\(Sigmoid(p) = z\),
then

$$
p = \frac{1}{1 + e^{- z}}
$$

Let the feature vector be \(X\) and coefficient vector \(\beta\). Substituting yields

$$
h(x) = \frac{1}{1 + e^{- X\beta^{T}}}
$$

Here \(h(x)\in[0,1]\), serving as the predicted account risk label. If \(h(x)\ge 0.5\), set \(y=1\); otherwise \(y=0\).

*The Python script for logistic regression is in `LR.py`.*

### Support Vector Machine

A **Support Vector Machine (SVM)** is a generalized linear classifier that performs binary classification via supervised learning and works well for small samples and complex models.

SVM separates classes with a maximum-margin hyperplane. Viewing features as points in high-dimensional space, we find the hyperplane maximizing the margin between two classes, written as
\(wx + b = 0\).
The closest points to the hyperplane are support vectors, denoted
\(z_{0}\).
The distance from a point \(x\) to the hyperplane is

$$
d = \frac{\left| {wx + b} \right|}{\left\| w \right\|}
$$

where
\(\left\| w \right\|\)
is the 2-norm of \(w\),
\(\left\| w \right\| = \sqrt{\sum\limits_{i}w_{i}^{2}}\).
By definition of support vectors,

$$
\frac{\left| {wx + b} \right|}{\left\| w \right\|} \geq \frac{\left| {wz_{0} + b} \right|}{\left\| w \right\|} = d_{0}
$$

which simplifies to

$$
\left| \frac{wx + b}{\left\| w \right\| d_{0}} \right| \geq 1
$$

For convenience, since
\(\left\| w \right\| d_{0} > 0\),
let it equal 1:

$$
\left| {wx + b} \right| \geq 1
$$

To maximize \(d_{0}\), maximize
\(\frac{1}{\left\| w \right\|}\).
Thus the SVM model is

$$
\max\limits_{}\frac{1}{\left\| w \right\|} \quad s.t.\left| {wx + b} \right| \geq 1
$$

*The Python script for the SVM model is in `SVM.py`.*

### XGBoost

**XGBoost** (eXtreme Gradient Boosting, XGB) is a gradient boosting decision tree (GBDT) method composed of an ensemble of CART trees, which performs efficiently and accurately in many tasks.

XGBoost uses forward additive modeling with \(K\) trees:

$$
{\hat{y}}_{i} = {\sum\limits_{k = 1}^{K}{f_{k}\left( x_{i} \right)}}, \quad f \in F
$$

where
\(f_{k}\)
is the \(k\)-th regression tree and \(F\) is the tree function space. Its objective is

$$
Obj(\Theta) = {\sum\limits_{i = 1}^{N}{l\left( {y_{i},{\hat{y}}_{i}} \right)}} + {\sum\limits_{j = 1}^{t}{\Omega\left( f_{j} \right)}}, \quad f_{j} \in F
$$

where \(l\) is the loss and \(\Omega\) is the regularization term related to complexity. Regularization prevents overfitting.

*The Python script for the XGBoost model is in `XGB.py`.*

### CatBoost

**CatBoost** (from “Category” and “Boosting”) is a machine learning library open-sourced by Yandex in 2017. It is one of the three mainstream GBDT tools, along with XGBoost and LightGBM.

Compared with XGBoost, CatBoost has the following characteristics:

- Converts categorical features into numerical form with innovative algorithms;  
- Uses combinations of categorical features to enrich feature space;  
- Uses ordered boosting to combat noise, avoiding bias in gradient estimation and solving prediction shift;  
- Uses fully symmetric trees as base learners.

This model performs best among standalone models for this problem.

*The Python script for the CatBoost model is in `CatBoost.py`.*

### One-Class Support Vector Machine

Tests show the above models do not perform well enough on this problem.

One reasonable hypothesis is that normal behaviors are similar while abnormal behaviors vary (or the inverse), making it difficult for supervised learning to distinguish them.

Another hypothesis is that the models are only suited for **outlier detection** (anomalies present in the training set) but not **novelty detection** (new types of anomalies unseen in training).

Therefore, we split the processed data into normal and abnormal behavior sets according to the account risk `label`, and introduce the following model.

The **One-Class Support Vector Machine (One-Class SVM)** is a typical single-class model, commonly used for novelty detection.

The training set of a One-Class SVM should contain only one type of behavior. Two common formulations are as follows.

The OCSVM method in reference [1] separates all data points from the origin in feature space \(F\) with a hyperplane and maximizes the hyperplane’s distance to the origin. Its optimization objective is

$$
{\min\limits_{w,\zeta_{i},\rho}{\frac{1}{2}\left\| w \right\|^{2}}} + \frac{1}{\nu n}{\sum\limits_{i = 1}^{n}\zeta_{i}} - \rho \quad s.t.\left( {w^{T}\phi\left( x_{i} \right)} \right) > \rho - \zeta_{i}, \quad i = 1,..,n
$$

where
\(\zeta_{i}\)
are slack variables with
\(\zeta_{i} > 0\),
and \(\nu\) adjusts the proportion of trusted samples.

The SVDD method in reference [2] instead obtains a spherical boundary around the data, minimizing its volume to reduce the effect of anomalies. The hypersphere has center \(a\) and radius \(R\). The optimization is

$$
{\min\limits_{R,a}R^{2}} + C{\sum\limits_{i = 1}^{n}\zeta_{i}} \quad s.t.\left\| {x_{i} - a} \right\|^{2} \leq R^{2} + \zeta_{i}, \quad i = 1,..,n
$$

*The Python script for the One-Class SVM model is in `OneClassSVM.py`.*

## References

[1] Bernhard H Schölkopf, Robert C Williamson, Alexander Smola, John C Shawe-Taylor, John C Platt. Support vector method for novelty detection[C]. NIPS'99: Proceedings of the 12th International Conference on Neural Information Processing Systems, 1999.  
[2] David M.J. Tax, Robert P.W. Duin. Support Vector Data Description[J]. Machine Learning, 2004, 54: 45-66.  
[3] Markus M. Breunig, Hans-Peter Kriegel, Raymond Tak Yan Ng, Jörg Sander. LOF: identifying density-based local outliers[C]. Proc. ACM SIGMOD 2000 Int. Conf. On Management of Data, 2000.  
[4] Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. Isolation Forest[C]. IEEE International Conference on Data Mining, 2008.  
[5] Jiajun Zhu, Gong Chen, Yong Shi, Zhi Xue. Abnormal Behavior Detection Based on User Profiling[J]. Communications Technology, 2017, 50(10): 2310-2315.  
[6] Jingyang Cui, Zhenguo Chen, Liqin Tian, Guanghua Zhang. A Survey of User and Entity Behavior Analytics Based on Machine Learning[J/OL]. Computer Engineering. https://doi.org/10.19678/j.issn.1000-3428.0062623.  
[7] Alice Zheng, Amanda Casari. Mastering Feature Engineering[M]. Beijing: Posts & Telecom Press, 2019.