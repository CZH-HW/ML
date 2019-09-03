
'''
    MNIST（Mixed National Institute of Standards and Technology database）是一个入门级的计算机视觉数据集,其中包含各种手写数字图片
    MNIST数据集被分为三部分：55000个训练样本（mnist.train),5000个验证集（mnist.validation),10000个测试样本（mnist.test)
    x为一个大小为28*28的手写数字图片像素值矩阵转化为了行向量,标签y是第n维度的数字为1的10位维向量。例如，标签3用one—hot向量表示为[0,0,0,1,0,0,0,0,0,0]
'''


import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score # 混淆矩阵，准确率和召回率
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt



# 获取数据集并随机排序
mnist = fetch_mldata('MNIST original')     # 下载引入数据集MNIST
X, y = mnist["data"], mnist["target"]   
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]   # 创建训练集和测试集
shuffle_index = np.random.permutation(60000)     # 获取一个60000长度的随机排列的序列即一维数组
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]   # 根据随机排序的索引来获取训练打乱后的训练集


###################################################################################################################


# 训练一个二分类器，分类5和非5,采用随机梯度下降分类器 SGD
y_train_5 = (y_train == 5)    # 布尔值，y_train_5为一个布尔值序列
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)   # 训练SGD分类器

# 使用交叉验证测量准确性
skfolds = StratifiedKFold(n_splits=3, random_state=42)     # K折分层采样交叉切分，n_splits代表K，(K-1)+1
for train_index, test_index in skfolds.split(X_train, y_train_5):   
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]      
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
# 或者使用cross_val_score()函数来评估SGDClassifier模型，同时使用 K 折交叉验证
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")   # cv即为k，scoring="accuracy"表示得分为精度


# 精度通常来说不是一个好的性能度量指标，特别是当你处理有偏差的数据集，比方说其中一些类比其他类频繁得多
# 对分类器来说，一个好得多的性能评估指标是混淆矩阵。大体思路是：输出类别A被分类成类别 B 的次数 
# 准确率（precision）与 召回率（recall），并画出曲线
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)    # 返回基于每一个测试折做出的一个预测值
confusion_matrix(y_train_5, y_train_pred)   # 比较目标类和预测类，得到混淆矩阵，有几种标签就有相等的行列数
precision_score(y_train_5, y_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_pred)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")  # 返回一个决策分数
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)   # thresholds阈值
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# ROC曲线，召回率对（1减真反例率）
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()
roc_auc_score(y_train_5, y_scores)  # 测量ROC曲线下的面积（AUC）


# 采用随机森林分类，画出相应的ROC曲线和SGD做对比
# predict_proba()方法返回一个数组，数组的每一行代表一个样例，每一列代表一个类，数组当中的值的意思是：给定一个样例属于给定类的概率
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")  
y_scores_forest = y_probas_forest[:, 1]  # 将正例的概率提取出来
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()     

####################################################################################################################


# 多类分类器和正则化提高精度
# Scikit-Learn可以探测出你想使用一个二分类器去完成多分类的任务，它会自动地执行 OvA
sgd_clf.fit(X_train, y_train)   # 实际上训练了 10 个二分类器，每个分类器都产到一张图片的决策数值，选择数值最高的那个类。
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))  
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# 多标签分类
y_train_large = (y_train >= 7)    
y_train_odd = (y_train % 2 == 1)     # 是否是奇数
y_multilabel = np.c_[y_train_large, y_train_odd]    # 包含2个目标标签
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
