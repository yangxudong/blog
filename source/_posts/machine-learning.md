---
title: 机器学习完整过程案例分步解析
date: 2014-05-24 15:18:34
tags: sklearn
categories: [机器学习]
---
所谓学习问题，是指观察由n个样本组成的集合，并根据这些数据来预测未知数据的性质。

## 学习任务（一个二分类问题）：

区分一个普通的互联网检索Query是否具有某个垂直领域的意图。假设现在有一个O2O领域的垂直搜索引擎，专门为用户提供团购、优惠券的检索；同时存在一个通用的搜索引擎，比如百度，通用搜索引擎希望能够识别出一个Query是否具有O2O检索意图，如果有则调用O2O垂直搜索引擎，获取结果作为通用搜索引擎的结果补充。

我们的目的是学习出一个分类器（classifier），分类器可以理解为一个函数，其输入为一个Query，输出为0（表示该Query不具有o2o意图）或1（表示该Query具有o2o意图）。

## 特征提取：

要完成这样一个学习任务，首先我们必须找出决定一个Query是否具有O2O意图的影响因素，这些影响因素称之为特征（feature）。特征的好坏很大程度上决定了分类器的效果。在机器学习领域我们都知道特征比模型（学习算法）更重要。（顺便说一下，工业界的人都是这么认为的，学术界的人可能不以为然，他们整天捣鼓算法，发出来的文章大部分都没法在实际中应用。）举个例子，如果我们的特征选得很好，可能我们用简单的规则就能判断出最终的结果，甚至不需要模型。比如，要判断一个人是男还是女（人类当然很好判断，一看就知道，这里我们假设由计算机来完成这个任务，计算机有很多传感器（摄像头、体重器等等）可以采集到各种数据），我们可以找到很多特征：身高、体重、皮肤颜色、头发长度等等。因为根据统计我们知道男人一般比女人重，比女人高，皮肤比女人黑，头发比女人短；所以这些特征都有一定的区分度，但是总有反例存在。我们用最好的算法可能准确率也达不到100%。假设计算机还能够读取人的身份证号码，那么我们可能获得一个更强的特征：身份证号码的倒数第二位是否是偶数。根据身份证编码规则，我们知道男性的身份证号码的倒数第二位是奇数，女生是偶数。因此，有了这个特征其他的特征都不需要了，而且我们的分类器也很简单，不需要复杂的算法。

言归正传，对于O2O Query意图识别这一学习任务，我们可以用的特征可能有：Query在垂直引擎里能够检索到的结果数量、Query在垂直引擎里能够检索到的结果的类目困惑度（perplexity）（检索结果的类目越集中说明其意图越强）、Query能否预测到特征的O2O商品类目、Query是否包含O2O产品词或品牌词、Query在垂直引擎的历史展现次数（PV)和点击率(ctr)、Query在垂直引擎的检索结果相关性等等。

<!-- more -->

## 特征表示：

特征表示是对特征提取结果的再加工，目的是增强特征的表示能力，防止模型（分类器）过于复杂和学习困难。比如对连续的特征值进行离散化，就是一种常用的方法。这里我们以“Query在垂直引擎里能够检索到的结果数量”这一特征为例，简要介绍一下特征值分段的过程。首先，分析一下这一维特征的分布情况，我们对这一维特征值的最小值、最大值、平均值、方差、中位数、三分位数、四分位数、某些特定值（比如零值）所占比例等等都要有一个大致的了解。获取这些值，Python编程语言的numpy模块有很多现成的函数可以调用。最好的办法就是可视化，借助python的matplotlib工具我们可以很容易地划出数据分布的直方图，从而判断出我们应该对特征值划多少个区间，每个区间的范围是怎样的。比如说我们要对“结果数量”这一维特征值除了“0”以为的其他值均匀地分为10个区间，即每个区间内的样本数大致相同。“0”是一个特殊的值，因此我们想把它分到一个单独的区间，这样我们一共有11个区间。python代码实现如下：

```python
import numpy as np

def bin(bins):
    assert isinstance(bins, (list, tuple))
    def scatter(x):
        if x == 0: return 0
        for i in range(len(bins)):
            if x <= bins[i]: return i + 1
        return len(bins)
    return np.frompyfunc(scatter, 1, 1)

data = np.loadtxt("D:\query_features.xls", dtype='int')
# descrete
o2o_result_num = data[:,0]
o2o_has_result = o2o_result_num[o2o_result_num > 0]
bins = [ np.percentile(o2o_has_result, x) for x in range(10, 101, 10) ]
data[:,0] = bin(bins)(o2o_result_num)
```
我们首先获取每个区间的起止范围，即分别算法特征向量的10个百分位数，并依此为基础算出新的特征值（通过bin函数，一个numpy的universal function）。

## 训练数据：

这里我们通过有监督学习的方法来拟合分类器模型。所谓有监督学习是指通过提供一批带有标注（学习的目标）的数据（称之为训练样本），学习器通过分析数据的规律尝试拟合出这些数据和学习目标间的函数，使得定义在训练集上的总体误差尽可能的小，从而利用学得的函数来预测未知数据的学习方法。注意这不是一个严格的定义，而是我根据自己的理解简化出来的。

一批带有标注的训练数据从何而来，一般而言都需要人工标注。我们从搜索引擎的日志里随机采集一批Query，并且保证这批Query能够覆盖到每维特征的每个取值（从这里也可以看出为什么要做特征分区间或离散化了，因为如不这样做我们就不能保证能够覆盖到每维特征的每个取值）。然后，通过人肉的方法给这边Query打上是否具有O2O意图的标签。数据标注是一个痛苦而漫长的过程，需要具有一定领域知识的人来干这样的活。标注质量的好坏很有可能会影响到学习到的模型（这里指分类器）在未知Query上判别效果的好坏。即正确的老师更可能教出正确的学生，反之，错误的老师教坏学生的可能性越大。在我自己标注数据的过程中，发现有一些Query的O2O意图比较模棱两可，导致我后来回头看的时候总觉得自己标得不对，反反复复修改了好几次。

## 选择模型：

在我们的问题中，模型就是要学习的分类器。有监督学习的分类器有很多，比如决策树、随机森林、逻辑回归、梯度提升、SVM等等。如何为我们的分类问题选择合适的机器学习算法呢？当然，如果我们真正关心准确率，那么最佳方法是测试各种不同的算法（同时还要确保对每个算法测试不同参数），然后通过交叉验证选择最好的一个。但是，如果你只是为你的问题寻找一个“足够好”的算法，或者一个起点，也是有一些还不错的一般准则的，比如如果训练集很小，那么高偏差/低方差分类器（如朴素贝叶斯分类器）要优于低偏差/高方差分类器（如k近邻分类器），因为后者容易过拟合。然而，随着训练集的增大，低偏差/高方差分类器将开始胜出（它们具有较低的渐近误差），因为高偏差分类器不足以提供准确的模型。

这里我们重点介绍一次完整的机器学习全过程，所以不花大篇幅在模型选择的问题上，推荐大家读一些这篇文章：《如何选择机器学习分类器？》。

## 通过交叉验证拟合模型：

机器学习会学习数据集的某些属性，并运用于新数据。这就是为什么习惯上会把数据分为两个集合，由此来评价算法的优劣。这两个集合，一个叫做训练集（train data），我们从中获得数据的性质；一个叫做测试集(test data)，我们在此测试这些性质，即模型的准确率。将一个算法作用于一个原始数据，我们不可能只做出随机的划分一次train和test data，然后得到一个准确率，就作为衡量这个算法好坏的标准。因为这样存在偶然性。我们必须好多次的随机的划分train data和test data，分别在其上面算出各自的准确率。这样就有一组准确率数据，根据这一组数据，就可以较好的准确的衡量算法的好坏。交叉验证就是一种在数据量有限的情况下的非常好evaluate performance的方法。

```python
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm

lr = linear_model.LogisticRegression()
lr_scores = cross_validation.cross_val_score(lr, train_data, train_target, cv=5)
print("logistic regression accuracy:")
print(lr_scores)

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
clf_scores = cross_validation.cross_val_score(clf, train_data, train_target, cv=5)
print("decision tree accuracy:")
print(clf_scores)

rfc = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=3, max_features=0.5, min_samples_split=5)
rfc_scores = cross_validation.cross_val_score(rfc, train_data, train_target, cv=5)
print("random forest accuracy:")
print(rfc_scores)

etc = ensemble.ExtraTreesClassifier(criterion='entropy', n_estimators=3, max_features=0.6, min_samples_split=5)
etc_scores = cross_validation.cross_val_score(etc, train_data, train_target, cv=5)
print("extra trees accuracy:")
print(etc_scores)

gbc = ensemble.GradientBoostingClassifier()
gbc_scores = cross_validation.cross_val_score(gbc, train_data, train_target, cv=5)
print("gradient boosting accuracy:")
print(gbc_scores)

svc = svm.SVC()
svc_scores = cross_validation.cross_val_score(svc, train_data, train_target, cv=5)
print("svm classifier accuracy:")
print(svc_scores)
```
上面的代码我们尝试同交叉验证的方法对比五种不同模型的准确率，结果如下：

```
logistic regression accuracy:
[ 0.76953125  0.83921569  0.85433071  0.81102362  0.83858268]
decision tree accuracy:
[ 0.73828125  0.8         0.77559055  0.71653543  0.83464567]
random forest accuracy:
[ 0.75        0.76862745  0.76377953  0.77165354  0.80314961]
extra trees accuracy:
[ 0.734375    0.78039216  0.7992126   0.76377953  0.79527559]
gradient boosting accuracy:
[ 0.7578125   0.81960784  0.83464567  0.80708661  0.84251969]
svm classifier accuracy:
[ 0.703125    0.78431373  0.77952756  0.77952756  0.80708661]
```
在O2O意图识别这个学习问题上，逻辑回归分类器具有最好的准确率，其次是梯度提升分类器；决策树和随机森林在我们的测试结果中并没有体现出明显的差异，可能是我们的特殊数量太少并且样本数也较少的原因；另外大名典典的SVM的表现却比较让人失望。总体而言，准确率只有82%左右，分析其原因，一方面我们实现的特征数量较少；另一方面暂时未能实现区分能力强的特征。后续会对此持续优化。

由于逻辑回归分类器具有最好的性能，我们决定用全部是可能训练数据来拟合之。
```python
lr = lr.fit(train_data, train_target)
```

## 模型数据持久化：

学到的模型要能够在将来利用起来，就必须把模型保存下来，以便下次使用。同时，数据离散化或数据分区的范围数据也要保存下来，在预测的时候同样也需要对特征进行区间划分。python提供了pickle模块用来序列号对象，并保存到硬盘上。同时，scikit-learn库也提供了更加高效的模型持久化模块，可以直接使用。

```python
from sklearn.externals import joblib
joblib.dump(lr, 'D:\lr.model')
import pickle
bin_file = open(r'D:\result_bin.data', 'wb')
pickle.dump(bins, bin_file)
bin_file.close()
```
##分类器的使用：

现在大功告成了，我们需要做的就是用学习到的分类器来判断一个新的Query到底是否具有O2O意图。因为我们分类器的输入是Query的特征向量，而不是Query本身，因此我们需要实现提取好Query的特征。假设我们已经离线算好了每个Query的特征，现在使用的时候只需要将其加载进内场即可。分类器使用的过程首先是从硬盘读取模型数据和Query特征，然后调用模型对Query进行预测，输出结果。

```python
# load result bin data and model
bin_file = open(r'D:\result_bin.data', 'rb')
bins = pickle.load(bin_file)
bin_file.close()

lr = joblib.load('D:\lr.model')

# load data
query = np.genfromtxt(r'D:\o2o_query_rec\all_query', dtype='U2', comments=None, converters={0: lambda x: str(x, 'utf-8')})
feature = np.loadtxt(r'D:\o2o_query_rec\all_features', dtype='int', delimiter='\001')

# descrite
feature[:,0] = bin(bins)(feature[:,0])
feature[:,1] = ufunc_segment(feature[:,1])

# predict
result = lr.predict(feature)

# save result
#np.savetxt(r'D:\o2o_query_rec\classify_result.txt', np.c_[query, result], fmt=u"%s", delimiter="\t")
result_file = open(r'D:\o2o_query_rec\classify_result.txt', 'w')
i = 0
for q in query:
    result_file.write('%s\t%d\n' % (q, result[i]))
    i += 1
result_file.close()
```
需要注意的是我们Query的编码是UTF-8，load的时候需要做相应的转换。另外，在python 3.3版本，numpy的savetxt函数并不能正确保持UTF-8格式的中文Query（第20行注释掉的代码输出的Query都变成了bytes格式的），如果小伙伴们有更好的办法能够解决这个问题，请告诉我，谢谢！
