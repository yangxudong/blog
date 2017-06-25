---
title: Tree ensemble算法的特征重要度计算
date: 2016-12-27 13:58:55
categories: [机器学习,特征工程]
tags: [集成学习,GBDT,特征重要度]
mathjax: true
---
集成学习因具有预测精度高的优势而受到广泛关注，尤其是使用决策树作为基学习器的集成学习算法。树的集成算法的著名代码有随机森林和GBDT。随机森林具有很好的抵抗过拟合的特性，并且参数（决策树的个数）对预测性能的影响较小，调参比较容易，一般设置一个比较大的数。GBDT具有很优美的理论基础，一般而言性能更有优势。关于GBDT算法的原理请参考我的前一篇博文{% post_link gbdt 《GBDT算法原理深入解析》%}。

基于树的集成算法还有一个很好的特性，就是模型训练结束后可以输出模型所使用的特征的相对重要度，便于我们选择特征，理解哪些因素是对预测有关键影响，这在某些领域（如生物信息学、神经系统科学等）特别重要。本文主要介绍基于树的集成算法如何计算各特征的相对重要度。
<!-- more -->
##使用boosted tree作为学习算法的优势：

 - 使用不同类型的数据时，不需要做特征标准化/归一化
 - 可以很容易平衡运行时效率和精度；比如，使用boosted tree作为在线预测的模型可以在机器资源紧张的时候截断参与预测的树的数量从而提高预测效率
 - 学习模型可以输出特征的相对重要程度，可以作为一种特征选择的方法
 - 模型可解释性好
 - 对数据字段缺失不敏感
 - 能够自动做多组特征间的interaction，具有很好的非性线性

##特征重要度的计算

Friedman在GBM的论文中提出的方法：

特征$j$的全局重要度通过特征$j$在单颗树中的重要度的平均值来衡量：
$$\hat{J_{j}^2}=\frac1M \sum_{m=1}^M\hat{J_{j}^2}(T_m)$$
其中，M是树的数量。特征$j$在单颗树中的重要度的如下：
$$\hat{J_{j}^2}(T)=\sum\limits_{t=1}^{L-1} \hat{i_{t}^2} 1(v_{t}=j)$$
其中，$L$为树的叶子节点数量，$L-1$即为树的非叶子节点数量（构建的树都是具有左右孩子的二叉树），$v_{t}$是和节点$t$相关联的特征，$\hat{i_{t}^2}$是节点$t$分裂之后平方损失的减少值。

##实现代码片段

为了更好的理解特征重要度的计算方法，下面给出scikit-learn工具包中的实现，代码移除了一些不相关的部分。

下面的代码来自于GradientBoostingClassifier对象的feature_importances属性的计算方法：
```python
def feature_importances_(self):
    total_sum = np.zeros((self.n_features, ), dtype=np.float64)
    for tree in self.estimators_:
        total_sum += tree.feature_importances_ 
    importances = total_sum / len(self.estimators_)
    return importances
```
其中，self.estimators_是算法构建出的决策树的数组，tree.feature_importances_ 是单棵树的特征重要度向量，其计算方法如下：
```cython
cpdef compute_feature_importances(self, normalize=True):
    """Computes the importance of each feature (aka variable)."""

    while node != end_node:
        if node.left_child != _TREE_LEAF:
            # ... and node.right_child != _TREE_LEAF:
            left = &nodes[node.left_child]
            right = &nodes[node.right_child]

            importance_data[node.feature] += (
                node.weighted_n_node_samples * node.impurity -
                left.weighted_n_node_samples * left.impurity -
                right.weighted_n_node_samples * right.impurity)
        node += 1

    importances /= nodes[0].weighted_n_node_samples

    return importances
```
上面的代码经过了简化，保留了核心思想。计算所有的非叶子节点在分裂时加权不纯度的减少，减少得越多说明特征越重要。

不纯度的减少实际上就是该节点此次分裂的收益，因此我们也可以这样理解，节点分裂时收益越大，该节点对应的特征的重要度越高。关于收益的定义请参考我的前一篇博文[《GBDT算法原理深入解析》](https://www.zybuluo.com/yxd/note/611571)中的等式(9)的定义。

## 参考资料
[1] [Feature Selection for Ranking using Boosted Trees](https://pdfs.semanticscholar.org/156e/3c979e7bc25381fdd0614d1bab60b7aa5dfd.pdf)
[2] [Gradient Boosted Feature Selection](http://alicezheng.org/papers/gbfs.pdf)
[3] [Feature Selection with Ensembles, Artificial Variables, and Redundancy Elimination](http://www.jmlr.org/papers/volume10/tuv09a/tuv09a.pdf)
