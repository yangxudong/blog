---
title: 用hadoop实现SimRank++算法(3)---- evidence矩阵的计算及性能优化总结
date: 2013-03-28 22:21:50
tags: [SimRank,SimRank++,相似性度量]
categories: [数据挖掘]
mathjax: true
---
# 背景

本文主要针对广告检索领域的查询重写应用，根据查询-广告点击二部图，在MapReduce框架上实现SimRank++算法，关于SimRank++算法的背景和原理请参看前一篇文章《{% post_link simrank-plus-plus SimRank++算法原理深入解析 %}》。关于权值转移概率矩阵的实现请参看另一位文章《{% post_link simrank-weight-matrix 用hadoop实现SimRank++算法(1)----权值转移矩阵的计算 %}》。关于算法的迭代计算过程请参考《{% post_link simrank-iteration 用hadoop实现SimRank++算法(2)---- 算法迭代过程 %}》。
<!--more-->

# Evidence矩阵的计算

证据矩阵用在算法的最后一步，用来修正算法在之前的步骤中计算出来的相似性分数，使之更精确合理。由于我们的目的是求Query之间的相似度，所以这里只给出Query和Query之间的证据矩阵计算过程。

证据矩阵每个元素的计算公式请参考《{% post_link simrank-plus-plus SimRank++算法原理深入解析 %}》。

计算证据矩阵的MapReduce作业的输入数据文件即《{% post_link simrank-weight-matrix 用hadoop实现SimRank++算法(1)----权值转移矩阵的计算 %}》提到的aqs文件，其每一行的数据格式为：``aqs ^A aid {^A qid ^B click_num}``。该作业的程序逻辑相对比较简单。

“Map”函数接受aqs文件的输入，输出有共同点击广告的两两Query。由于证据矩阵是一个对称矩阵，因此我们只计算出其上三角矩阵。程序的伪代码如下所示。
```
Map (line_no, line_txt) {
    content ← Parser(line_txt)
    queries ← content.queries
    queries ← sort(queries)
    length ← length(queries)
    for i in 0 : (length-1)
    	for j in (i+1) : length
    		emit <queries[i], queries[j]>, 1
}
```
为了减少“Shuffle” 和 “Sort”阶段的数据传输量，我们设计了一个“Combiner”函数来合并本地的“Map”输出结果，其逻辑非常简单，把相同键的值求和即可。“Combiner”函数的程序伪代码如下：
```
Reduce(key, valueList) {
    Emit key, sum(valueList)
}
```
“Reduce”函数把相同键的所有值求和，结果就是该键对应的两个Query共同点击的广告个数。假设键为<q1, q2>，则“Reduce”函数首先对相应的值列表中的所有元素求和，结果为 。然后“Reduce”函数根据计算出的 值求出$evidence(q1, q2)$。考虑到实际应用中会有大量的 值重复，为了提高程序的运行效率，可以使用缓存把计算结果保存下来，这样可以避免重复计算Evidence值。具体过程为，当新的 值求出来之后，首先到缓存里查找对应的Evidence值，若查找到相应的项则返回该项的值，否则调用计算Evidence的函数，计算出新的Evidence值并保存在缓存里。Reduce函数的另一个重要的工作是保证输出一个完整的证据矩阵，根据对称性，需要把同一个值输出两次，不过对应的两个Key的元素位置正好相反，即调换目标矩阵的行号和列号。

计算证据矩阵的“Reduce”函数的伪代码如表下：
```
Reduce(key, valueList) {
    size ← sum(valueList)
    if (cache.containKey(size))
    	value ← cache.get(size)
    else
    	value ← computeEvidence(size)
    	cache.add(size, value)
    emit key, value
    emit <key.index2, key.index1>, value
}
```

# 性能优化总结

在实际应用中，数据的规模比较庞大，因而必须精简数据结构和算法流程。考虑到算法的效率和中间输出结果的规模，本章在实现SimRank++算法时采用了以下几项优化技术：

(1)	阈值过滤

如果点击关系二部图的边的权值比较小，说明对应的Query和广告相关性不高，因此可以考虑在计算权值矩阵时，过滤掉点击次数低于某个阈值的所有权值。另外，在SimRank算法的每一轮迭代结果中过滤掉低于某个阈值的相似性分数，可以大大减少以后迭代过程中的计算工作量。因为相似性分数过低，说明相应的对象之间的相关性不高，因而在实际应用中的作用几乎可以忽略。我们把权值的默认阈值设为3，把相似度分数的默认阈值设为0.0001。

(2)	自适应数据分片大小

根据本文所采用的矩阵乘法方法，矩阵乘法作业会在“Mapper”任务输出大量的中间键值对，其数据规模是“Mapper”任务输入数据量的若干倍。大量的输出，会导致“Mapper”任务需要不断地Spilt数据到磁盘上，因而任务运行时间较长，性能较低。为了减少这种现象，可以自定义输入分片的大小，而不是采用默认值。本章通过自定义的SizeCustomizeSequenceFileInputFormat类来实现所需的功能。

由于不同的矩阵乘法策略所带来的“Map”任务输出数量的膨胀程度不同，在某些策略下“Map”任务的输出会发生数据倾斜，因此可以根据“Mapper”任务处理的具体数据来自适应数据分片的大小。例如，根据矩阵乘法的策略3，A矩阵的数据会膨胀若干倍，而B矩阵的数据不会膨胀，因此，我们使用MultipleInputs类来为不同的输入定义不同的InputFormat，从而控制不同输入的不同数据分片大小。通过这种方法，每个“Mapper”任务的工作负载比较均衡，整个作业”Map”阶段所需的总时间大大降低。

(3)	Inplace技术

由计算公式可知，在第K轮迭代时，计算(Q-Q)[k]时需要用到(A-A)[k-1]，计算(A-A)[k]时需要用到(Q-Q)[k-1]。假设在第K轮迭代时已经计算出了(A-A)[k]，那么在计算(Q-Q)[k]时就可以用本轮迭代计算出的(A-A)[k]，而不是上一轮计算出来的(A-A)[k-1]，因为(A-A)[k]总是比(A-A)[k-1]更精确。这种方法在机器学习领域叫做Inplace技术，其有两个显著的优点：

- 能够加快算法的收敛速度。因为在迭代过程中，总是尽可能早地利用已经计算出来的结果，不去使用过期数据。
- 能够减少所需要的存储开销。当本轮迭代计算出的(A-A)[k]数据可以覆盖掉(A-A)[k-1]的数据，因为其已经过期且不会再次被使用。如果不采用Inplace技术，那么在计算(Q-Q)k时就必须同时保留(A-A)[k]数据和(A-A)[k-1]的数据。

(4)	压缩中间结果

根据本文所采用的矩阵乘法方法，矩阵乘法作业会在“Mapper”任务输出大量的中间结果，导致大量的存储开销和网络带宽，导致算法的性能较低。为了较少性能瓶颈，可以使用压缩技术对中间结果进行压缩。

Hadoop框架提供了一系列的接口用来对”Map”端或”Reduce”端的输出数据进行压缩，以减少所需的HDFS空间和节点间的网络带宽。可以根据实际情况选择不同的压缩算法，权衡好压缩率和压缩时间开销，以便实现性能的最大提升。我们在实现SimRank++算法时，在不同的作业中，根据实际情况选择不同的压缩方法。基本的设置压缩算法的语句如下：
```
job.setBoolean("mapred.output.compress", true);
job.setClass("mapred.output.compression.codec", BZip2Codec.class, CompressionCodec.class);
SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
SequenceFileOutputFormat.setOutputCompressorClass(job, BZip2Codec.class);
```

(5)	存储优化

分块子矩阵的行列索引用short，不要用integer，以便节省空间。利用相似性矩阵的对称性，知道输出结果为相似性矩阵时，只计算和存储下三角矩阵，避免计算和存储全部矩阵。

# 总结

系统文章详细描述了在Hadoop MapReduce上实现SimRank++算法的细节，包括权值矩阵和证据矩阵的计算、算法迭代过程、相似度值的计算。同时给出了算法的一些可行的性能优化方法。在算法实现过程中，主要的创新性工作列举如下：

+ (1) 矩阵的转置操作和衰减因子c的乘法操作以及结果矩阵对角线元素的重置操作内嵌到矩阵乘法作业中。
+ (2) 修改SimRank计算公式，通过矩阵转置的等价变换，省去了对权值矩阵的转置矩阵的存储。
+ (3) 拆分SimRank计算公式为两个部分，使得计算规模大大降低。
+ (4) 采用了阈值过滤、自适应数据分片大小、Inplace技术、压缩中间结果等性能优化方法。

# 系列文章

本系列的文章写道这里就要告一段落了，关于MapReduce上的矩阵乘法的实现有机会再和大家一起探讨。有兴趣的读者可以看看这篇文章《[A MapReduce Algorithm for Matrix Multiplication](http://www.norstad.org/matrix-multiply/)》。

- {% post_link simrank-plus-plus SimRank++算法原理深入解析 %}
- {% post_link simrank-weight-matrix 用hadoop实现SimRank++算法(1)----权值转移矩阵的计算 %}
- {% post_link simrank-iteration 用hadoop实现SimRank++算法(2)---- 算法迭代过程 %}
- {% post_link simrank-evidence 用hadoop实现SimRank++算法(3)---- evidence矩阵的计算及性能优化总结 %}
