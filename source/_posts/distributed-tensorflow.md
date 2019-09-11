---
title: 一文说清楚Tensorflow分布式训练必备知识
date: 2019-03-22 11:18:29
tags: [tensorflow, deep learning, 深度学习]
categories: [机器学习, 深度学习]
---

> Methods that scale with computation are the future of AI.
—Rich Sutton, 强化学习之父

大数据时代的互联网应用产生了大量的数据，这些数据就好比是石油，里面蕴含了大量知识等待被挖掘。深度学习就是挖掘数据中隐藏知识的利器，在许多领域都取得了非常成功的应用。然而，大量的数据使得模型的训练变得复杂，使用多台设备分布式训练成了必备的选择。
 
Tensorflow是目前比较流行的深度学习框架，本文着重介绍tensorflow框架是如何支持分布式训练的。
<!--more--> 
## 分布式训练策略
 
### 模型并行

所谓模型并行指的是将模型部署到很多设备上（设备可能分布在不同机器上，下同）运行，比如多个机器的GPUs。当神经网络模型很大时，由于显存限制，它是难以完整地跑在单个GPU上，这个时候就需要把模型分割成更小的部分，不同部分跑在不同的设备上，例如将网络不同的层运行在不同的设备上。

由于模型分割开的各个部分之间有相互依赖关系，因此计算效率不高。所以在模型大小不算太大的情况下一般不使用模型并行。

在tensorflow的术语中，模型并行称之为"in-graph replication"。

### 数据并行

数据并行在多个设备上放置相同的模型，各个设备采用不同的训练样本对模型训练。每个Worker拥有模型的完整副本并且进行各自单独的训练。

![](https://d3ansictanv2wj.cloudfront.net/figure1-1cd2c0441cf54f2237e3d8720180cb45.png)

相比较模型并行，数据并行方式能够支持更大的训练规模，提供更好的扩展性，因此数据并行是深度学习最常采用的分布式训练策略。

在tensorflow的术语中，数据并行称之为"between-graph replication"。
 
## 分布式并行模式

深度学习模型的训练是一个迭代的过程。在每一轮迭代中，前向传播算法会根据当前参数的取值计算出在一小部分训练数据上的预测值，然后反向传播算法再根据损失函数计算参数的梯度并更新参数。在并行化地训练深度学习模型时，不同设备（GPU或CPU）可以在不同训练数据上运行这个迭代的过程，而不同并行模式的区别在于不同的参数更新方式。

深度学习模型训练流程图
 ![深度学习模型训练流程图](https://user-gold-cdn.xitu.io/2017/4/10/7eda6fce5eec9cbce366426fcecbe56f?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)
 
数据并行可以是同步的（synchronous），也可以是异步的（asynchronous）。

### 异步训练

异步训练中，各个设备完成一个mini-batch训练之后，不需要等待其它节点，直接去更新模型的参数。从下图中可以看到，在每一轮迭代时，不同设备会读取参数最新的取值，但因为不同设备读取参数取值的时间不一样，所以得到的值也有可能不一样。根据当前参数的取值和随机获取的一小部分训练数据，不同设备各自运行反向传播的过程并独立地更新参数。可以简单地认为异步模式就是单机模式复制了多份，每一份使用不同的训练数据进行训练。

![异步模式深度学习模型训练流程图](https://user-gold-cdn.xitu.io/2017/4/10/0b3dc701f60001d2454b268469245cb8?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

异步训练总体会训练速度会快很多，但是异步训练的一个很严重的问题是梯度失效问题（stale gradients），刚开始所有设备采用相同的参数来训练，但是异步情况下，某个设备完成一步训练后，可能发现模型参数已经被其它设备更新过了，此时这个设备计算出的梯度就过期了。由于梯度失效问题，异步训练可能陷入次优解（sub-optimal training performance）。图10-3中给出了一个具体的样例来说明异步模式的问题。其中黑色曲线展示了模型的损失函数，黑色小球表示了在t0时刻参数所对应的损失函数的大小。假设两个设备d0和d1在时间t0同时读取了参数的取值，那么设备d0和d1计算出来的梯度都会将小黑球向左移动。假设在时间t1设备d0已经完成了反向传播的计算并更新了参数，修改后的参数处于图10-3中小灰球的位置。然而这时的设备d1并不知道参数已经被更新了，所以在时间t2时，设备d1会继续将小球向左移动，使得小球的位置达到图10-3中小白球的地方。从图10-3中可以看到，当参数被调整到小白球的位置时，将无法达到最优点。

![](https://user-gold-cdn.xitu.io/2017/4/10/90617a718625f2f6005e6627dc2c1837?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

在tensorflow中异步训练是默认的并行训练模式。

### 同步训练

所谓同步指的是所有的设备都是采用相同的模型参数来训练，等待所有设备的mini-batch训练完成后，收集它们的梯度后执行模型的一次参数更新。在同步模式下，所有的设备同时读取参数的取值，并且当反向传播算法完成之后同步更新参数的取值。单个设备不会单独对参数进行更新，而会等待所有设备都完成反向传播之后再统一更新参数 。

![同步模式深度学习模型训练流程图](https://user-gold-cdn.xitu.io/2017/4/10/0d89e28d77235fa27c522b9d03c940ba?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

同步模式相当于通过聚合多个设备上的mini-batch形成一个更大的batch来训练模型，**相对于异步模式，在同步模型下根据并行的worker数量线性增加学习速率会取得不错的效果**。如果使用tensorflow estimator接口来分布式训练模型的话，在同步模式下需要适当减少训练步数（相对于采用异步模式来说），否则需要花费较长的训练时间。Tensorflow estimator接口唯一支持的停止训练的条件就全局训练步数达到指定的max_steps。

Tensorflow提供了[tf.train.SyncReplicasOptimizer](https://www.tensorflow.org/versions/master/api_docs/python/tf/train/SyncReplicasOptimizer)类用于执行同步训练。通过使用SyncReplicasOptimzer，你可以很方便的构造一个同步训练的分布式任务。把异步训练改造成同步训练只需要两步：

1. 在原来的Optimizer上封装SyncReplicasOptimizer，将参数更新改为同步模式；
`optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=num_workers)`
2. 在MonitoredTrainingSession或者EstimatorSpec的hook中增加sync_replicas_hook：
`sync_replicas_hook = optimizer.make_session_run_hook(is_chief, num_tokens=0)`

### 小结

下图可以一目了然地看出同步训练与异步训练之间的区别。
![](https://d3ansictanv2wj.cloudfront.net/figure2-f3599b8db486355f7427b3bb860692c3.png)

同步训练看起来很不错，但是实际上需要各个设备的计算能力要均衡，而且要求集群的通信也要均衡，类似于木桶效应，一个拖油瓶会严重拖慢训练进度，所以同步训练方式相对来说训练速度会慢一些。

虽然异步模式理论上存在缺陷，但因为训练深度学习模型时使用的随机梯度下降本身就是梯度下降的一个近似解法，而且即使是梯度下降也无法保证达到全局最优值。在实际应用中，在相同时间内使用异步模式训练的模型不一定比同步模式差。所以这两种训练模式在实践中都有非常广泛的应用。

## 分布式训练架构

### Parameter Server架构

Parameter server架构（PS架构）是深度学习最常采用的分布式训练架构。在PS架构中，集群中的节点被分为两类：parameter server和worker。其中parameter server存放模型的参数，而worker负责计算参数的梯度。在每个迭代过程，worker从parameter sever中获得参数，然后将计算的梯度返回给parameter server，parameter server聚合从worker传回的梯度，然后更新参数，并将新的参数广播给worker。

![](https://gw.alipayobjects.com/zos/skylark/a242d040-441b-4bb2-af22-2d40f95102cf/2018/png/c3d9d865-211f-4880-98c2-a7505ccb6a1d.png)

### Ring AllReduce架构

PS架构中，当worker数量较多时，ps节点的网络带宽将成为系统的瓶颈。

Ring AllReduce架构中各个设备都是worker，没有中心节点来聚合所有worker计算的梯度。Ring AllReduce算法将 device 放置在一个逻辑环路（logical ring）中。每个 device 从上行的device 接收数据，并向下行的 deivce 发送数据，因此可以充分利用每个 device 的上下行带宽。

![Ring-allreduce architecture for synchronous stochastic gradient descent](https://d3ansictanv2wj.cloudfront.net/figure4-7564694e76d08e091ce453f681515e59.png)

使用 Ring Allreduce 算法进行某个稠密梯度的平均值的基本过程如下：

1. 将每个设备上的梯度 tensor 切分成长度大致相等的 num_devices 个分片；
2. ScatterReduce 阶段：通过 num_devices - 1 轮通信和相加，在每个 device 上都计算出一个 tensor 分片的和；
3. AllGather 阶段：通过 num_devices - 1 轮通信和覆盖，将上个阶段计算出的每个 tensor 分片的和广播到其他 device；
4. 在每个设备上合并分片，得到梯度和，然后除以 num_devices，得到平均梯度；

以 4 个 device上的梯度求和过程为例：

ScatterReduce 阶段：
![](https://private-alipayobjects.alipay.com/alipay-rmsdeploy-image/skylark/png/1869bb5b-29ce-4ca8-9461-e23ad3a7bb45.png)
经过 num_devices - 1 轮后，每个 device 上都有一个 tensor 分片进得到了这个分片各个 device 上的和；

AllGather 阶段：
![](https://private-alipayobjects.alipay.com/alipay-rmsdeploy-image/skylark/png/5f715f8d-9eed-481f-a969-13c78d50529c.png)

经过 num_devices - 1 轮后，每个 device 上都每个 tensor 分片都得到了这个分片各个 device 上的和；

由上例可以看出，通信数据量的上限不会随分布式规模变大而变大一次 Ring Allreduce 中总的通信数据量是：
$$ 2 \cdot \frac{num\_devices - 1}{num\_devices} \cdot {tensor\_size} \approx 2 \cdot tensor\_size $$

相比PS架构，Ring Allreduce架构是带宽优化的，因为集群中每个节点的带宽都被充分利用。此外，在深度学习训练过程中，计算梯度采用BP算法，其特点是后面层的梯度先被计算，而前面层的梯度慢于前面层，Ring-allreduce架构可以充分利用这个特点，在前面层梯度计算的同时进行后面层梯度的传递，从而进一步减少训练时间。Ring Allreduce的训练速度基本上线性正比于GPUs数目（worker数）。

2017年2月百度在PaddlePaddle平台上首次引入了[ring-allreduce](https://github.com/baidu-research/baidu-allreduce)的架构，随后将其提交到tensorflow的contrib package中。同年8月，Uber为tensorflow平台开源了一个更加易用和高效的ring allreduce分布式训练库[Horovod](https://github.com/uber/horovod)。最后，tensorflow官方终于也在1.11版本中支持了allreduce的分布式训练策略[CollectiveAllReduceStrategy](https://github.com/logicalclocks/hops-examples/tree/master/tensorflow/notebooks/Distributed_Training/collective_allreduce_strategy)，其跟estimator配合使用非常方便，只需要构造`tf.estimator.RunConfig`对象时传入CollectiveAllReduceStrategy参数即可。

![](https://hopshadoop47880376.files.wordpress.com/2018/10/null9.png)

## 分布式tensorflow

推荐使用 TensorFlow Estimator API 来编写分布式训练代码，理由如下：

* 开发方便，比起low level的api开发起来更加容易
* 可以方便地和其他的高阶API结合使用，比如Dataset、FeatureColumns、Head等
* 模型函数model_fn的开发可以使用任意的low level函数，依然很灵活
* 单机和分布式代码一致，且不需要考虑底层的硬件设施
* 可以比较方便地和一些分布式调度框架（e.g. xlearning）结合使用

要让tensorflow分布式运行，首先我们需要定义一个由参与分布式计算的机器组成的集群，如下：
```
 cluster = {'chief': ['host0:2222'],
             'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
```
集群中一般有多个worker，需要指定其中一个worker为主节点（cheif），chief节点会执行一些额外的工作，比如模型导出之类的。在PS分布式架构环境中，还需要定义ps节点。

要运行分布式Estimator模型，只需要设置好`TF_CONFIG`环境变量即可，可参考如下代码：
```
  # Example of non-chief node:
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': 'worker', 'index': 1}})
  
  # Example of chief node:     
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': 'chief', 'index': 0}})
  
  # Example of evaluator node (evaluator is not part of training cluster)     
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': 'evaluator', 'index': 0}})
```
定义好上述环境变量后，调用`tf.estimator.train_and_evaluate`即可开始分布式训练和评估，其他部分的代码跟开发单机的程序是一样的，可以参考下面的资料：

1. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列: 开篇](https://zhuanlan.zhihu.com/p/38470806)
2. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列: 基于Dataset API处理Input pipeline](https://zhuanlan.zhihu.com/p/38421397)
3. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列: 自定义Estimator（以文本分类CNN模型为例）](https://zhuanlan.zhihu.com/p/41473323)
4. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列: 特征工程 Feature Column](https://zhuanlan.zhihu.com/p/41663141)
5. [基于Tensorflow高阶API构建大规模分布式深度学习模型系列: CVR预估案例之ESMM模型](https://zhuanlan.zhihu.com/p/42214716)

## 参考资料

1. [Distributed TensorFlow](https://www.oreilly.com/ideas/distributed-tensorflow)
2. [Goodbye Horovod, Hello CollectiveAllReduce
](https://www.logicalclocks.com/goodbye-horovod-hello-tensorflow-collectiveallreduce/)
3. [Overview: Distributed training using TensorFlow Estimator APIs](https://cloud.google.com/solutions/partners/quantiphi-distributed-training-using-tensorflow)
