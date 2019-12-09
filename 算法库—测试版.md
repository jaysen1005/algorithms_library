<h1>算法库</h1>

<h2>聚类算法</h2>

对于监督学习(supervised learning)，其训练样本是带有标记信息的，并且监督学习的目的是：对带有标记的数据集进行模型学习，从而便于对新的样本进行分类。而在无监督学习(unsupervised learning)中，训练样本的标记信息是未知的，目标是通过对无标记训练样本的学习来揭示数据的内在性质及规律，为进一步的数据分析提供基础。对于无监督学习，应用最广的便是聚类(clustering)。

聚类算法试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个簇(cluster)，通过这样的划分，每个簇可能对应于一些潜在的概念或类别。

<h3>K_Means</h3>

**介绍**

kmeans算法又名k均值算法。其算法思想大致为：先从样本集中随机选取 k个样本作为簇中心，并计算所有样本与这 k 个簇中心的距离，对于每一个样本，将其划分到与其距离最近的簇中心所在的簇中，对于新的簇计算各个簇的新的簇中心。

**理论知识**

1.k 值的选择 

k 的选择一般是按照实际需求进行决定，或在实现算法时直接给定 k 值。

2.距离的度量 

距离的度量方法主要分为以下几种：
*  闵可夫斯基距离(Minkowski distance)：

![image](/uploads/d618baa5b3766e465cf9227af1aecbf7/image.png)

*  欧氏距离(Euclidean distance)，即当 p=2 时的闵可夫斯基距离:

![image](/uploads/7eda6b0012384d7c6b5543cb5cc2b208/image.png)

*  c.曼哈顿距离(Manhattan distance)，即当 p=1 时的闵可夫斯基距离:

![image](/uploads/5abe462995bfaead2d96d56d54854e0d/image.png)

3.更新簇中心

对于划分好的各个簇，计算各个簇中的样本点均值，将其均值作为新的簇中心。

4.算法过程

> 输入：训练数据集 D=x(1),x(2),...,x(m) ,聚类簇数 k ;

> 过程：函数 kMeans(D,k,maxIter) .

> 1：从 DD 中随机选择 k个样本作为初始“簇中心”向量： μ(1),μ(2),...,,μ(k) :

> 2：repeat

> 3：  令 Ci=∅(1≤i≤k)

> 4：  for j=1,2,...,m do

> 5：  计算样本 x(j) 与各“簇中心”向量 μ(i)(1≤i≤k)的欧式距离

> 6：  根据距离最近的“簇中心”向量确定 x(j) 的簇标记： λj=argmini∈{1,2,...,k}dji

> 7：  将样本 x(j)划入相应的簇： Cλj=Cλj⋃{x(j)} ;

> 8：  end for

> 9：  for i=1,2,...,k do

> 10：    计算新“簇中心”向量： (μ(i))′=1/|Ci|∑x∈Cix ;

> 11：    if (μ(i))′=μ(i) then

> 12：      将当前“簇中心”向量 μ(i) 更新为 (μ(i))′

> 13：    else

> 14：      保持当前均值向量不变

> 15：    end if

> 16：  end for

> 17：  else

> 18：until 当前“簇中心”向量均未更新

> 输出：簇划分 C=C1,C2,...,CK

**简单示例**

> 1.导入：`from sklearn.cluster import KMeans`;

> 2.创建模型:`kmeans_model=KMeans(n_clusters= n_clusters,max_iter= max_iter, init = 'random', random_state = 100)`

> 3.训练：`km_fit = kmeans_model.fit(data_in)`

> 4.预测：`y_pre = kmeans_model.predict(x_test)`

**参数说明**

> n_clusters：聚类数，模型聚类结果的类别数目，大于0的整数。 

> max_iter：最大迭代次数，进行迭代次数的上限，大于0的整数。

> init：初始值选择方式，可选值：'k-means++'（用均值）、'random'（随机）、an ndarray（指定一个数组），默认为'k-means++'。

> random_state:随机种子数。

**适用场景**

kmeans算法是一个比较通用的算法，适用于凸样本集，比如均匀的簇大小, 平面几何, 不是太多的簇等。

-----------------

<h3>密度聚类</h3>

**介绍**

DBSCAN，具有噪声的基于密度的聚类方法，是一种很典型的密度聚类算法。

**理论知识**

DBSCAN是一种基于密度的聚类算法，这类密度聚类算法一般假定类别可以通过样本分布的紧密程度决定。同一类别的样本，他们之间的紧密相连的，也就是说，在该类别任意样本周围不远处一定有同类别的样本存在。 

通过将紧密相连的样本划为一类，这样就得到了一个聚类类别。通过将所有各组紧密相连的样本划为各个不同的类别，则我们就得到了最终的所有聚类类别结果。

2.距离的度量 

距离的度量方法主要分为以下几种：
*  闵可夫斯基距离(Minkowski distance)：

![image](/uploads/d618baa5b3766e465cf9227af1aecbf7/image.png)

*  欧氏距离(Euclidean distance)，即当 p=2 时的闵可夫斯基距离:

![image](/uploads/7eda6b0012384d7c6b5543cb5cc2b208/image.png)

*  c.曼哈顿距离(Manhattan distance)，即当 p=1 时的闵可夫斯基距离:

![image](/uploads/5abe462995bfaead2d96d56d54854e0d/image.png)

3.更新簇中心

对于划分好的各个簇，计算各个簇中的样本点均值，将其均值作为新的簇中心。

4.算法过程

> 输入：训练数据集 D=x(1),x(2),...,x(m) ,聚类簇数 k ;

> 过程：函数 kMeans(D,k,maxIter) .

> 1：从 DD 中随机选择 k个样本作为初始“簇中心”向量： μ(1),μ(2),...,,μ(k) :

> 2：repeat

> 3：  令 Ci=∅(1≤i≤k)

> 4：  for j=1,2,...,m do

> 5：  计算样本 x(j) 与各“簇中心”向量 μ(i)(1≤i≤k)的欧式距离

> 6：  根据距离最近的“簇中心”向量确定 x(j) 的簇标记： λj=argmini∈{1,2,...,k}dji

> 7：  将样本 x(j)划入相应的簇： Cλj=Cλj⋃{x(j)} ;

> 8：  end for

> 9：  for i=1,2,...,k do

> 10：    计算新“簇中心”向量： (μ(i))′=1/|Ci|∑x∈Cix ;

> 11：    if (μ(i))′=μ(i) then

> 12：      将当前“簇中心”向量 μ(i) 更新为 (μ(i))′

> 13：    else

> 14：      保持当前均值向量不变

> 15：    end if

> 16：  end for

> 17：  else

> 18：until 当前“簇中心”向量均未更新

> 输出：簇划分 C=C1,C2,...,CK

**简单示例**

> 1.导入：`from sklearn.cluster import KMeans`;

> 2.创建模型:`kmeans_model=KMeans(n_clusters= n_clusters,max_iter= max_iter, init = 'random', random_state = 100)`

> 3.训练：`km_fit = kmeans_model.fit(data_in)`

> 4.预测：`y_pre = kmeans_model.predict(x_test)`

**参数说明**

> n_clusters：聚类数，模型聚类结果的类别数目，大于0的整数。 

> max_iter：最大迭代次数，进行迭代次数的上限，大于0的整数。

> init：初始值选择方式，可选值：'k-means++'（用均值）、'random'（随机）、an ndarray（指定一个数组），默认为'k-means++'。

> random_state:随机种子数。

**适用场景**

kmeans算法是一个比较通用的算法，适用于凸样本集，比如均匀的簇大小, 平面几何, 不是太多的簇等。

-------------

<h3>层次聚类</h3>

**介绍**

kmeans算法又名k均值算法。其算法思想大致为：先从样本集中随机选取 k个样本作为簇中心，并计算所有样本与这 k 个簇中心的距离，对于每一个样本，将其划分到与其距离最近的簇中心所在的簇中，对于新的簇计算各个簇的新的簇中心。

**理论知识**

1.k 值的选择 

k 的选择一般是按照实际需求进行决定，或在实现算法时直接给定 k 值。

2.距离的度量 

距离的度量方法主要分为以下几种：
*  闵可夫斯基距离(Minkowski distance)：

![image](/uploads/d618baa5b3766e465cf9227af1aecbf7/image.png)

*  欧氏距离(Euclidean distance)，即当 p=2 时的闵可夫斯基距离:

![image](/uploads/7eda6b0012384d7c6b5543cb5cc2b208/image.png)

*  c.曼哈顿距离(Manhattan distance)，即当 p=1 时的闵可夫斯基距离:

![image](/uploads/5abe462995bfaead2d96d56d54854e0d/image.png)

3.更新簇中心

对于划分好的各个簇，计算各个簇中的样本点均值，将其均值作为新的簇中心。

4.算法过程

> 输入：训练数据集 D=x(1),x(2),...,x(m) ,聚类簇数 k ;

> 过程：函数 kMeans(D,k,maxIter) .

> 1：从 DD 中随机选择 k个样本作为初始“簇中心”向量： μ(1),μ(2),...,,μ(k) :

> 2：repeat

> 3：  令 Ci=∅(1≤i≤k)

> 4：  for j=1,2,...,m do

> 5：  计算样本 x(j) 与各“簇中心”向量 μ(i)(1≤i≤k)的欧式距离

> 6：  根据距离最近的“簇中心”向量确定 x(j) 的簇标记： λj=argmini∈{1,2,...,k}dji

> 7：  将样本 x(j)划入相应的簇： Cλj=Cλj⋃{x(j)} ;

> 8：  end for

> 9：  for i=1,2,...,k do

> 10：    计算新“簇中心”向量： (μ(i))′=1/|Ci|∑x∈Cix ;

> 11：    if (μ(i))′=μ(i) then

> 12：      将当前“簇中心”向量 μ(i) 更新为 (μ(i))′

> 13：    else

> 14：      保持当前均值向量不变

> 15：    end if

> 16：  end for

> 17：  else

> 18：until 当前“簇中心”向量均未更新

> 输出：簇划分 C=C1,C2,...,CK

**简单示例**

> 1.导入：`from sklearn.cluster import KMeans`;

> 2.创建模型:`kmeans_model=KMeans(n_clusters= n_clusters,max_iter= max_iter, init = 'random', random_state = 100)`

> 3.训练：`km_fit = kmeans_model.fit(data_in)`

> 4.预测：`y_pre = kmeans_model.predict(x_test)`

**参数说明**

> n_clusters：聚类数，模型聚类结果的类别数目，大于0的整数。 

> max_iter：最大迭代次数，进行迭代次数的上限，大于0的整数。

> init：初始值选择方式，可选值：'k-means++'（用均值）、'random'（随机）、an ndarray（指定一个数组），默认为'k-means++'。

> random_state:随机种子数。

**适用场景**

和K-Means，BIRCH这些一般只适用于凸样本集的聚类相比，DBSCAN既可以适用于凸样本集，也可以适用于非凸样本集。

---------

<h2>分类算法</h2>

<h2>回归算法</h2>