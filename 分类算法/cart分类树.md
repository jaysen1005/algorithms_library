

<h3>CART分类树</h3>

##### 简介

  CART算法是一种二分递归分割技术，把当前样本划分为两个子样本，使得生成的每个非叶子结点都有两个分支，因此CART算法生成的决策树是结构简洁的二叉树。由于CART算法构成的是一个二叉树，它在每一步的决策时只能是“是”或者“否”，即使一个feature有多个取值，也是把数据分为两部分。

##### 理论知识

CART分类树算法使用基尼系数来选择特征，基尼系数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好。这和信息增益（比）相反。假设K个类别，第k个类别的概率为pk，概率分布的基尼系数表达式：

![1235684-20190320142851428-1194269689](/uploads/661970b827fd83d358806fddfe817a4a/1235684-20190320142851428-1194269689.png)

如果是二分类问题，第一个样本输出概率为p，概率分布的基尼系数表达式为：

![1235684-20190320143135391-255494198](/uploads/0a84f7446e1fa4e3a74bf414587b65f8/1235684-20190320143135391-255494198.png)

对于样本D，个数为|D|，假设K个类别，第k个类别的数量为|Ck|，则样本D的基尼系数表达式：

![1235684-20190320143545086-323258879](/uploads/7a503b6e9e80e640b12eb4d5d2fa1305/1235684-20190320143545086-323258879.png)

对于样本D，个数为|D|，根据特征A的某个值a，把D分成|D1|和|D2|，则在特征A的条件下，样本D的基尼系数表达式为：

![1235684-20190320144250953-1096584260](/uploads/ca84f30418e1285f715141534bb37033/1235684-20190320144250953-1096584260.png)

在CART算法中主要分为两个步骤：

1、将样本递归划分进行建树过程。

 算法从根节点开始，用训练集递归建立CART分类树。

 >(1)对于当前节点的数据集为D，如果样本个数小于阈值或没有特征，则返回决策子树，当前节点停止递归。

 >(2)计算样本集D的基尼系数，如果基尼系数小于阈值，则返回决策树子树，当前节点停止递归。

 >(3)计算当前节点现有的各个特征的各个特征值对数据集D的基尼系数。

 >(4)在计算出来的各个特征的各个特征值对数据集D的基尼系数中，选择基尼系数最小的特征A和对应的特征值a。根据这个最优特征和最优特征值，把数据集划分成两部分D1和D2，同时建立当前节点的左右节点，做节点的数据集D为D1，右节点的数据集D为D2。

 >(5)对左右的子节点递归的调用1-4步，生成决策树。

2、用验证数据进行剪枝。

决策树很容易对训练集过拟合，导致泛化能力差，所以要对CART树进行剪枝，即类似线性回归的正则化。CART采用后剪枝法，即先生成决策树，然后产生所有剪枝后的CART树，然后使用交叉验证检验剪枝的效果，选择泛化能力最好的剪枝策略。用验证数据集对生成的树进行剪枝并选择最优子树，损失函数最小作为剪枝的标准。CART分类树的剪枝策略在度量损失的时候用基尼系数。剪枝损失函数表达式：

![1235684-20190322105719685-676636285](/uploads/62acbae9f3d0c687de3c262867e5dbc0/1235684-20190322105719685-676636285.png)

具体过程如下：

>(1)初始化αmin = ∞，最优子树集合ω = {T}。

>(2)从叶子结点开始自下而上计算内部节点 t 的训练误差损失函数Cα(Tt)（回归树为均方差，分类树为基尼系数），叶子节点数|Tt|，以及正则化阈值![1235684-20190323114336846-360679535](/uploads/aa5d9ad99255b3ff375598b30e27d588/1235684-20190323114336846-360679535.png)，更新αmin = α

>(3)得到所有节点的α值得集合M。

>(4)从M中选择最大的值αk，自上而下的访问子树 t 的内部节点，如果时，进行剪枝。并决定叶子节点 t 的值。如果![1235684-20190323114615583-1980334920](/uploads/a72ff4d2e9e61976ce961e8ffa46f6ba/1235684-20190323114615583-1980334920.png)是分类树，这是概率最高的类别，如果是回归树，这是所有样本输出的均值。这样得到αk对应的最优子树Tk

>(5)最优子树集合ω = ωυTk，M = M - {αk}。

>(6)如果M不为空，则回到步骤4。否则就已经得到了所有的可选最优子树集合ω。

>(7)采用交叉验证在ω选择最优子树Tα。

##### 应用场景

CART建模方法在数据统计方面具有很好的灵活性，因此它在医疗判断、气象预测、物流管理、数据挖掘、投资风险分析等行业领域已经有成功的应用。

##### 参数及说明

`class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)`

*  **参数**

>**criterion：**特征选择标准，【entropy, gini】。默认gini，即CART算法。

>**splitter：**特征划分标准，【best, random】。best在特征的所有划分点中找出最优的划分点，random随机的在部分划分点中找局部最优的划分点。默认的‘best’适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐‘random’。

>**max_depth：**决策树最大深度，【int,  None】。默认值是‘None’。一般数据比较少或者特征少的时候可以不用管这个值，如果模型样本数量多，特征也多时，推荐限制这个最大深度，具体取值取决于数据的分布。常用的可以取值10-100之间，常用来解决过拟合。

>**min_samples_split：**内部节点（即判断条件）再划分所需最小样本数，【int, float】。默认值为2。如果是int，则取传入值本身作为最小样本数；如果是float，则取ceil(min_samples_split*样本数量)作为最小样本数。（向上取整）

>**min_samples_leaf：**叶子节点（即分类）最少样本数。如果是int，则取传入值本身作为最小样本数；如果是float，则取ceil(min_samples_leaf*样本数量)的值作为最小样本数。这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。

>**min_weight_fraction_leaf：**叶子节点（即分类）最小的样本权重和，【float】。这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。默认是0，就是不考虑权重问题，所有样本的权重相同。一般来说如果我们有较多样本有缺失值或者分类树样本的分布类别偏差很大，就会引入样本权重，这时就要注意此值。

>**max_features：**在划分数据集时考虑的最多的特征值数量，【int值】。在每次split时最大特征数；【float值】表示百分数，即（max_features*n_features）

>**random_state：**【int, randomSate instance, None】，默认是None

>**max_leaf_nodes：**最大叶子节点数。【int, None】，通过设置最大叶子节点数，可以防止过拟合。默认值None，默认情况下不设置最大叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征多，可以加限制，具体的值可以通过交叉验证得到。

>**min_impurity_decrease：**节点划分最小不纯度，【float】。默认值为‘0’。限制决策树的增长，节点的不纯度（基尼系数，信息增益，均方差，绝对差）必须大于这个阈值，否则该节点不再生成子节点。

>**min_impurity_split（已弃用）：**信息增益的阀值。决策树在创建分支时，信息增益必须大于这个阈值，否则不分裂。（从版本0.19开始不推荐使用：min_impurity_split已被弃用，以0.19版本中的min_impurity_decrease取代。min_impurity_split的默认值将在0.23版本中从1e-7变为0，并且将在0.25版本中删除。 请改用min_impurity_decrease。）

>**class_weight：**类别权重，【dict, list of dicts, balanced】，默认为None。（不适用于回归树，sklearn.tree.DecisionTreeRegressor）。指定样本各类别的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。balanced，算法自己计算权重，样本量少的类别所对应的样本权重会更高。如果样本类别分布没有明显的偏倚，则可以不管这个参数。

>**presort：**bool，默认是False，表示在进行拟合之前，是否预分数据来加快树的构建。对于数据集非常庞大的分类，presort=true将导致整个分类变得缓慢；当数据集较小，且树的深度有限制，presort=true才会加速分类。

*  **方法**

>（1）训练（拟合）：fit(X, y[, sample_weight])——fit(train_x, train_y)。

>（2）预测：predict(X)返回标签、predict_log_proba(X)、predict_proba(X)返回概率，每个点的概率和为1，一般取predict_proba(X)[:, 1]。

>（3）评分（返回平均准确度）：score(X, y[, sample_weight])——score(test_x, test_y)。等效于准确率accuracy_score。

>（4）参数类：获取分类器的参数get_params([deep])、设置分类器的参数set_params(**params)。——print(clf.get_params()) ，clf.set_params(***)。

* **DecisionTreeClassifier的其他方法：** 

>*  apply(X[, check_input])	
 Returns the index of the leaf that each sample is predicted as.  
 返回每个样本被预测为叶子的索引。
>*  decision_path(X[, check_input])	
 Return the decision path in the tree.
 返回树的决策路径
>*  get_depth()	                            
 Returns the depth of the decision tree.  
 获取决策树的深度
>*  get_n_leaves()	
 Returns the number of leaves of the decision tree.  
 获取决策树的叶子节点数

##### **python/pyspark**样例代码

`from sklearn import tree`

`X =``[``[0, 0], [1, 1]``]`

`Y = [0, 1]`

`clf = tree.DecisionTreeClassifier()`

`clf = clf.fit(X, Y)`

`clf.predict([``[2., 2.]``])`

`tree.plot_tree(clf)`