

<h3>ID3算法</h3>

##### 简介

  ID3算法是一种贪心算法，用来构造决策树。ID3算法起源于概念学习系统（CLS），以信息熵的下降速度为选取测试属性的标准，即在每个节点选取还尚未被用来划分的具有最高信息增益的属性作为划分标准，然后继续这个过程，直到生成的决策树能完美分类训练样例。

##### 理论知识

ID3算法的核心是在决策树各个结点上应用信息增益准则选择特征，递归地构建决策树。具体方法是：从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该结点的不同取值建立子结点；再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息增益均很小或没有特征可以选择为止。最后得到一个决策树。ID3算法是以信息熵和信息增益作为衡量标准的分类算法。
*  信息熵(Entropy)

熵的概念主要是指信息的混乱程度，变量的不确定性越大，熵的值也就越大，熵的公式可以表示为：![e1](/uploads/57c4824ed028be3010b270afac4c0592/e1.gif)。其中，![e2](/uploads/a1af2a665d770e380072e8780057859a/e2.gif)，![e3](/uploads/46374abac5734fc2e03c7bc5c88256c4/e3.gif)为类别![e4](/uploads/150905cae1ec3001ea88cf03a820d48d/e4.gif)在样本S中出现的概率。

*  信息增益(Information gain)

信息增益指的是划分前后熵的变化，可以用下面的公式表示：![e5](/uploads/8d16078df18f36eb6a6b6a3203fd48c7/e5.gif)。其中，A表示样本的属性，![e6](/uploads/05622413ae585b9fce0c3620163f70b3/e6.gif)是属性A所有的取值集合。V是A的其中一个属性值，![e7](/uploads/f9ae3fb3e3daf23ab33d0505a883354d/e7.gif)是S中A的值为V的样例集合。

##### 应用场景

企业管理实践，企业投资决策，由于决策树很好的分析能力，在决策过程应用较多。

##### 参数及说明

`class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)`

*  **参数**

>**criterion：**特征选择标准，【entropy, gini】。默认gini，即CART算法。该参数选择entropy，即为ID3算法。

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

`from sklearn import tree
X = [``[0, 0], [1, 1]``]
Y = [0, 1]
criterion = "entropy"
splitter = "best"
max_depth = None
min_samples_split = 2
min_samples_leaf  = 1
min_weight_fraction_leaf = 0
id3 = tree.DecisionTreeClassifier(criterion=criterion], splitter=splitter, max_depth = max_depth , min_samples_split = min_samples_split ,min_samples_leaf = min_samples_leaf ,min_weight_fraction_leaf = min_weight_fraction_leaf)
clf = id3.fit(X, Y)
clf.predict([``[2., 2.]``])
tree.plot_tree(clf)`