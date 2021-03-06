<h2>CartRegressor</h2>

<h3>介绍</h3>

　　CART分类与回归树本质上是一样的，构建过程都是逐步分割特征空间，预测过程都是从根节点开始一层一层的判断直到叶节点给出预测结果。只不过分类树给出离散值，而回归树给出连续值(通常是叶节点包含样本的均值），另外分类树基于Gini指数选取分割点，而回归树基于平方误差选取分割点。

<h3>理论知识</h3>

1.基本原理

　　CART假设决策树是二叉树，内部结点特征的取值为“是”和“否”，左分支是取值为“是”的分支，右分支是取值为“否”的分支。这样的决策树等价于递归地二分每个特征，将输入空间即特征空间划分为有限个单元，并在这些单元上确定预测的概率分布，也就是在输入给定的条件下输出的条件概率分布。

　　CART算法由以下两步组成：

　　（1）决策树的生成：基于训练数据集生成决策树，生成的决策树要尽量大（大是为了更好地泛化）

　　（2）决策树剪枝：用验证数据集对已生成的树进行剪枝并选择最优子树，这时损失函数最小作为剪枝的标准。

2.回归树的生成

　　 在训练数据集所在的输入空间中，递归地将每个区域划分为两个子区域并决定每个子区域上的输出值，构建二叉决策树。

> （1）选择最优切分变量j与切分点s，求解![image](/uploads/d418df0f328a9e541528b4b7b746ca5d/image.png)

> 遍历变量j，对固定的切分变量j扫描切分点s，选择使上式达到最小值的对(j,s)j。

> （2）用选定的对(j,s)划分区域并决定响应的输出值：![image](/uploads/cfcf1db60e31185d974e7a427c63515a/image.png)

> （3）继续对两个子区域条用步骤（1），（2），直至满足停止条件。

> （4）将输入空间划分为M个区域R1,R2,...,RM,生成决策树：![image](/uploads/4576f9bd1c6bc57c0fe693912aed6b39/image.png)

<h3>python/pyspark样例代码</h3>

> 1.导入：`from sklearn.tree import DecisionTreeRegressor`;

> 2.创建模型:`model = DecisionTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease,presort)`

> 3.训练：`cartModel = model.fit(X_train,y_train)`

> 4.预测：`y_pre = model.predict(X_test)` 

<h3>参数说明</h3>

> criterion ：切分评价准则。切分时的评价准则。可选mse（均方差），friedman_mse（平均绝对误差），默认为均方差'mse'

> splitter:切分原则。可选参数:'best'(最优),'random'(随机)。

> max_depth:树的最大深度。默认为None，表示树的深度不限。直到所有的叶子节点都是纯净的，即叶子节点

> min_samples_split: 子树划分所需最小样本数。子树继续划分所需最小样本数。int，默认为2

> min_samples_leaf:叶子节点最少样本数。int,默认为1

> min_weight_fraction_leaf:叶子节点最小的样本权重和。非负数类型，默认为0.0

> max_features：最大特征数。模型保留最大特征数。可输入int, float类型的数值，也可选择输入auto（原特征数）, sqrt（开方）, log2, None（原特征数）。

> max_leaf_nodes:最大叶子节点数。正float、int类型或None。默认为None。

> min_impurity_decrease:节点划分最小减少不纯度。非负数类型,默认为0.0。

> presort:预排序。数据是否预排序，bool。False或True。

<h3>适用场景</h3>

　　cart回归树算法适用于当数据拥有众多特征并且特征之间关系十分复杂时，构建全局模型就显得太难了，可以利用该回归算法技术来建模，例如预测用户的购买倾向、市场细分和客户促销研究等。