<h2>Lasso</h2>

<h3>介绍</h3>

　　lasso算法是最小绝对值收敛和选择算子、套索算法。该方法是一种压缩估计。它通过构造一个罚函数得到一个较为精炼的模型，使得它压缩一些系数，同时设定一些系数为零。因此保留了子集收缩的优点，是一种处理具有复共线性数据的有偏估计。

<h3>理论知识</h3>

1.岭回归算法原理

　　Lasso 的基本思想是在回归系数的绝对值之和小于一个常数的约束条件下，使残差平方和最小化，从而能够产生某些严格等于0 的回归系数，得到可以解释的模型。

2.公式

　　L1正则化与L2正则化的区别在于惩罚项的不同：

　　![image](/uploads/cc85a22c8b15869c839063c8180d2324/image.png)

　　L1正则化表现的是θ的绝对值，变化为上面提到的w1和w2可以表示为：

　　![image](/uploads/b840e1cbf70d6d09e57feba74bc8e9da/image.png)

<h3>简单示例</h3>

> 1.导入：`from sklearn.linear_model import Lasso`;

> 2.创建模型:`model = Lasso(alpha, fit_intercept, max_iter, normalize, precompute, tol, warm_start, positive, selection)`

> 3.训练：`laModel = model.fit(features, label)`

> 4.预测：`laModel.predict(x_test)` 

<h3>参数说明</h3>

> alpha：正则化系数，较大的值指定更强的正则化

> fit_intercept:是否计算模型的截距，默认为True，计算截距

> normalize:在需要计算截距时，如果值为True，则变量x在进行回归之前先进行归一化,如果需要进行标准化则normalize=False。若不计算截距，则忽略此参数

> precompute :默认为True，将复制X；否则，X可能在计算中被覆盖。

> max_iter:共轭梯度求解器的最大迭代次数。对于sparse_cg和lsqr,默认值由scipy.sparse.linalg确定。对于sag求解器，默认值为1000。

> tol:float类型，指定计算精度

> warm_start ：bool, 热启动,可选为 True 时, 重复使用上一次学习作为初始化，否则直接清除上次方案。

> positive:bool, 强制正相关,可选 设为 True 时，强制使系数为正。

> selection:str, 选择器,默认 ‘cyclic’若设为 ‘random’, 每次循环会随机更新参数

<h3>适用场景</h3>

　　lasso回归算法可以在参数估计的同时实现变量的选择，较好的解决回归分析中的多重共线性问题，并且能够很好的解释结果。Lasso回归(L1正则化)可以使得一些特征的系数变小,甚至还使一些绝对值较小的系数直接变为0，从而增强模型的泛化能力 。对于高维的特征数据,尤其是线性关系是稀疏的，就采用Lasso回归,或者是要在一堆特征里面找出主要的特征，那么Lasso回归更是首选了。