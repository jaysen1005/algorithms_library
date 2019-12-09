<h2>Lasso</h2>

<h3>介绍</h3>

　　lasso算法是最小绝对值收敛和选择算子、套索算法。该方法是一种压缩估计。它通过构造一个罚函数得到一个较为精炼的模型，使得它压缩一些系数，同时设定一些系数为零。因此保留了子集收缩的优点，是一种处理具有复共线性数据的有偏估计。

<h3>理论知识</h3>

1.岭回归算法原理

　　Lasso 的基本思想是在回归系数的绝对值之和小于一个常数的约束条件下，使残差平方和最小化，从而能够产生某些严格等于0 的回归系数，得到可以解释的模型。

　　其次，岭回归是基于最小二乘法![image](/uploads/00c342a054783b5972f95e0ad464abf1/image.png)， 最小二乘法中有时候 ![image](/uploads/9070bd308d57009299c2af84564c811f/image.png)可能不是满秩矩阵，也就是此时行列式为零，无法求逆 (![image](/uploads/713c1996df1cc8561591d0b583b7edee/image.png) 其中![image](/uploads/b70d55c9be4d4cd488d68f10d3faea86/image.png)是伴随矩阵)

2.λ的选择

　　* 模型的方差：回归系数的方差 

　　* 模型的偏差：预测值和真实值的差异 

　　随着模型复杂度的提升，在训练集上的效果就越好，即模型的偏差就越小；但是同时模型的方差就越大。对于岭回归的λ而言，随着λ的增大，![image](/uploads/f43a73fb7277d8bb467f2a4357656f21/image.png)就越大，![image](/uploads/a8ff722118bb6ad87e0ce8c0a0f85b93/image.png)
 就越小，模型的方差就越小；而λ越大使得![image](/uploads/09a773cbae34fa419dd0a778a9d7f376/image.png)的估计值更加偏离真实值，模型的偏差就越大。所以岭回归的关键是找到一个合理的λ值来平衡模型的方差和偏差。

3.岭回归的一些性质

　　1）.当岭参数λ=0时，得到的解是最小二乘解

　　2）.当岭参数λ趋向更大时，岭回归系数wi趋向于0，约束项t很小

<h3>简单示例</h3>

> 1.导入：`from sklearn.linear_model import Ridge`;

> 2.创建模型:`ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)`

> 3.训练：`ridge.fit(X_train,y_train)`

> 4.预测：`print(ridge.coef_)` ，`print(ridge.intercept_)` 

<h3>参数说明</h3>

> alpha：正则化系数，较大的值指定更强的正则化

> fit_intercept:是否计算模型的截距，默认为True，计算截距

> normalize:在需要计算截距时，如果值为True，则变量x在进行回归之前先进行归一化,如果需要进行标准化则normalize=False。若不计算截距，则忽略此参数

> copy_X:默认为True，将复制X；否则，X可能在计算中被覆盖。

> max_iter:共轭梯度求解器的最大迭代次数。对于sparse_cg和lsqr,默认值由scipy.sparse.linalg确定。对于sag求解器，默认值为1000。

> tol:float类型，指定计算精度

> solver：求解器,可选值：{auto,svd,cholesky,lsqr,sparse_cg,sag,saga}

> random_state:随机数生成器的种子。

<h3>适用场景</h3>

　　　lasso回归算法可以在参数估计的同时实现变量的选择，较好的解决回归分析中的多重共线性问题，并且能够很好的解释结果。