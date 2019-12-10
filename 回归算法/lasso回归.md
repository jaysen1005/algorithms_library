<h2>Lasso</h2>

<h3>介绍</h3>

　　lasso算法是最小绝对值收敛和选择算子、套索算法。该方法是一种压缩估计。它通过构造一个罚函数得到一个较为精炼的模型，使得它压缩一些系数，同时设定一些系数为零。因此保留了子集收缩的优点，是一种处理具有复共线性数据的有偏估计。

<h3>理论知识</h3>

1.岭回归算法原理

　　Lasso 的基本思想是在回归系数的绝对值之和小于一个常数的约束条件下，使残差平方和最小化，从而能够产生某些严格等于0 的回归系数，得到可以解释的模型。

2.公式

> 给定数据集![image](/uploads/9c244d870a780c4aa756e05bedeaece9/image.png)其中，![image](/uploads/8c1c4928ce31c383fc4282434d56028b/image.png),![image](/uploads/2b32c24a46dca691cfc6797cb07cc611/image.png)

> 代价函数为：![image](/uploads/67cc6d0072f6c2ba255e9f11f1e15017/image.png)

> L1范数正则化（LASSO，最小绝对收缩选择算子）代价函数为：

> ![image](/uploads/db6c8e7b8ea04cc7b76d5ced0bbb0cd2/image.png)

> 其中，L1范数正则化有助于降低过拟合风险

3.Lasso回归求解

　　由于L1范数用的是绝对值，导致LASSO的优化目标不是连续可导的，也就是说，最小二乘法，梯度下降法，牛顿法，拟牛顿法都不能用。

　　L1正则化问题求解可采用近端梯度下降法(PGD):

> （1）优化目标:

> 优化目标为：![image](/uploads/6948c3688f1dbf0ca7c2e69466f7d6a4/image.png)

> 若![image](/uploads/dbfed5a390bd11115b291994c8f289f5/image.png)可导，梯度![image](/uploads/569e18bc2dc1415541418a897cfca4c9/image.png)满足L-Lipschitz条件（利普希茨连续条件），即存在常数L> 0，使得：![image](/uploads/98bf8d050991d0346c833d624b95605f/image.png)

> （2）泰勒展开

> 在![image](/uploads/6b7e5c51972242c306a4f14f29e56b55/image.png)处将![image](/uploads/ee765c13e2f0aa73ebe8d1f9e07c2392/image.png)进行二阶泰勒展开：![image](/uploads/de731f62375d3b8d04dee49f3b6d4fdc/image.png)

> 得到![image](/uploads/006766b185daf6a91624d8b844b7dccd/image.png)

> （3）简化泰勒展开式

> 将上式化简：![image](/uploads/7c29d56a401adb594366ebce356eac82/image.png)其中，![image](/uploads/c458341abf6dc2aa6a913028adecbf72/image.png)是X无关的常数。

> （4）简化优化问题

> 这里若通过梯度下降法对![image](/uploads/aafa19b395204514fdb89afe12aa5452/image.png)进行最小化，则每一步下降迭代实际上等价于最小化二次函数![image](/uploads/5c439f4de7006bb510631ff7602b94bc/image.png),推广到优化目标，可得到每一步迭代公式：![image](/uploads/e57711efeaeded32fd78a76a1eb695cf/image.png)

> 令![image](/uploads/44431eb6d8aa357275399923da2914ae/image.png),则可以先求ž，再求解优化问题：![image](/uploads/fd40b765387866c29b4127ccadc69b05/image.png)

> （5）求解

> 令![image](/uploads/d53ff275dfa9225bf624828982af2d93/image.png)为X的第一i个分量，将上式按分量展开，其中不存在![image](/uploads/0810f97044e9a2726ceb20d8387a241f/image.png)这样的项，即X的各分量之间互不影响，所以有闭式解。对于上述优化问题需要用到soft thresholding软阈值函数，即对于优化问题：![image](/uploads/8919d6bc5a09c9cd351b3eb7fcbad176/image.png)

> 其解为：![image](/uploads/0fbe56b754b7543c7492ee298ad77e2e/image.png)

> 而我们的优化问题得到闭式解为：![image](/uploads/148c6e1a59bb498895b92899e2879d5b/image.png)其中，![image](/uploads/5d1f94da221a46491e7328c17ee96f92/image.png)与![image](/uploads/3dc80b2e22ba174bb2fd2f06633faccc/image.png)分别是![image](/uploads/97db0ab387a192a57c1d39061bbe8865/image.png)与z的第一i个分量。因此，通过PGD能使LASSO和其他基于L1范数最小化的方法得以快速求解。

<h3>python/pyspark样例代码</h3>

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