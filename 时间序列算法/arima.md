

<h3>ARIMA</h3>

##### 简介

　　ARIMA模型（AutoregressiveIntegratedMovingAverage model），差分整合移动平均自回归模型，又称整合移动平均自回归模型（移动也可称作滑动），时间序列预测分析方法之一。ARIMA（p，d，q）中，AR是"自回归"，p为自回归项数；MA为"滑动平均"，q为滑动平均项数，d为使之成为平稳序列所做的差分次数（阶数）。“差分”一词虽未出现在ARIMA的英文名称中，却是关键步骤。ARIMA（p，d，q）模型是ARMA（p，q）模型的扩展。ARIMA（p，d，q）模型可以表示为：![t1](/uploads/8150f4403c49bddd66181e01d3e816d7/t1.png)。其中L是滞后算子（Lag operator），![t2](/uploads/a58d707b3d2159e2d7a05fada6b8f1a0/t2.png)。

##### 理论知识

ARIMA模型建立流程如下：
>1.根据时间序列的散点图、自相关函数和偏自相关函数图识别其平稳性。
*  自相关函数ACF(autocorrelation function)：
自相关函数ACF描述的是时间序列观测值与其过去的观测值之间的线性相关性。计算公式如下：![t3](/uploads/3afb8fdbb6e197a187c8c1b387021baf/t3.png)。其中k代表滞后期数，如果k=2，则代表yt和yt-2。
*  偏自相关函数PACF(partial autocorrelation function)：
偏自相关函数PACF描述的是在给定中间观测值的条件下，时间序列观测值预期过去的观测值之间的线性相关性。
举个简单的例子，假设k=3，那么我们描述的是yt和yt-3之间的相关性，但是这个相关性还受到yt-1和yt-2的影响。PACF剔除了这个影响，而ACF包含这个影响。

>2.对非平稳的时间序列数据进行平稳化处理。直到处理后的自相关函数和偏自相关函数的数值非显著非零。
平稳性就是要求经由样本时间序列所得到的拟合曲线，在未来的一段时间内仍能顺着现有状态“惯性”地延续下去；序列的均值和方差不发生明显变化；一般采用差分法对非平稳序列进行平稳化处理。                           

>3.根据所识别出来的特征建立相应的时间序列模型。平稳化处理后，若偏自相关函数是截尾的，而自相关函数是拖尾的，则建立AR模型；若偏自相关函数是拖尾的，而自相关函数是截尾的，则建立MA模型；若偏自相关函数和自相关函数均是拖尾的，则序列适合ARMA模型。                                                                     

>**拖尾**指序列以指数率单调递减或震荡衰减，而截尾指序列从某个时点变得非常小：
![t4](/uploads/db780124f33a2e0b37d87920135916f1/t4.png)                                  
出现以下情况，通常视为(偏)自相关系数d阶截尾：
1）在最初的d阶明显大于2倍标准差范围。
2）之后几乎95%的(偏)自相关系数都落在2倍标准差范围以内。
3）且由非零自相关系数衰减为在零附近小值波动的过程非常突然。                                      
![t5](/uploads/56b73d29a141485160bf9e0a5b0779ca/t5.png)                                  
出现以下情况，通常视为(偏)自相关系数拖尾：
1）如果有超过5%的样本(偏)自相关系数都落入2倍标准差范围之外
2）或者是由显著非0的(偏)自相关系数衰减为小值波动的过程比较缓慢或非常连续。
![t6](/uploads/ddeb511a2c37a5f2b354cb6d35fb4081/t6.png)                                   
**p，q阶数的确定：**
根据刚才判定截尾和拖尾的准则，p，q的确定基于如下的规则：                                           
![t7](/uploads/10d57fe0437238b75644f8ab636942c0/t7.png)                                  
根据不同的截尾和拖尾的情况，我们可以选择AR模型，也可以选择MA模型，当然也可以选择ARIMA模型。

>4.参数估计，检验是否具有统计意义。

>　通过拖尾和截尾对模型进行定阶的方法，往往具有很强的主观性。在相同的预测误差情况下，根据奥斯卡姆剃刀准则，模型越小是越好的。那么，平衡预测误差和参数个数，我们可以根据信息准则函数法，来确定模型的阶数。预测误差通常用平方误差即残差平方和来表示。

>　常用的信息准则函数法有下面几种：

>*  AIC准则
AIC准则全称为全称是最小化信息量准则（Akaike Information Criterion），计算公式如下：AIC = =2 *（模型参数的个数）-2ln（模型的极大似然函数）
>*  BIC准则
AIC准则存在一定的不足之处。当样本容量很大时，在AIC准则中拟合误差提供的信息就要受到样本容量的放大，而参数个数的惩罚因子却和样本容量没关系（一直是2），因此当样本容量很大时，使用AIC准则选择的模型不收敛与真实模型，它通常比真实模型所含的未知参数个数要多。BIC（Bayesian InformationCriterion）贝叶斯信息准则弥补了AIC的不足，计算公式如下：BIC = ln(n) * (模型中参数的个数) – 2ln(模型的极大似然函数值)，n是样本容量

>5.假设检验，判断（诊断）残差序列是否为白噪声序列。

>   这里的模型检验主要有两个：
>  1）检验参数估计的显著性（t检验）。
>  2）检验残差序列的随机性，即残差之间是独立的。
>  残差序列的随机性可以通过自相关函数法来检验，即做残差的自相关函数图。

>6.利用已通过检验的模型进行预测。

##### 应用场景

ARIMA模型的基本思想是：将预测对象随时问推移而形成的数据序列视为—个随机序列.以时间序列的自相关分析为基础.用一定的数学模型来近似描述这个序列。这个模型一旦被识别后就可以从时间序列的过去值及现在值来预测未来值。ARlMA模型在经济预测过程中既考虑了经济现象在时间序列上的依存性，又考虑了随机波动的干扰性，对于经济运行短期趋势的预测准确率较高，是近年应用比较广泛的方法之一。 

##### 参数及说明

`class statsmodels.tsa.statespace.sarimax.SARIMAX(endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None, measurement_error=False, time_varying_regression=False, mle_regression=True, simple_differencing=False, enforce_stationarity=True, enforce_invertibility=True, hamilton_representation=False, concentrate_scale=False, **kwargs)`

*  **参数**

>**endog：**观察（自）变量y。

>**exog：**外部变量。

>**order：**自回归，差分，滑动平均项 (p,d,q)。

>**seasonal_order：**季节因素的自回归，差分，移动平均，周期 (P,D,Q,s)。

>**trend：**趋势，c表示常数，t:线性，ct:常数+线性。

>**measurement_error：**自变量的测量误差。

>**time_varying_regression：**外部变量是否存在不同的系数。

>**mle_regression：**是否选择最大似然极大参数估计方法。

>**simple_differencing：**简单差分，是否使用部分条件极大似然。

>**enforce_stationarity：**是否在模型种使用强制平稳。

>**enforce_invertibility：**是否使用移动平均转换。

>**hamilton_representation：**是否使用汉密尔顿表示。

>**concentrate_scale：**是否允许标准误偏大。

>**trend_offset：**是否存在趋势。

>**use_exact_diffuse：**是否使用非平稳的初始化。

>**kwargs：接受不定数量的参数，如空间状态矩阵和卡尔曼滤波。

##### **python/pyspark**样例代码

`from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(np.asarray(timeseries).astype(np.float64),order=order)
model_fit = model.fit(disp=0)
model_fit.predict(len(df),len(df))
model_fit.summary()`
