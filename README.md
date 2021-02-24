# BI Project

这个Repository包含了我在推荐系统学习和工作中的一些Project的整理, **如果您觉得不错，求个star**

### 1: [TextSimilarityDetection] 文本抄袭自动检测分析
主要技术：Python / jieba / TF-IDF / MultinomialNB / KMeans / editdistance / TopN

项目简介：通过分析不同机构发布的文章，判断是否有文章抄袭的情况，并找到原文和抄袭的文章，以及具体相似的句子。可以应用于毕业论文查重，IP作品及文本抄袭检测

主要工作：对采集的文档s进行数据清洗，采用TF-IDF提取文本特征，使用朴素贝叶斯分类器进行写作风格分类，并针对模仿自己写作风格的文章进行抄袭检测。先采用聚类算法对文档进行聚类降维，针对预测写作风格一致的作品，进行相似度检测及编辑距离检测

Github：https://github.com/wangyang0922/BI/TextSimilarityDetection

### 2: [Santandery] 银行产品购买预测
主要工作：采用Item-based CF方法，对Santandery银行的用户产品购买数据进行分析，并对未来可能购买的产品进行预测

Github：https://github.com/wangyang0922/BI/Santandery

### 3: [Netflix] 电影推荐算法
主要工作：基于矩阵分解的协同过滤算法（ALS，SVD，SVD++，FunkSVD） 给Netflix网站进行推荐算法，RMSE降低到0.9111

Github：https://github.com/wangyang0922/BI/Netflix

###4: [Avazu-Ctr-Prediction] CTR广告点击率预测

主要工作：采用基于神经网络的DeepFM算法，对DSP公司Avazu的网站的广告转化率进行预测，项目中使用了线性模型及非线性模型，并进行了对比分析

Github：https://github.com/wangyang0922/BI/Avazu-Ctr-Prediction

### 5: [netflix] 房屋价格走势预测引擎
主要工作：通过时间序列算法，分析北京、上海、广州过去4年（2015.8-2019.12）的房屋历史价格，预测未来6个月（2020.1-2020.6）不同区的价格走势 

Github：https://github.com/xxx/House-Price-Prediction

### 6: [Email-Data-Analysis] 邮件数据分析
主要工作：通过PageRank算法分析邮件中的人物关系图谱，并针对邮件数量较大的情况筛选出重要的人物，进行绘制

Github：https://github.com/xxx/PageRank

### 7: [Movie-Actors] 电影数据集关联规则挖掘

主要技术：Python / apriori / one_hot_encoding 

电影数据集关联规则挖掘：采用Apriori算法，分析电影数据集中的导演和演员信息，从而发现导演和演员之间的频繁项集及关联规则：https://github.com/xxx/Apriori 

Github：https://github.com/wangyang0922/BI/MovieActors

### 8: [Email-Data-Analysis] 信用卡违约率分析
信用卡违约率分析：针对台湾某银行信用卡的数据，构建一个分析信用卡违约率的分类器。采用Random Forest算法，信用卡违约率识别率在80%左右：https://github.com/xxx/credit_default

### 9: [Email-Data-Analysis] 信用卡欺诈分析
信用卡欺诈分析：针对欧洲某银行信用卡交易数据，构建一个信用卡交易欺诈识别器。采用逻辑回归算法，通过数据可视化方式对混淆矩阵进行展示，统计模型的精确率，召回率和F1值，F1值为0.712，并绘制了精确率和召回率的曲线关系：https://github.com/xxx/credit_fraud

### 10: [Email-Data-Analysis] 比特币走势分析

比特币走势分析：分析2012年1月1日到2018年10月31日的比特币价格数据，并采用时间序列方法，构建自回归滑动平均模型（ARMA模型），预测未来8个月比特币的价格走势。预测结果表明比特币将在8个月内降低到4000美金左右，与实际比特币价格趋势吻合（实际最低降到4000美金以下）：https://github.com/xxx/bitcoin

 
