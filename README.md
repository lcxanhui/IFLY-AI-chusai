# 科大讯飞AI智能营销算法大赛
本次是第一次参加数据挖掘类比赛，虽然在初赛阶段只取得50多一点的名次，与复赛擦肩而过（复赛要求初赛排名前50名），但是作为新手，能够取得这样的成绩已经心满意足，在此分享本次参赛的具体代码，希望能给未来参加此种类型比赛的新手一些启迪：

运行环境：python 3.6.5               Anaconda-Jupyter notebook

必要的wheel：`lightgbm`    version:`2.2.0`     
		               
			       `xgboost`     version:0.8.0

具体比赛内容：[科大讯飞AI智能营销算法大赛--竞赛信息](http://www.pkbigdata.com/common/cmpt/2018%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9EAI%E8%90%A5%E9%94%80%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

比赛的baseline:
https://zhuanlan.zhihu.com/p/44956113

参考的比赛代码：

		IJCAI-18 阿里妈妈搜索广告转化预测
		
		2018腾讯广告算法大赛
		
		TalkingData AdTracking Fraud Detection Challenge--kaggle

备注：
		阿里妈妈的比赛代码主要参考rank29的代码
		
		腾讯广告算法大赛参考了liupengsay的代码rank11
		
		kaggle的比赛稍微浏览，没有借鉴太多的（也可能信息有疏漏，没有太多关注）

代码:
		1.ronghe.py      结合baseline，进行特征工程以及特征重要性选择（特征较多，大概暴力求解每次有2500-3000左右特征）
		
		2.ronghe2.py      利用选择的特征重要性得到结果（选择得分较高的前1100——1500（具体看运行结果）个特征）
		
		3.模型融合（代码暂时缺失）

具体得分：初赛线上0.42270左右


lightgbm的参数值已修改，请谨慎使用
