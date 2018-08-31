# -*- coding:utf-8 -*-

# step0:加载必要的模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets as datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# step1:数据获取

boston = datasets.load_boston()
train = boston.data
target = boston.target

X_train,x_test,y_train,y_true = train_test_split(train,target,test_size=0.2)
print(X_train,x_test)
print(y_train,y_true)

# step2: 初始化model
LR = LinearRegression()
RR= Ridge()
LLR = Lasso()
KNNR = KNeighborsRegressor()
DR = DecisionTreeRegressor()
SVMR = SVR()

# step3: Train and save and predict Model
# 这里省略了模型训练中评价和调参的过程
# 假设这就是美国房产中介Leader要我训练的
# 最终Model

LR.fit(X_train,y_train)
RR.fit(X_train,y_train)
LLR.fit(X_train,y_train)
KNNR.fit(X_train,y_train)
DR.fit(X_train,y_train)
SVMR.fit(X_train,y_train)


# 模型保存与持久化

joblib.dump(LR, "LR_model.m")
joblib.dump(RR, "RR_model.m")
joblib.dump(LLR, "LLR_model.m")
joblib.dump(KNNR, "KNNR_model.m")
joblib.dump(DR, "DR_model.m")
joblib.dump(SVMR, "SVMR_model.m")

# 模型加载
lr_m = joblib.load("LR_model.m")
rr_m = joblib.load("RR_model.m")
llr_m = joblib.load("LLR_model.m")
knnr_m = joblib.load("KNNR_model.m")
dr_m = joblib.load("DR_model.m")
svmr_m = joblib.load("SVMR_model.m")

y_LR = lr_m.predict(x_test)
y_RR = rr_m.predict(x_test)
y_LLR = llr_m.predict(x_test)
y_KNNR = knnr_m.predict(x_test)
y_DR = dr_m.predict(x_test)
y_SVMR = svmr_m.predict(x_test)


model_pre = pd.DataFrame({'LinearRegression()':list(y_LR),'Ridge()':list(y_RR),'Lasso()':list(y_LLR),
	'KNeighborsRegressor()':list(y_KNNR),'DecisionTreeRegressor()':list(y_DR),
	'SVR()':list(y_SVMR)})

# Plot

def model_plot(y_true,model_pre):
	'''
	y_true:真实的label
	model_pre: 预测的数据(数据框)
	'''
	cols = model_pre.columns
	plt.style.use("ggplot")
	plt.figure(figsize=(24,24))
	plt.rcParams['font.sans-serif'] = ['FangSong'] 
	plt.rcParams['axes.unicode_minus'] = False 
	for i in range(6):
		plt.subplot(2,3,i+1)
		plt.scatter(x=range(len(y_true)),y=y_true,label='true')
		plt.scatter(x=range(len(model_pre[cols[i]])),y=model_pre[cols[i]],label=cols[i])
		plt.title(str(cols[i])+':真实Label Vs 预测Label')
		plt.legend()

	plt.savefig("plot/model_plot.png")

model_plot(y_true, model_pre)