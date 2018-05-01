import numpy as np
import matplotlib.pyplot as plt
import random

#产生数据集
def genData(num,bias,target):
	x = np.zeros(shape=(num,3))
	y = np.zeros(shape=num)
	#x为数据集，y为标签集
	for i in range(0,num):
		x[i][0] = bias + random.uniform(0,1)
		x[i][1] = bias + random.uniform(0,1)
		x[i][2] = 1
		y[i] = target
	return x,y 

#感知器
#迭代次数x，安全裕量s，学习速率n
def perceptron(x,s,n,x1,y1,x2,y2):
	w = [[1],[1],[1]]
	w_all = np.zeros(shape=(x,3))#存储所有的权向量
	for i in range(x):
		m = i % (x1.shape[0]+x2.shape[0])
		#print(m)
		if(m < x1.shape[0]):
			a = np.dot(x1[m],w)*y1[m]
			if(a < s):
				for j in range(3):
					w[j][0] = w[j][0] + n * x1[m][j]*y1[m]
		else:
			a = np.dot(x2[m-x1.shape[0]], w)* y2[m-x1.shape[0]]
			if(a < s):
				for j in range(3):
					w[j][0] = w[j][0] + n * x2[m-x1.shape[0]][j]*y2[m-x1.shape[0]]
		w_all[i] = np.transpose(w)
	#画出分类效果图
	plt.scatter(x1[:,0],x1[:,1],color='b',marker='x') #x1的点
	plt.scatter(x2[:,0],x2[:,1],color='red',marker='x') #x2的点
	print(w)
	xi = np.arange(0,6,0.1)
	yi = -(w[0][0]/w[1][0]) * xi -(w[2][0]/w[1][0])
	plt.plot(xi,yi)
	plt.show()
	#绘制权向量迭代示意图
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for i in range(x-1):
		p = [w_all[i][0],w_all[i+1][0]]
		q = [w_all[i][1],w_all[i+1][1]]
		r = [w_all[i][2],w_all[i+1][2]]
		ax.scatter(w_all[i][0],w_all[i][1],w_all[i][2],marker='x')
		ax.plot(p,q,r)
	ax.text(w_all[0][0],w_all[0][1],w_all[0][2],'Start')
	ax.text(w_all[x-1][0],w_all[x-1][1],w_all[x-1][2],'Finish')
	plt.show()
   	
#基本参数设置
x = 500 #迭代次数
s = 0  #安全裕量
n = 0.01 #学习速率
#生成数据集:x为数据集，y为标签集
x1,y1 = genData(60,4,1)
x2,y2 = genData(60,5,-1)
#运行感知器
perceptron(x,s,n,x1,y1,x2,y2)
