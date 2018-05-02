import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 

#计算风险值
def Risk(x,y):
    #先验概率
    p = [0.59,0.4,0.01]
    #求类条件概率
    #p = [1,1,1]
    #损失表
    loss = [[0,1,1],[3,0,1],[9,10,0]]
    #求最小错误率贝叶斯决策
    #loss = [[0,1,1],[1,0,1],[1,1,0]]
    #求后验概率
    y1_after = stats.norm.pdf(x,0,1) * stats.norm.pdf(y,0,1) * p[0]
    y2_after = stats.norm.pdf(x,4,0.8) * stats.norm.pdf(y,4,0.8) * p[1]
    y3_after = stats.norm.pdf(x,-5,1.1) * stats.norm.pdf(y,-5,1.1) * p[2]
    #求风险值
    R1 = loss[0][1]*y2_after + loss[0][2]*y3_after
    R2 = loss[1][0]*y1_after + loss[1][2]*y3_after
    R3 = loss[2][0]*y1_after + loss[2][1]*y2_after
    
    return R1,R2,R3


#生成数据集x1和类概率密度函数y1
x1 = np.random.randn(60,2)

x2 = 0.8 * np.random.randn(60,2) + 4

x3 = 1.1 * np.random.randn(60,2) - 5

#画出散点图
plt.plot(x1[:,0],x1[:,1],'+',color='b')
plt.plot(x2[:,0],x2[:,1],'+',color='y')
plt.plot(x3[:,0],x3[:,1],'+',color='r')




#画图：根据最小风险值分类
n = 0.2
x_ = np.arange(-9,8,n)
y_ = np.arange(-9,8,n)
for i in range(x_.shape[0]):
    for j in range(y_.shape[0]):
        R1,R2,R3 = Risk(x_[i],y_[j])
        if((R1<R2)&(R1<R3)):
            plt.scatter(x_[i],y_[j],color='b',marker=',',alpha=0.05)
        elif((R2<R1)&(R2<R3)):
            plt.scatter(x_[i],y_[j],color='y',marker=',',alpha=0.05)
        else:
            plt.scatter(x_[i],y_[j],color='r',marker=',',alpha=0.05)
            
