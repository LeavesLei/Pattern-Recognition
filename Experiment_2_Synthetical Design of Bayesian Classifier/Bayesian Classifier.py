import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#导入样本
x_normal = [ -3.9847,-3.5549,-1.2401,-0.9780,-0.7932,-2.8531,-2.7605,-3.7287, 
-3.5414,-2.2692,-3.4549,-3.0752,-3.9934, -0.9780,-1.5799,-1.4885,
-0.7431,-0.4221,-1.1186,-2.3462,-1.0826,-3.4196,-1.3193,-0.8367,
-0.6579,-2.9683]

x_abnormal = [2.8792,0.7932,1.1882,3.0682,4.2532,0.3271,0.9846,2.7648,2.6588]

#计算样本的均值与标准差
x_n_ave = np.mean(x_normal)
x_a_ave = np.mean(x_abnormal)

x_n_std = np.std(x_normal)
x_a_std = np.std(x_abnormal)

#生成基于高斯分布的类概率密度曲线
x = np.arange(-6,7,0.0001)
y1 = stats.norm.pdf(x,x_n_ave,np.square(x_n_std))
y2 = stats.norm.pdf(x,x_a_ave,np.square(x_a_std))

plt.figure()
plt.plot(x,y1)
plt.plot(x,y2)

plt.title('Class probability density curve')
plt.xlabel('x')
plt.ylabel('P(x|w)')
plt.text(-1.8, 0.28,'P(x|normal)')
plt.text(2.6,0.26,'P(x|abnormal)')

#先验概率
p_1 = 0.9
p_2 = 0.1

#损失表
a_11 = 0
a_12 = 1
a_21 = 6
a_22 = 0

#求后验概率
y_n = y1 * 0.9 / (y1 * 0.9 + y2 * 0.1)
y_a = y2 * 0.1 / (y1 * 0.9 + y2 * 0.1)

#绘制后验概率曲线
plt.figure()
plt.plot(x,y_n)
plt.plot(x,y_a)

plt.title('Posterior probability density curve')
plt.xlabel('x')
plt.ylabel('P(w|x)')
plt.text(4, 0.05,'P(normal|x)')
plt.text(4,0.93,'P(abnormal|x)')
#计算条件风险
R1 = a_11 * y_n + a_12 * y_a 
R2 = a_21 * y_n + a_22 * y_a

#绘制条件风险曲线图
plt.figure()
plt.plot(x,R1)
plt.plot(x,R2) 

plt.title('Conditional risk curve')
plt.xlabel('x')
plt.ylabel('R')
plt.text(4, 1.2,'R(normal)')
plt.text(4,0.3,'R(abnormal)')

#求条件风险曲线的交点
for i in range(130000):
    if(np.abs(R1[i] - R2[i])<0.0001):
        plt.scatter(x[i],R1[i],color='b')
        print('---------------------------------------')
        print('decision boundary is : %.4f'%x[i])
        print('当观测到的特征值x > %.4f，则细胞会被判定为abnormal，若x <= %.4f 时，细胞被判定为normal' %(x[i],x[i]))
        print('---------------------------------------')
        break
    

    
    
    
    
    